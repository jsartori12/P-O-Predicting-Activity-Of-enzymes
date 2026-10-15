#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py

LoRA fine-tuning of ESM2 + MLP head on raw sequences. Same run-dir
convention as ESM2Classifier: runs/<run_name>/config.json,
train_log.jsonl (every log_every_steps), eval_metrics.jsonl (every
eval_every_steps, on a fixed val subset), checkpoints/best.pt + last.pt.

Resume after a SLURM time limit with --resume runs/<run_name>/checkpoints/last.pt
(submit_izar_lora.slurm does this automatically).
"""
import argparse
import dataclasses
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from config import Config
from data.sequence_dataset import SequenceDataset, get_dataloader
from model.lora_classifier import EsmLoraClassifier
from training.metrics import format_metrics
from training.trainer import compute_loss, evaluate, append_jsonl, save_checkpoint, load_checkpoint


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gpu_mem(device, context_gb=0.0):
    """GB. allocated = live tensors; reserved = what PyTorch's caching allocator holds
    (this is what runs out); needed = peak reserved + CUDA context = what the job
    needs from the GPU, i.e. what nvidia-smi shows at the peak."""
    if device.type != "cuda":
        return {}
    free, total = torch.cuda.mem_get_info(device)
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1e9
    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
        "reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "peak_reserved_gb": peak_reserved,
        "needed_gb": peak_reserved + context_gb,
        "gpu_total_gb": total / 1e9,
    }


def log_mem(path, stage, device, context_gb, **extra):
    m = gpu_mem(device, context_gb)
    if not m:
        return m
    append_jsonl(path, {"stage": stage, **m, **extra})
    print(f"[mem] {stage:18s} | allocated {m['allocated_gb']:6.2f} GB | reserved {m['reserved_gb']:6.2f} GB | "
          f"peak alloc {m['peak_allocated_gb']:6.2f} GB | needed {m['needed_gb']:6.2f} / "
          f"{m['gpu_total_gb']:.1f} GB", flush=True)
    return m


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None, help="path to a last.pt checkpoint")
    p.add_argument("--model-name", default=None, help="e.g. facebook/esm2_t6_8M_UR50D")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--lr-lora", type=float, default=None)
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--val-eval-samples", type=int, default=None)
    p.add_argument("--no-grad-checkpointing", action="store_true")
    p.add_argument("--patience", type=int, default=None,
                   help="early-stop patience in evals")
    p.add_argument("--no-early-stop", action="store_true",
                   help="train the full --epochs; best.pt still tracks the best val_loss")
    p.add_argument("--pooling",
                   choices=["mean", "attention", "max", "mean_max", "cls", "cls_mean"],
                   default=None,
                   help="how residue vectors are aggregated before the head")
    p.add_argument("--head-dim", type=int, default=None, help="width of the head's hidden layers")
    p.add_argument("--head-layers", type=int, default=None, help="number of hidden layers in the head")
    p.add_argument("--head-layer-norm", action="store_true")
    p.add_argument("--head-dropout", type=float, default=None)
    p.add_argument("--pos-weight", default=None,
                   help="BCE weight on active examples: a number, or 'auto' = n_not_active / n_active")
    return p.parse_args()


def apply_overrides(cfg, args):
    overrides = [
        (args.model_name, cfg.esm, "model_name"),
        (args.epochs, cfg.train, "epochs"),
        (args.batch_size, cfg.train, "batch_size"),
        (args.grad_accum, cfg.train, "grad_accum_steps"),
        (args.lr_lora, cfg.train, "lr_lora"),
        (args.eval_every, cfg.train, "eval_every_steps"),
        (args.max_train_samples, cfg.data, "max_train_samples"),
        (args.val_eval_samples, cfg.data, "val_eval_samples"),
    ]
    for value, section, name in overrides:
        if value is not None:
            setattr(section, name, value)
    if args.lora_r is not None:
        cfg.lora.r = args.lora_r
        cfg.lora.alpha = 2 * args.lora_r      # keep alpha/r = 2
    if args.no_grad_checkpointing:
        cfg.train.gradient_checkpointing = False
    if args.patience is not None:
        cfg.train.early_stop_patience = args.patience
    if args.no_early_stop:
        # keep the loop running to the last step; best.pt still follows val_loss
        cfg.train.early_stop_patience = 10**9
    if args.pooling is not None:
        cfg.head.pooling = args.pooling
    if args.head_dim is not None:
        cfg.head.hidden_dim = args.head_dim
    if args.head_layers is not None:
        cfg.head.n_layers = args.head_layers
    if args.head_dropout is not None:
        cfg.head.dropout = args.head_dropout
    if args.head_layer_norm:
        cfg.head.layer_norm = True


def main():
    args = parse_args()
    cfg = Config()
    apply_overrides(cfg, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = cfg.train.fp16 and device.type == "cuda"
    set_seed(cfg.train.seed)

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(run_dir, exist_ok=True)
    print(f"run dir: {run_dir} | device: {device} | fp16: {fp16}")

    # --- data ---------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.esm.model_name)
    train_ds = SequenceDataset(cfg.data.train_csv, cfg, max_samples=cfg.data.max_train_samples,
                               seed=cfg.train.seed)
    val_ds = SequenceDataset(cfg.data.val_csv, cfg, max_samples=cfg.data.val_eval_samples,
                             seed=cfg.train.seed)
    train_loader, train_sampler = get_dataloader(train_ds, tokenizer, cfg, cfg.train.batch_size, shuffle=True)
    val_loader, _ = get_dataloader(val_ds, tokenizer, cfg, cfg.train.batch_size * 4, shuffle=False)

    batches_per_epoch = len(train_sampler)
    steps_per_epoch = math.ceil(batches_per_epoch / cfg.train.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.train.epochs
    print(f"train: {len(train_ds):,} seqs ({train_ds.labels.mean():.3f} active) | "
          f"val subset: {len(val_ds):,} seqs ({val_ds.labels.mean():.3f} active)")
    print(f"{batches_per_epoch:,} batches/epoch | {steps_per_epoch:,} optimizer steps/epoch | "
          f"{total_steps:,} total | effective batch {cfg.train.batch_size * cfg.train.grad_accum_steps}")

    # resolved here (not in apply_overrides) because 'auto' needs the train labels;
    # deterministic, so a resumed run with the same args gets the same value
    if args.pos_weight == "auto":
        n_pos = float(train_ds.labels.sum())
        cfg.train.pos_weight = (len(train_ds) - n_pos) / n_pos
    elif args.pos_weight is not None:
        cfg.train.pos_weight = float(args.pos_weight)
    pos_weight = None
    if cfg.train.pos_weight is not None:
        pos_weight = torch.tensor([cfg.train.pos_weight], device=device)
        print(f"weighted BCE: pos_weight = {cfg.train.pos_weight:.4f} on active examples (train loss only)")

    if args.resume is None:
        with open(os.path.join(run_dir, "config.json"), "w") as fh:
            json.dump(dataclasses.asdict(cfg), fh, indent=2)

    # --- model / optim ----------------------------------------------------------
    mem_log_path = os.path.join(run_dir, "memory.jsonl")
    context_gb = 0.0
    if device.type == "cuda":
        torch.cuda.init()
        free, total = torch.cuda.mem_get_info(device)
        context_gb = (total - free) / 1e9    # driver/cuBLAS/cuDNN, outside PyTorch's allocator
        print(f"[mem] {torch.cuda.get_device_name(device)} | {total / 1e9:.1f} GB total | "
              f"CUDA context {context_gb:.2f} GB")

    model = EsmLoraClassifier(cfg).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {n_train:,} / {n_total:,} ({100 * n_train / n_total:.2f}%)")
    log_mem(mem_log_path, "model_loaded", device, context_gb,
            params_total=n_total, params_trainable=n_train,
            grad_checkpointing=cfg.train.gradient_checkpointing, batch_size=cfg.train.batch_size)

    optimizer = AdamW(model.param_groups(cfg))
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.train.warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    global_step, start_epoch, start_batch = 0, 0, 0
    best_val_loss, evals_without_improvement = float("inf"), 0
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        global_step = state["global_step"]
        start_epoch, start_batch = state["epoch"], state["batch_in_epoch"]
        best_val_loss = state["best_val_loss"]
        evals_without_improvement = state["evals_without_improvement"]
        print(f"resumed from {args.resume}: step {global_step}, epoch {start_epoch}, "
              f"batch {start_batch}, best val_loss {best_val_loss:.4f}")

    train_log_path = os.path.join(run_dir, "train_log.jsonl")
    eval_log_path = os.path.join(run_dir, "eval_metrics.jsonl")

    # --- loop ---------------------------------------------------------------
    model.train()
    stop = False
    running_loss, running_n = 0.0, 0
    max_tokens_window, max_tokens_run = 0, 0
    first_step_logged = False
    t_log = time.time()

    for epoch in range(start_epoch, cfg.train.epochs):
        skip = start_batch if epoch == start_epoch else 0
        train_sampler.set_epoch(epoch, skip_batches=skip)
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_loader, start=skip):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            max_tokens_window = max(max_tokens_window, input_ids.size(1))

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
                logits = model(input_ids, attention_mask)
            loss = compute_loss(logits, labels, pos_weight)
            scaler.scale(loss / cfg.train.grad_accum_steps).backward()
            running_loss += loss.item()
            running_n += 1

            last_batch = i + 1 == batches_per_epoch
            if (i + 1) % cfg.train.grad_accum_steps != 0 and not last_batch:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.train.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if not first_step_logged:
                # weights + autocast copies + activations of one step + AdamW state (created lazily here)
                log_mem(mem_log_path, "after_first_step", device, context_gb,
                        step=global_step, batch_tokens=max_tokens_window)
                first_step_logged = True

            if global_step % cfg.train.log_every_steps == 0:
                mem = gpu_mem(device, context_gb)
                max_tokens_run = max(max_tokens_run, max_tokens_window)
                elapsed = time.time() - t_log
                record = {
                    "step": global_step,
                    "epoch": epoch + (i + 1) / batches_per_epoch,
                    "train_loss": running_loss / running_n,
                    "lr_lora": scheduler.get_last_lr()[0],
                    "lr_head": scheduler.get_last_lr()[1],
                    "grad_norm": float(grad_norm),
                    "sec_per_step": elapsed / cfg.train.log_every_steps,
                    # peaks are since the start of the run (or resume); they grow until the
                    # longest batch has been seen -- max_batch_tokens tracks that
                    "gpu_mem_allocated_gb": mem.get("allocated_gb", 0.0),
                    "gpu_mem_peak_gb": mem.get("peak_allocated_gb", 0.0),
                    "gpu_mem_peak_reserved_gb": mem.get("peak_reserved_gb", 0.0),
                    "gpu_mem_needed_gb": mem.get("needed_gb", 0.0),
                    "max_batch_tokens_window": max_tokens_window,
                    "max_batch_tokens_run": max_tokens_run,
                }
                append_jsonl(train_log_path, record)
                eta_h = record["sec_per_step"] * (total_steps - global_step) / 3600
                print(f"step {global_step}/{total_steps} | epoch {record['epoch']:.3f} | "
                      f"train loss {record['train_loss']:.4f} | lr {record['lr_lora']:.2e} | "
                      f"{record['sec_per_step']:.2f}s/step | eta {eta_h:.1f}h | "
                      f"mem now {record['gpu_mem_allocated_gb']:.1f} / peak {record['gpu_mem_peak_gb']:.1f} / "
                      f"needed {record['gpu_mem_needed_gb']:.1f} GB (longest batch so far "
                      f"{max_tokens_run} tok)", flush=True)
                running_loss, running_n = 0.0, 0
                max_tokens_window = 0
                t_log = time.time()

            if global_step % cfg.train.eval_every_steps == 0 or global_step == total_steps:
                t0 = time.time()
                val_loss, val_metrics = evaluate(model, val_loader, device, fp16)
                val_time = time.time() - t0
                print(f"  eval @ step {global_step} | val loss {val_loss:.4f} | "
                      f"{format_metrics(val_metrics)} ({val_time:.0f}s)", flush=True)
                append_jsonl(eval_log_path, {
                    "step": global_step,
                    "epoch": epoch + (i + 1) / batches_per_epoch,
                    "val_loss": val_loss,
                    "val_time_sec": val_time,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    evals_without_improvement = 0
                    save_checkpoint(os.path.join(ckpt_dir, "best.pt"), model,
                                    global_step=global_step, val_loss=val_loss)
                    print(f"  -> new best, saved {ckpt_dir}/best.pt")
                else:
                    evals_without_improvement += 1

                next_epoch, next_batch = (epoch + 1, 0) if last_batch else (epoch, i + 1)
                save_checkpoint(os.path.join(ckpt_dir, "last.pt"), model, optimizer, scheduler, scaler,
                                global_step=global_step, epoch=next_epoch, batch_in_epoch=next_batch,
                                best_val_loss=best_val_loss,
                                evals_without_improvement=evals_without_improvement)

                t_log += time.time() - t0   # keep eval time out of the next sec_per_step

                if evals_without_improvement >= cfg.train.early_stop_patience:
                    print(f"early stopping: val_loss hasn't improved in {cfg.train.early_stop_patience} "
                          f"evals (best {best_val_loss:.4f})")
                    stop = True
                    break
        if stop:
            break

    log_mem(mem_log_path, "end_of_training", device, context_gb,
            step=global_step, max_batch_tokens=max(max_tokens_run, max_tokens_window))
    print(f"done. best val_loss (subset) {best_val_loss:.4f}. "
          f"Full-split evaluation: python3 evaluate.py --run-dir {run_dir}")


if __name__ == "__main__":
    main()
