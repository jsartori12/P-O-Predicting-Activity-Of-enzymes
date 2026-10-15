#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_memory.py

Worst-case GPU memory for one LoRA training step: synthetic batches of random
residues at the longest length (max_seq_len tokens), one forward + backward +
optimizer step each, peak memory reported per batch size. No data, no
checkpoints, nothing saved -- only answers "does this config fit on the GPU".

Real batches are length-grouped (median ~340 residues), so typical steps use
far less than this; the worst case is what decides OOM.

    python3 probe_memory.py                         # 650M, batch sizes 1 2 4 8 16
    python3 probe_memory.py --batch-sizes 8 12 --no-grad-checkpointing
    python3 probe_memory.py --batch-sizes 8 --snapshot mem.pickle   # drag into https://pytorch.org/memory_viz

Three numbers per batch size:
  allocated  peak bytes held by live tensors (what the model actually uses)
  reserved   peak bytes PyTorch's caching allocator grabbed from the driver
             (allocated + fragmentation/cache) -- this is what hits OOM
  needed     reserved + CUDA context (driver/cuBLAS/cuDNN, outside PyTorch),
             i.e. what nvidia-smi would show; compare this against the GPU size
"""
import argparse

import torch
from torch.optim import AdamW

from config import Config
from model.lora_classifier import EsmLoraClassifier
from training.trainer import compute_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default=None)
    p.add_argument("--seq-len", type=int, default=None, help="tokens incl. <cls>/<eos>; default max_seq_len")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--no-grad-checkpointing", action="store_true")
    p.add_argument("--snapshot", default=None, help="write a per-allocation memory snapshot (.pickle)")
    args = p.parse_args()

    cfg = Config()
    if args.model_name:
        cfg.esm.model_name = args.model_name
    if args.lora_r:
        cfg.lora.r, cfg.lora.alpha = args.lora_r, 2 * args.lora_r
    if args.no_grad_checkpointing:
        cfg.train.gradient_checkpointing = False
    seq_len = args.seq_len or cfg.esm.max_seq_len

    device = torch.device("cuda")
    total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f"{torch.cuda.get_device_name(device)} | {total_gb:.1f} GB | {cfg.esm.model_name} | "
          f"seq_len {seq_len} | grad checkpointing {cfg.train.gradient_checkpointing} | fp16 {cfg.train.fp16}")

    model = EsmLoraClassifier(cfg).to(device)
    # from_pretrained() leaves the ESM layers in eval mode, and HF only checkpoints
    # layers in training mode -- without this the probe silently measures no-checkpointing
    # (and no LoRA dropout), unlike train.py, which calls model.train() before its loop
    model.train()
    optimizer = AdamW(model.param_groups(cfg))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.fp16)
    torch.cuda.synchronize()
    base_gb = torch.cuda.memory_allocated() / 1e9
    free, total = torch.cuda.mem_get_info()
    context_gb = (total - free - torch.cuda.memory_reserved()) / 1e9
    print(f"model weights on GPU: {base_gb:.2f} GB | CUDA context (outside PyTorch): {context_gb:.2f} GB")

    if args.snapshot:
        torch.cuda.memory._record_memory_history(max_entries=200000)

    for bs in args.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # token ids 4..23 = the 20 standard amino acids in the ESM vocab; 0 = <cls>, 2 = <eos>
        input_ids = torch.randint(4, 24, (bs, seq_len), device=device)
        input_ids[:, 0], input_ids[:, -1] = 0, 2
        attention_mask = torch.ones_like(input_ids)
        labels = torch.randint(0, 2, (bs,), device=device).float()
        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.train.fp16):
                logits = model(input_ids, attention_mask)
            scaler.scale(compute_loss(logits, labels)).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            alloc = torch.cuda.max_memory_allocated() / 1e9
            reserved = torch.cuda.max_memory_reserved() / 1e9
            needed = reserved + context_gb
            print(f"batch {bs:3d} x {seq_len} tokens: allocated {alloc:5.2f} GB | reserved {reserved:5.2f} GB | "
                  f"needed {needed:5.2f} GB ({100 * needed / total_gb:.0f}% of GPU)")
        except torch.cuda.OutOfMemoryError:
            print(f"batch {bs:3d} x {seq_len} tokens: OOM")
            optimizer.zero_grad(set_to_none=True)
            break

    if args.snapshot:
        torch.cuda.memory._dump_snapshot(args.snapshot)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"snapshot written to {args.snapshot} -- open it at https://pytorch.org/memory_viz")


if __name__ == "__main__":
    main()
