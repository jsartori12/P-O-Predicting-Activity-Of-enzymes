#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py

Full-val-split evaluation of runs/<run_name>/checkpoints/best.pt (training
only evaluates a fixed val subset). Writes to the run dir:
  final_eval.json   overall metrics + per-source_file breakdown
  predictions.csv   id, source_file, label, p_active

Per-source matters here: every active is a SwissProt sequence and every
not_active is a 5A/10A/15A mutant or decoy, so a model could score well by
telling natural from engineered sequences rather than active from inactive.
"""
import argparse
import json
import os

import pandas as pd
import torch
from transformers import AutoTokenizer

from config import Config
from data.sequence_dataset import SequenceDataset, get_dataloader
from model.lora_classifier import EsmLoraClassifier
from training.metrics import compute_classification_metrics, format_metrics
from training.trainer import predict, per_source_metrics, load_checkpoint


def load_run_config(run_dir):
    cfg = Config()
    with open(os.path.join(run_dir, "config.json")) as fh:
        saved = json.load(fh)
    for section, values in saved.items():
        for k, v in values.items():
            setattr(getattr(cfg, section), k, v)
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default=None, help="default: <run-dir>/checkpoints/best.pt")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    cfg = load_run_config(args.run_dir)
    cfg.train.gradient_checkpointing = False     # inference only
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = cfg.train.fp16 and device.type == "cuda"

    ckpt_path = args.checkpoint or os.path.join(args.run_dir, "checkpoints", "best.pt")
    model = EsmLoraClassifier(cfg).to(device)
    state = load_checkpoint(ckpt_path, model)
    print(f"loaded {ckpt_path} (step {state.get('global_step')}, subset val_loss {state.get('val_loss')})")

    tokenizer = AutoTokenizer.from_pretrained(cfg.esm.model_name)
    val_ds = SequenceDataset(cfg.data.val_csv, cfg)
    val_loader, _ = get_dataloader(val_ds, tokenizer, cfg, args.batch_size, shuffle=False)
    print(f"full val split: {len(val_ds):,} sequences")

    loss, labels, probs, idx = predict(model, val_loader, device, fp16)
    metrics = compute_classification_metrics(labels, probs)
    sources = [val_ds.sources[i] for i in idx]
    by_source = per_source_metrics(labels, probs, sources)

    print(f"val loss {loss:.4f} | {format_metrics(metrics)}")
    for src, m in by_source.items():
        print(f"  {src:32s} n={m['n']:6d} label={m['label']} acc={m['accuracy']:.4f} "
              f"mean_p_active={m['mean_p_active']:.4f}")

    with open(os.path.join(args.run_dir, "final_eval.json"), "w") as fh:
        json.dump({"checkpoint": ckpt_path, "val_loss": loss, **{f"val_{k}": v for k, v in metrics.items()},
                   "per_source": by_source}, fh, indent=2)
    pd.DataFrame({
        "id": [val_ds.ids[i] for i in idx],
        "source_file": sources,
        "label": labels.astype(int),
        "p_active": probs,
    }).to_csv(os.path.join(args.run_dir, "predictions.csv"), index=False)
    print(f"written {args.run_dir}/final_eval.json and predictions.csv")


if __name__ == "__main__":
    main()
