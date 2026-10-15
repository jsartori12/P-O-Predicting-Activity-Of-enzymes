#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training/trainer.py

Evaluation, logging and checkpoint helpers for the LoRA run. The training
loop itself lives in train.py (step-based, with gradient accumulation and
fp16 autocast).
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from training.metrics import compute_classification_metrics


def compute_loss(logits, labels, pos_weight=None):
    """pos_weight (1-element tensor) multiplies the loss of positive (active)
    examples; None = plain BCE. Only the training loss is weighted -- predict()
    stays unweighted so val_loss is comparable across runs."""
    return F.binary_cross_entropy_with_logits(logits.float(), labels, pos_weight=pos_weight)


@torch.no_grad()
def predict(model, loader, device, fp16):
    """Returns (mean loss, labels, probs, dataset indices) over the whole loader."""
    model.eval()
    total_loss, n = 0.0, 0
    all_labels, all_probs, all_idx = [], [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
            logits = model(input_ids, attention_mask)
        loss = compute_loss(logits, labels)

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        all_labels.append(labels.cpu())
        all_probs.append(torch.sigmoid(logits.float()).cpu())
        all_idx.append(batch["idx"])

    model.train()
    return (total_loss / n,
            torch.cat(all_labels).numpy(),
            torch.cat(all_probs).numpy(),
            torch.cat(all_idx).numpy())


def evaluate(model, loader, device, fp16):
    loss, labels, probs, _ = predict(model, loader, device, fp16)
    return loss, compute_classification_metrics(labels, probs)


def per_source_metrics(labels, probs, sources):
    """Metrics per source_file. Each source holds a single class, so only
    accuracy (= recall for that class) and mean P(active) are meaningful."""
    sources = np.asarray(sources)
    out = {}
    for src in sorted(set(sources)):
        m = sources == src
        preds = (probs[m] >= 0.5).astype(int)
        out[src] = {
            "n": int(m.sum()),
            "label": int(labels[m][0]),
            "accuracy": float((preds == labels[m]).mean()),
            "mean_p_active": float(probs[m].mean()),
        }
    return out


def append_jsonl(path, record):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def save_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, **state):
    """best.pt: weights only. last.pt: weights + optimizer/scheduler/scaler +
    loop state, enough to resume after a SLURM time limit."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {"model": model.trainable_state_dict(), **state}
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
        ckpt["scheduler"] = scheduler.state_dict()
        ckpt["scaler"] = scaler.state_dict()
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)   # never leave a half-written last.pt if the job is killed mid-save


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert not unexpected and not (trainable & set(missing)), \
        f"checkpoint doesn't match model: missing={trainable & set(missing)} unexpected={unexpected}"
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt
