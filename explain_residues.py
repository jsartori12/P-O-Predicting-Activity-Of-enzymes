#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explain_residues.py

Per-residue attribution ("where is the model looking?") for the enzyme
active/not-active classifiers, plus a quantitative test against the known
catalytic sites.

WHY NOT LITERAL GRAD-CAM
------------------------
Grad-CAM weights feature maps by the spatially-averaged gradient. Here the
pooling is a plain mean over residues:

    pooled = (1/N) * sum_l h[l]          h: (B, L, 1280)

so d(logit)/d(h[l]) is IDENTICAL for every residue l -- the gradient alone
carries no positional information and a literal Grad-CAM map would be flat.
The informative quantity is gradient x activation:

    contribution(l) = g . h[l] / N       g = d(logit)/d(pooled)

which is not an approximation but an EXACT additive decomposition of the
pre-head pooled vector: sum_l contribution(l) = g . pooled. It is first-order
with respect to the non-linear head, so `--method intgrad` is also provided,
integrating gradients along a path from a baseline to the real input.

LAYERS
  --layers last  attribute only at the final hidden state (default)
  --layers all   attribute at every one of the 34 hidden states (embeddings + 33
                 blocks), giving a layer x position map per sequence.
                 The exact-decomposition property above holds at the LAST layer
                 only. At an intermediate layer k the logit depends on h_k through
                 all the blocks above it, so d(logit)/d(h_k[l]) varies with the
                 position l (attention mixes positions) and grad x activation is a
                 first-order Taylor attribution, not an exact split. It still
                 answers "which residues at depth k move the logit", which is the
                 question -- but it is an approximation and should be read as one.

METHODS
  gradcam   g . h[l] / N at the last hidden state. One backward pass. Default.
  intgrad   integrated gradients on the input embeddings, `--steps` passes.
            Slower, but accounts for the head's non-linearity and for the
            backbone's own mixing (relevant for LoRA, whose backbone adapted).

OUTPUT
  <out>/attributions.csv   id, position, residue, attribution, is_catalytic
  <out>/summary.json       enrichment of attribution at catalytic sites

Usage:
  python3 explain_residues.py --run lora_r8 --fasta seqs.fasta --out xai_lora_r8
  python3 explain_residues.py --run lora_r8 --catalytic-sites <...>/catalytic_sites.csv \\
          --max-seqs 200 --method intgrad --out xai_lora_r8
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from config import Config                                  # noqa: E402
from model.lora_classifier import EsmLoraClassifier   # noqa: E402
from training.trainer import load_checkpoint               # noqa: E402


def load_model(run_dir, ckpt, device):
    cfg = Config()
    with open(os.path.join(run_dir, "config.json")) as fh:
        for section, values in json.load(fh).items():
            for k, v in values.items():
                setattr(getattr(cfg, section), k, v)
    cfg.train.gradient_checkpointing = False       # gradients must flow normally
    model = EsmLoraClassifier(cfg).to(device).eval()
    load_checkpoint(os.path.join(run_dir, ckpt), model)
    return model, cfg


def attribution_gradcam(model, input_ids, attention_mask, all_layers=False):
    """Per-residue attribution by gradient x activation.

    all_layers=False -> (B, 1, L), the final hidden state only. Exact: the
    contributions sum to g . pooled.
    all_layers=True  -> (B, 34, L), every hidden state. Layer 0 is the token
    embedding, 1..33 the transformer blocks. Only the last row is exact; the
    others are first-order (see LAYERS in the module docstring).
    """
    # The backbone weights are frozen and input_ids are integers, so the embedding
    # output would have requires_grad=False and retain_grad() would fail on it.
    # Feeding inputs_embeds with requires_grad makes the whole chain differentiable
    # and gives an attribution for hidden state 0 as well.
    emb = model.backbone.get_input_embeddings()(input_ids).detach().requires_grad_(True)
    out = model.backbone(inputs_embeds=emb, attention_mask=attention_mask,
                         output_hidden_states=all_layers)
    hidden = out.last_hidden_state
    states = list(out.hidden_states) if all_layers else [hidden]
    for h in states:
        if h.requires_grad:      # keep grads on non-leaf intermediates
            h.retain_grad()
    # the run's own pooling (mean / attention / max / mean_max / cls / cls_mean).
    # The exact-decomposition argument above holds for mean pooling only; for the
    # others last-layer grad x activation is first-order like the intermediate layers
    # (and for cls it is zero on every residue -- only <cls> reaches the head).
    pooled = model.pool(hidden, attention_mask)
    logit = model.head(pooled.float()).squeeze(-1)
    model.zero_grad(set_to_none=True)
    logit.sum().backward()
    # one backward pass populates .grad on every retained state at once
    # a state with no gradient (nothing upstream required grad) contributes nothing
    contrib = torch.stack([(h.grad * h).sum(-1) if h.grad is not None
                           else torch.zeros(h.shape[:2], device=h.device)
                           for h in states], dim=1)                        # (B, n, L)
    return contrib.detach(), logit.detach()


def attribution_intgrad(model, input_ids, attention_mask, steps=32):
    """Integrated gradients on the input embeddings, baseline = all-<pad>
    embedding. Captures the head's non-linearity and the backbone's mixing."""
    emb_layer = model.backbone.get_input_embeddings()
    x = emb_layer(input_ids).detach()                      # (B, L, D)
    baseline = torch.zeros_like(x)
    total = torch.zeros(x.shape[:2], device=x.device)
    for a in torch.linspace(1.0 / steps, 1.0, steps, device=x.device):
        xi = (baseline + a * (x - baseline)).requires_grad_(True)
        hidden = model.backbone(inputs_embeds=xi,
                                attention_mask=attention_mask).last_hidden_state
        logit = model.head(model.pool(hidden, attention_mask).float()).squeeze(-1)
        model.zero_grad(set_to_none=True)
        g, = torch.autograd.grad(logit.sum(), xi)
        total += (g * (x - baseline)).sum(-1)
    return (total / steps).detach(), logit.detach()


def read_fasta(path):
    seqs, rid, chunks = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if rid is not None:
                    seqs[rid] = "".join(chunks)
                rid, chunks = line[1:].split()[0], []
            elif line:
                chunks.append(line)
    if rid is not None:
        seqs[rid] = "".join(chunks)
    return seqs


def parse_act_sites(x):
    """UniProt 'ACT_SITE_list' is a space/semicolon separated list of 1-based positions."""
    if not isinstance(x, str):
        return []
    return [int(t) for t in x.replace(";", " ").split() if t.strip().isdigit()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run dir name under this directory or runs/")
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--fasta", default=None, help="sequences to explain")
    p.add_argument("--catalytic-sites", default=None,
                   help="catalytic_sites.csv; supplies sequences AND the ground truth")
    p.add_argument("--max-seqs", type=int, default=200)
    p.add_argument("--method", choices=["gradcam", "intgrad"], default="gradcam")
    p.add_argument("--layers", choices=["last", "all"], default="last",
                   help="'all' produces a layer x position map (gradcam only)")
    p.add_argument("--steps", type=int, default=32, help="intgrad only")
    p.add_argument("--max-len", type=int, default=1022)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    run_dir = next((d for d in (os.path.join(HERE, args.run), os.path.join(HERE, "runs", args.run))
                    if os.path.isfile(os.path.join(d, "config.json"))), None)
    if run_dir is None:
        raise FileNotFoundError(f"run '{args.run}' not found under {HERE}/ or {HERE}/runs/")
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- sequences + ground truth ------------------------------------------
    truth = {}
    if args.catalytic_sites:
        cs = pd.read_csv(args.catalytic_sites,
                         usecols=["Entry", "Sequence", "ACT_SITE_list"], low_memory=False)
        cs = cs[cs.ACT_SITE_list.notna() & cs.Sequence.notna()]
        cs = cs[cs.Sequence.str.len() <= args.max_len].head(args.max_seqs)
        seqs = dict(zip(cs.Entry, cs.Sequence))
        truth = {e: parse_act_sites(a) for e, a in zip(cs.Entry, cs.ACT_SITE_list)}
    elif args.fasta:
        seqs = dict(list(read_fasta(args.fasta).items())[:args.max_seqs])
    else:
        p.error("give --fasta or --catalytic-sites")
    print(f"{len(seqs)} sequences | method={args.method} | device={device}")

    model, cfg = load_model(run_dir, args.ckpt, device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.esm.model_name)

    rows = []
    for i, (sid, seq) in enumerate(seqs.items(), 1):
        enc = tok(seq, return_tensors="pt", truncation=True, max_length=args.max_len + 2).to(device)
        if args.method == "gradcam":
            attr, logit = attribution_gradcam(model, enc["input_ids"], enc["attention_mask"],
                                              all_layers=(args.layers == "all"))
        else:
            attr, logit = attribution_intgrad(model, enc["input_ids"], enc["attention_mask"],
                                              args.steps)
            attr = attr.unsqueeze(1)
        a = attr[0].float().cpu().numpy()[:, 1:len(seq) + 1]   # (n_layers, L) drop <cls>/<eos>
        n_layers = a.shape[0]
        cat = set(truth.get(sid, []))
        p_act = float(torch.sigmoid(logit)[0])
        attn_w = None
        if model.attn_pool is not None:
            with torch.no_grad():
                h = model.backbone(input_ids=enc["input_ids"],
                                   attention_mask=enc["attention_mask"]).last_hidden_state
                attn_w = model.attn_pool.weights(h, enc["attention_mask"])[0].float().cpu().numpy()[1:len(seq) + 1]
        for li in range(n_layers):
            # layer index: -1 when only the final state was computed
            layer_id = li if args.layers == "all" else -1
            for pos, (res, val) in enumerate(zip(seq, a[li]), start=1):
                row = {"id": sid, "layer": layer_id, "position": pos, "residue": res,
                       "attribution": float(val), "is_catalytic": pos in cat,
                       "p_active": p_act}
                if attn_w is not None:
                    row["attn_weight"] = float(attn_w[pos - 1])
                rows.append(row)
        if i % 25 == 0:
            print(f"  {i}/{len(seqs)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "attributions.csv"), index=False)

    # --- does the model look at the catalytic residues? --------------------
    summary = {"run": args.run, "method": args.method, "pooling": model.pooling,
               "n_sequences": len(seqs), "n_residues": len(df)}
    if truth:
        # rank each residue within its own sequence AND layer, so both are comparable
        df["pct_rank"] = df.groupby(["id", "layer"]).attribution.rank(pct=True)
        per_layer = []
        for layer, sub in df.groupby("layer"):
            c = sub[sub.is_catalytic]
            per_layer.append({
                "layer": int(layer),
                "mean_pct_rank_catalytic": round(float(c.pct_rank.mean()), 4),
                "mean_pct_rank_background": round(float(sub[~sub.is_catalytic].pct_rank.mean()), 4),
                # 0.10 of catalytic residues land in the top 10% by chance alone
                "catalytic_in_top10pct": round(float((c.pct_rank >= 0.90).mean()), 4),
                "enrichment_top10pct": round(float((c.pct_rank >= 0.90).mean()) / 0.10, 3),
            })
        summary["n_catalytic_residues"] = int(df.is_catalytic.sum() / df.layer.nunique())
        summary["expected_by_chance"] = 0.10
        summary["per_layer"] = per_layer
        if "attn_weight" in df:
            # the AttentionPool weights themselves -- no gradient involved
            w = df[df.layer == df.layer.max()].copy()
            w["w_rank"] = w.groupby("id").attn_weight.rank(pct=True)
            c = w[w.is_catalytic]
            summary["attention_pool"] = {
                "mean_pct_rank_catalytic": round(float(c.w_rank.mean()), 4),
                "catalytic_in_top10pct": round(float((c.w_rank >= 0.90).mean()), 4),
                "enrichment_top10pct": round(float((c.w_rank >= 0.90).mean()) / 0.10, 3),
                "weight_share_on_catalytic": round(float(c.attn_weight.sum() / w.attn_weight.sum()), 5),
                "uniform_share_on_catalytic": round(len(c) / len(w), 5),
            }
        best = max(per_layer, key=lambda r: r["enrichment_top10pct"])
        summary["best_layer"] = best["layer"]
        summary["best_enrichment"] = best["enrichment_top10pct"]
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nwritten to {args.out}/")


if __name__ == "__main__":
    main()
