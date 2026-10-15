#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_lora_results.py

Four-panel summary of a LoRA run (default lora_r8):
  a) train loss vs val loss (subset) per step, best eval + early stop marked
  b) val MCC / F1 / ROC-AUC (subset) per eval
  c) full-val-split MCC across every approach so far
  d) per-source accuracy on the full val split, pilot vs this run

    python3 plot_lora_results.py [run_dir]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # categorical slots 1-3, fixed order
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#8a8a86"
SURFACE = "#fcfcfb"

# full val split (103,420 seqs); earlier runs from their final evaluations
BENCHMARK = [
    ("From-scratch encoder\n(best, d_ff=1024)", 0.293),
    ("ESM2 frozen + MLP v1", 0.700),
    ("ESM2 frozen + MLP v2", 0.713),
    ("ESM2 + LoRA pilot\n(20k seqs)", 0.902),
]
# runs/lora_pilot/final_eval.json (Izar, 2026-09-25)
PILOT_PER_SOURCE = {
    "sequencias_swissprot.fasta": 0.9399, "5A_cata_mut.fasta": 0.9522, "10A_cata_mut.fasta": 0.9855,
    "15A_cata_mut.fasta": 0.9922, "decoys_cataliticals.fasta": 0.8988,
}
SOURCE_LABELS = [
    ("sequencias_swissprot.fasta", "SwissProt\n(active)"),
    ("5A_cata_mut.fasta", "5A mut"),
    ("10A_cata_mut.fasta", "10A mut"),
    ("15A_cata_mut.fasta", "15A mut"),
    ("decoys_cataliticals.fasta", "decoys\ncatalíticos"),
]


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(True, color=MUTED, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=8.5)


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "lora_r8"
    run = os.path.basename(os.path.normpath(run_dir))
    train = [json.loads(l) for l in open(os.path.join(run_dir, "train_log.jsonl"))]
    evals = [json.loads(l) for l in open(os.path.join(run_dir, "eval_metrics.jsonl"))]
    final = json.load(open(os.path.join(run_dir, "final_eval.json")))
    best = min(evals, key=lambda e: e["val_loss"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=SURFACE)
    (a, b), (c, d) = axes

    # a) losses
    style(a, "a) Loss (val = fixed 20k subset)")
    a.plot([t["step"] for t in train], [t["train_loss"] for t in train], color=SERIES[0], lw=2, label="train")
    a.plot([e["step"] for e in evals], [e["val_loss"] for e in evals], color=SERIES[1], lw=2,
           marker="o", ms=8, mec=SURFACE, mew=2, label="val")
    a.set_yscale("log")
    a.annotate(f"best {best['val_loss']:.4f}\n(step {best['step']:,})", (best["step"], best["val_loss"]),
               xytext=(12, -28), textcoords="offset points", fontsize=8.5, color=INK_2,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    last = evals[-1]["step"]
    a.axvline(last, color=MUTED, lw=1, ls=":")
    a.text(last, a.get_ylim()[1], " early stop", color=INK_2, fontsize=8.5, va="top", ha="right")
    a.set_xlabel("optimizer step (effective batch 32)", color=INK_2, fontsize=9)
    a.set_ylabel("BCE loss (log scale)", color=INK_2, fontsize=9)
    a.legend(frameon=False, fontsize=9)

    # b) val metrics
    style(b, "b) Val metrics (20k subset)")
    for (key, label), color in zip([("val_mcc", "MCC"), ("val_f1", "F1"), ("val_roc_auc", "ROC-AUC")], SERIES):
        ys = [e[key] for e in evals]
        b.plot([e["step"] for e in evals], ys, color=color, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label=label)
        b.annotate(f"{label} {ys[-1]:.3f}", (evals[-1]["step"], ys[-1]), xytext=(6, 0),
                   textcoords="offset points", fontsize=8.5, color=INK_2, va="center")
    b.axvline(best["step"], color=MUTED, lw=1, ls=":")
    b.set_xlim(right=evals[-1]["step"] * 1.13)
    b.set_xlabel("optimizer step", color=INK_2, fontsize=9)
    b.legend(frameon=False, fontsize=9, loc="lower center", ncol=3)

    # c) benchmark
    style(c, "c) MCC on the full val split (103,420 seqs)")
    bench = BENCHMARK + [(f"ESM2 + LoRA r=8\n(full, {run})", final["val_mcc"])]
    ys = range(len(bench))
    colors = [MUTED] * (len(bench) - 1) + [SERIES[0]]
    c.barh(ys, [v for _, v in bench], color=colors, height=0.6)
    for y, (_, v) in zip(ys, bench):
        c.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=9, color=INK)
    c.set_yticks(list(ys))
    c.set_yticklabels([n for n, _ in bench], fontsize=8.5, color=INK_2)
    c.set_xlim(0, 1.08)
    c.grid(axis="y", visible=False)

    # d) per source
    style(d, "d) Accuracy per source, full val split")
    keys = [k for k, _ in SOURCE_LABELS]
    x = range(len(keys))
    w = 0.38
    run_acc = [final["per_source"][k]["accuracy"] for k in keys]
    pilot_acc = [PILOT_PER_SOURCE[k] for k in keys]
    d.bar([i - w / 2 - 0.01 for i in x], pilot_acc, width=w, color=SERIES[1], label="pilot (20k seqs)")
    d.bar([i + w / 2 + 0.01 for i in x], run_acc, width=w, color=SERIES[0], label=f"{run} (full)")
    for i, v in zip(x, run_acc):
        d.text(i + w / 2 + 0.01, v + 0.002, f"{v:.3f}", ha="center", fontsize=8, color=INK)
    d.set_xticks(list(x))
    d.set_xticklabels([l for _, l in SOURCE_LABELS], fontsize=8.5, color=INK_2)
    d.set_ylim(0.85, 1.01)
    d.set_ylabel("accuracy (recall for SwissProt)", color=INK_2, fontsize=9)
    d.legend(frameon=False, fontsize=9, loc="upper left")
    d.grid(axis="x", visible=False)

    fig.suptitle(f"ESM2-650M + LoRA (r=8, q/k/v) - {run}", color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    out = os.path.join(run_dir, f"{run}_results.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
