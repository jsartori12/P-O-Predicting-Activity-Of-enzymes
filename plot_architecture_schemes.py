#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_architecture_schemes.py

Two schematic figures, same visual language as xai_scheme.png:

  architectures_scheme.png      shared pipeline + the six pooling modes + the two head types
  attention_pooling_scheme.png  attention pooling step by step, with the REAL learned
                                weights of lora_attnpool_full on one enzyme

The attention example is read from xai_lora_attnpool_full/attributions.csv (layer 33,
attn_weight column, written by explain_residues.py). The enzyme shown is the one whose
catalytic weight share is closest to the median, not the best one.

Run:  python3 plot_architecture_schemes.py [--example UNIPROT_ID]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
INK, INK_2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8a86", "#fcfcfb", "#e4e3df"

# same colours as external_validation_results_full.png
C = {"mean": "#2a78d6", "attention": "#1baf7a", "mean_max": "#008300", "max": "#4a3aa7",
     "cls": "#e34948", "cls_mean": "#e34948", "head": "#eb6834", "backbone": "#2a78d6",
     "neutral": "#8a8a86"}


def tint(hex_, a):
    """Mix a colour with the surface: a=0 surface, a=1 full colour."""
    h = hex_.lstrip("#")
    rgb = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255
    s = np.array([0xfc, 0xfc, 0xfb]) / 255
    return tuple(s + (rgb - s) * a)


def box(ax, x, y, w, h, text, color, strong=False, fs=10, bold=False, ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6",
                                fc=tint(color, 0.16), ec=color if strong else "none",
                                lw=1.6 if strong else 0))
    tx = x + w / 2 if ha == "center" else x + 1.2
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.45)


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=2, label=None, lc=None):
    ax.annotate("", (x1, y1), (x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=14,
                                shrinkA=0, shrinkB=0))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 1.2, label, ha="center", va="bottom",
                fontsize=8.5, color=lc or INK_2)


def section(ax, x, y, text, color):
    ax.text(x, y, text, fontsize=12.5, color=color, fontweight="bold", va="center")


def canvas(w, h):
    fig = plt.figure(figsize=(w, h), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * h / w)
    ax.axis("off")
    return fig, ax


# --------------------------------------------------------------------------
TOKENS = ["<cls>", "M", "K", "T", "H", "Y", "D", "…", "<eos>"]
SPECIAL = {0, len(TOKENS) - 1}


def token_strip(ax, x, y, cell, weights, color, label_top=True, special_on=False, ellipsis=7):
    """A row of token cells; fill intensity = weight (0..1). Specials greyed and crossed
    unless special_on (cls modes)."""
    for i, t in enumerate(TOKENS):
        cx = x + i * cell
        sp = i in SPECIAL
        if sp and not (special_on and i == 0):
            fc, ec = tint(C["neutral"], 0.10), tint(C["neutral"], 0.45)
        elif i == ellipsis:
            fc, ec = SURFACE, SURFACE
        else:
            fc, ec = tint(color, 0.08 + 0.92 * weights[i]), tint(color, 0.6)
        ax.add_patch(Rectangle((cx, y), cell * 0.9, cell * 0.9, fc=fc, ec=ec, lw=0.9))
        if sp and not (special_on and i == 0):
            ax.plot([cx + cell * 0.12, cx + cell * 0.78], [y + cell * 0.12, y + cell * 0.78],
                    color=tint(C["neutral"], 0.6), lw=0.9)
        if label_top:
            ax.text(cx + cell * 0.45, y + cell * 1.05, t, ha="center", va="bottom",
                    fontsize=6.6 if sp else 8, color=INK_2)


def pooling_card(ax, x, y, w, h, title, color, draw, formula, out, note=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=tint(color, 0.07), ec=color, lw=1.4))
    ax.text(x + 1.4, y + h - 2.0, title, fontsize=11, color=color, fontweight="bold", va="center")
    draw(x + 1.6, y + h - 9.6)
    ax.text(x + 1.4, y + 6.3, formula, fontsize=9, color=INK, va="center", style="italic")
    ax.text(x + 1.4, y + 2.9, out, fontsize=8.6, color=INK_2, va="center")
    if note:
        ax.text(x + w - 1.2, y + h - 2.0, note, fontsize=8, color=INK_2, va="center", ha="right")


def architectures_scheme(out):
    fig, ax = canvas(16.5, 14.2)
    ax.text(1, 83.5, "ESM2-650M + LoRA classifiers — architectures compared", fontsize=17,
            fontweight="bold", color=INK, va="center")
    ax.text(1, 81.0, "one backbone, six ways to turn per-residue vectors into one protein vector, "
            "two head sizes", fontsize=11.5, color=INK_2, va="center")

    # 1 shared pipeline -----------------------------------------------------
    section(ax, 1.5, 77.0, "1 · SHARED BY EVERY MODEL", C["backbone"])
    y, hh = 66.5, 8.0
    box(ax, 1.0, y, 11.5, hh, "sequence\nMKTHYD…", C["backbone"])
    arrow(ax, 12.5, y + hh / 2, 15.0, y + hh / 2)
    box(ax, 15.0, y, 13.0, hh, "tokenizer\n<cls> M K T … <eos>", C["backbone"])
    arrow(ax, 28.0, y + hh / 2, 30.5, y + hh / 2)
    box(ax, 30.5, y - 1.0, 21.0, hh + 2.0,
        "ESM2-650M  (33 blocks, frozen)\n+ LoRA r=8, α=16 on query / key / value\n2.03 M LoRA params (trainable)",
        C["backbone"], strong=True, bold=True, fs=10)
    arrow(ax, 51.5, y + hh / 2, 54.0, y + hh / 2)
    box(ax, 54.0, y, 14.5, hh, "last hidden state  H\n(L+2) × 1280\none vector per token", C["backbone"])
    arrow(ax, 68.5, y + hh / 2, 71.0, y + hh / 2)
    box(ax, 71.0, y, 10.5, hh, "POOLING\n(section 2)", C["attention"], strong=True, bold=True)
    arrow(ax, 81.5, y + hh / 2, 84.0, y + hh / 2)
    ax.text(82.75, y - 1.2, "1280 or\n2560", ha="center", va="top", fontsize=8, color=INK_2)
    box(ax, 84.0, y, 7.5, hh, "HEAD\n(sec. 3)", C["head"], strong=True, bold=True)
    arrow(ax, 91.5, y + hh / 2, 93.2, y + hh / 2)
    ax.text(93.5, y + hh / 2, "σ →\nP(active)", fontsize=10, color=INK, va="center", fontweight="bold")

    # 2 pooling -------------------------------------------------------------
    section(ax, 1.5, 61.0, "2 · POOLING  — what differs between runs   (colour intensity = weight a token "
            "gets; grey ✕ = excluded)", C["attention"])
    cw, ch, gx, gy = 31.6, 17.0, 1.6, 1.8
    cell = 2.35
    x0s = [1.0, 1.0 + cw + gx, 1.0 + 2 * (cw + gx)]
    y_top, y_bot = 59.0 - ch, 59.0 - 2 * ch - gy

    uni = [0, .45, .45, .45, .45, .45, .45, 0, 0]
    att = [0, .06, .04, .10, 1.0, .08, .55, 0, 0]

    def d_mean(x, y):
        token_strip(ax, x, y, cell, uni, C["mean"])

    def d_att(x, y):
        token_strip(ax, x, y, cell, att, C["attention"])
        ax.text(x + 9 * cell + 0.6, y + cell * 0.45, "learned w,\nsums to 1", fontsize=7.8,
                color=INK_2, va="center")

    def d_max(x, y):
        # a 3-dim slice of H: for each dimension the largest residue wins
        rng = np.random.default_rng(3)
        vals = rng.random((3, 6))
        for r in range(3):
            win = vals[r].argmax()
            for j in range(6):
                cx, cy = x + (j + 1) * cell, y + cell * 1.2 - r * cell * 0.62
                on = j == win
                ax.add_patch(Rectangle((cx, cy), cell * 0.9, cell * 0.55,
                                       fc=tint(C["max"], 0.95 if on else 0.10),
                                       ec=tint(C["max"], 0.6), lw=0.8))
            ax.text(x + 7 * cell + 0.6, y + cell * 1.2 - r * cell * 0.62 + cell * 0.27,
                    f"dim {r + 1}: max", fontsize=7.4, color=INK_2, va="center")
        ax.text(x + 0.2, y + cell * 1.75, "residues →", fontsize=7.4, color=INK_2)

    def d_meanmax(x, y):
        token_strip(ax, x, y, cell, uni, C["mean_max"])
        ax.text(x + 9 * cell + 0.6, y + cell * 0.45, "mean  ⊕\nmax (per dim)", fontsize=7.8,
                color=INK_2, va="center")

    def d_cls(x, y):
        token_strip(ax, x, y, cell, [1, 0, 0, 0, 0, 0, 0, 0, 0], C["cls"], special_on=True)
        ax.text(x + 9 * cell + 0.6, y + cell * 0.45, "only <cls>;\nresidues reach it\n"
                "via self-attention", fontsize=7.6, color=INK_2, va="center")

    def d_clsmean(x, y):
        token_strip(ax, x, y, cell, [1, .45, .45, .45, .45, .45, .45, 0, 0], C["cls"], special_on=True)
        ax.text(x + 9 * cell + 0.6, y + cell * 0.45, "<cls>  ⊕\nmean of residues", fontsize=7.8,
                color=INK_2, va="center")

    cards = [
        ("mean  (baseline)", C["mean"], d_mean, "pooled = (1/N) Σₗ hₗ",
         "→ 1280   ·   every residue capped at 1/N", "lora_r8_full"),
        ("attention", C["attention"], d_att, "wₗ = softmax(u·hₗ + b) ;  pooled = Σₗ wₗ hₗ",
         "→ 1280   ·   +1,281 params   ·   see attention scheme", "lora_attnpool_full"),
        ("max", C["max"], d_max, "pooled[d] = maxₗ hₗ[d]",
         "→ 1280   ·   one residue per dimension", "lora_maxpool_full"),
        ("mean + max", C["mean_max"], d_meanmax, "pooled = [ mean ; max ]",
         "→ 2560   ·   no extra params", "lora_meanmax_full"),
        ("cls", C["cls"], d_cls, "pooled = h₍cls₎",
         "→ 1280   ·   HF EsmForSequenceClassification route", "lora_cls_full"),
        ("cls + mean", C["cls_mean"], d_clsmean, "pooled = [ h₍cls₎ ; mean ]",
         "→ 2560", "lora_clsmean_full"),
    ]
    for i, (t, c, dfn, f, o, run) in enumerate(cards):
        x, yy = x0s[i % 3], (y_top if i < 3 else y_bot)
        pooling_card(ax, x, yy, cw, ch, t, c, dfn, f, o, note=run)

    # 3 head ----------------------------------------------------------------
    section(ax, 1.5, 18.6, "3 · HEAD", C["head"])
    box(ax, 1.0, 6.0, 47.5, 10.0,
        "1 × 256   (default)\n\nLinear(1280|2560 → 256) → ReLU → Dropout 0.3 → Linear(256 → 1)\n"
        "all pooling runs, + lora_r8_posw_full (BCE pos_weight 1.42)",
        C["head"], strong=True, fs=9.8)
    box(ax, 50.5, 6.0, 48.5, 10.0,
        "6 × 1280   or   6 × 2560   (mean pooling only)\n\n"
        "[ Linear → LayerNorm → ReLU → Dropout 0.3 ] × 6 → Linear(→ 1)\n"
        "lora_deep6x1280_full (11.9 M trainable) · lora_deep6x2560_full (38.1 M)",
        C["head"], strong=True, fs=9.8)
    ax.text(1, 2.6, "<cls>/<eos>/padding are excluded from every residue pooling. Training: BCE, AdamW, "
            "lr 2e-4 (LoRA) / 1e-3 (head + pooler), effective batch 32, fp16.",
            fontsize=9, color=INK_2)
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print("written", out)


# --------------------------------------------------------------------------
def attention_scheme(out, example):
    xdir = os.path.join(HERE, "xai_lora_attnpool_full")
    df = pd.read_csv(os.path.join(xdir, "attributions.csv"),
                     usecols=["id", "layer", "position", "residue", "is_catalytic", "p_active", "attn_weight"])
    df = df[df.layer == df.layer.max()]
    stats = df.groupby("id").agg(L=("position", "size"), ncat=("is_catalytic", "sum"))
    stats["share"] = df[df.is_catalytic].groupby("id").attn_weight.sum()
    if example is None:
        pool = stats[(stats.ncat >= 2) & stats.L.between(150, 400)]
        example = (pool.share - pool.share.median()).abs().idxmin()
    ex = df[df.id == example].sort_values("position")
    L, share = len(ex), stats.loc[example, "share"]
    # set-wide numbers, recomputed here rather than copied
    w_rank = df.groupby("id").attn_weight.rank(pct=True)
    all_share = df[df.is_catalytic].attn_weight.sum() / df.attn_weight.sum()
    uni_share = df.is_catalytic.mean()
    top10 = (w_rank[df.is_catalytic] >= 0.9).mean()

    fig, ax = canvas(16.5, 13.6)
    ax.text(1, 80.0, "Attention pooling — how lora_attnpool_full builds the protein vector",
            fontsize=17, fontweight="bold", color=INK, va="center")
    ax.text(1, 77.4, "a learned, per-residue weighting replaces the fixed 1/N of mean pooling", fontsize=11.5,
            color=INK_2, va="center")

    A = C["attention"]
    section(ax, 1.5, 73.0, "1 · SCORE every residue with one shared linear layer", A)
    y, hh = 61.5, 9.0
    box(ax, 1.0, y, 17.5, hh, "H  from ESM2 + LoRA\n(L+2) × 1280\nh₀ = <cls> … h_L+1 = <eos>", C["backbone"])
    arrow(ax, 18.5, y + hh / 2, 21.5, y + hh / 2)
    box(ax, 21.5, y, 22.5, hh, "score  sₗ = u · hₗ + b\nnn.Linear(1280 → 1)\nsame u for every position "
        "(1,281 params)", A, strong=True, fs=9.8)
    arrow(ax, 44.0, y + hh / 2, 47.0, y + hh / 2)
    box(ax, 47.0, y, 20.0, hh, "mask:  sₗ = −∞  for\n<cls>, <eos>, padding\n(finite min under fp16)", C["neutral"])
    arrow(ax, 67.0, y + hh / 2, 70.0, y + hh / 2)
    box(ax, 70.0, y, 29.0, hh, "softmax over residues\nwₗ = exp(sₗ) / Σⱼ exp(sⱼ)\nwₗ ≥ 0,   Σₗ wₗ = 1",
        A, strong=True, fs=9.8)

    section(ax, 1.5, 57.0, "2 · WEIGHTED SUM, then the usual head", A)
    y2 = 46.0
    box(ax, 70.0, y2, 29.0, hh, "pooled = Σₗ wₗ · hₗ\n1280-d protein vector\n(the same hₗ from H, now weighted)",
        A, strong=True, bold=True, fs=10)
    ax.annotate("", (84.5, y2 + hh), (84.5, y),
                arrowprops=dict(arrowstyle="-|>", color=A, lw=2, mutation_scale=14, shrinkA=0, shrinkB=0))
    ax.text(85.5, (y + y2 + hh) / 2, "w", color=A, fontsize=12, fontweight="bold", va="center")
    arrow(ax, 70.0, y2 + hh / 2, 67.0, y2 + hh / 2)
    box(ax, 33.0, y2, 34.0, hh, "head 1 × 256\nLinear(1280→256) → ReLU → Dropout 0.3 → Linear(256→1)",
        C["head"], strong=True, fs=9.6)
    arrow(ax, 33.0, y2 + hh / 2, 30.0, y2 + hh / 2)
    box(ax, 16.0, y2, 14.0, hh, "σ(logit)\nP(active)", C["head"], strong=True, bold=True)
    ax.text(1.2, y2 + hh / 2, "trained end-to-end\nwith LoRA + head;\nno catalytic-site\nlabels ever given",
            fontsize=8.6, color=INK_2, va="center")

    # 3 real example ---------------------------------------------------------
    section(ax, 1.5, 41.0, f"3 · WHAT IT LEARNED — real weights on {example} ({L} aa, "
            f"P(active) = {ex.p_active.iloc[0]:.2f})", A)
    bx = fig.add_axes([0.065, 0.085, 0.62, 0.36])
    bx.set_facecolor(SURFACE)
    pos, w = ex.position.to_numpy(), ex.attn_weight.to_numpy()
    cat = ex.is_catalytic.to_numpy()
    bx.bar(pos[~cat], w[~cat], width=1.0, color=tint(C["neutral"], 0.75), lw=0)
    bx.bar(pos[cat], w[cat], width=2.2, color=A, lw=0)
    bx.axhline(1 / L, color=C["mean"], lw=1.4, ls=(0, (4, 3)))
    bx.text(L * 0.02, 1 / L + 0.003, f"mean pooling: every residue 1/{L} = {100 / L:.2f}%", color=C["mean"],
            fontsize=8.5, ha="left", va="bottom")
    for p_, r_, w_ in ex[ex.is_catalytic][["position", "residue", "attn_weight"]].itertuples(index=False):
        bx.annotate(f"{r_}{p_}\n{w_ * 100:.1f}%  (catalytic)", (p_, w_), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=A, lw=1))
    top_nc = ex[~ex.is_catalytic].nlargest(2, "attn_weight")
    for p_, r_, w_ in top_nc[["position", "residue", "attn_weight"]].itertuples(index=False):
        bx.annotate(f"{r_}{p_}  {w_ * 100:.1f}%", (p_, w_), xytext=(14, 4), textcoords="offset points",
                    fontsize=8, color=INK_2)
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        bx.spines[s].set_color(MUTED)
    bx.tick_params(colors=INK_2, labelsize=8.5)
    bx.grid(True, axis="y", color=MUTED, alpha=0.25, lw=0.6)
    bx.set_axisbelow(True)
    bx.set_xlim(0, L + 1)
    bx.set_ylim(0, max(w.max() * 1.35, 0.05))
    bx.set_xlabel("residue position", color=INK_2, fontsize=9.5)
    bx.set_ylabel("attention weight wₗ", color=INK_2, fontsize=9.5)
    bx.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    bx.legend([Rectangle((0, 0), 1, 1, fc=A), Rectangle((0, 0), 1, 1, fc=tint(C["neutral"], 0.75))],
              ["annotated catalytic residue (UniProt ACT_SITE)", "other residues"],
              frameon=False, fontsize=8.5, loc="upper left")

    tx = 71.0
    ax.add_patch(FancyBboxPatch((tx, 4.5), 28.0, 33.0, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=tint(A, 0.08), ec="none"))
    lines = [
        ("This enzyme", True),
        (f"{int(cat.sum())} catalytic residues = {100 * cat.mean():.1f}% of positions", False),
        (f"receive {100 * share:.1f}% of the attention", False),
        (f"(typical: median share among enzymes of", False),
        (f" 150–400 aa with ≥ 2 annotated sites)", False),
        ("", False),
        (f"All {df.id.nunique()} XAI enzymes", True),
        (f"{100 * all_share:.1f}% of attention on catalytic residues", False),
        (f"vs {100 * uni_share:.2f}% if weights were uniform", False),
        (f"{100 * top10:.1f}% of catalytic residues in each", False),
        ("enzyme's top 10% of weights", False),
        ("", False),
        ("Why it matters", True),
        ("mean pooling caps any residue at 1/N;", False),
        ("attention can give one residue 20–60%", False),
        ("of the vector the classifier sees.", False),
    ]
    yy = 35.0
    for t, b in lines:
        ax.text(tx + 1.4, yy, t, fontsize=9.6 if not b else 10.5, color=INK if b else INK_2,
                fontweight="bold" if b else "normal", va="center")
        yy -= 1.85
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print("written", out, "| example", example, f"share {share:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--example", default=None, help="UniProt id for the attention example")
    a = p.parse_args()
    architectures_scheme(os.path.join(HERE, "architectures_scheme.png"))
    attention_scheme(os.path.join(HERE, "attention_pooling_scheme.png"), a.example)


if __name__ == "__main__":
    main()
