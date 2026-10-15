#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_xai.py

Figure for explain_residues.py output.

  a) enrichment of catalytic residues in the top-10% most-attributed, per layer
     -- 1.0 is chance. This is the number that answers the question.
  b) mean percentile rank of catalytic vs background residues, per layer
  c) layer x position attribution map for one example sequence
  d) distribution of percentile ranks, catalytic vs background, at the best layer
  e) THE CONTROL: within each amino-acid type, catalytic vs non-catalytic rank.
     Catalytic residues are chemically distinctive (His/Cys/Asp are 62% of them
     but 9.5% of the background), so a model that merely up-weighted those letters
     would score a high enrichment while understanding nothing. Comparing like
     with like removes that confound entirely.

Run:  python3 plot_xai.py <xai_output_dir>
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, INK_2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8a86", "#fcfcfb"
CAT, BG = "#eb6834", "#2a78d6"          # catalytic / background: slots 2 and 1


def style(ax, title, xlab=None, ylab=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left")
    ax.grid(True, axis="y", color=MUTED, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=8)
    if xlab:
        ax.set_xlabel(xlab, color=INK_2, fontsize=9)
    if ylab:
        ax.set_ylabel(ylab, color=INK_2, fontsize=9)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    df = pd.read_csv(os.path.join(d, "attributions.csv"))
    summary = json.load(open(os.path.join(d, "summary.json")))
    per_layer = pd.DataFrame(summary.get("per_layer", []))
    if per_layer.empty:
        print("no per-layer summary (was --catalytic-sites given?)")
        return

    fig = plt.figure(figsize=(14, 13), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.45, wspace=0.22,
                          top=0.93, bottom=0.055, left=0.07, right=0.98)

    # a) enrichment per layer -- the headline
    ax = fig.add_subplot(gs[0, 0])
    style(ax, "a) Catalytic residues among the top-10% attributed",
          "hidden state (0 = embedding)", "enrichment over chance")
    ax.bar(per_layer.layer, per_layer.enrichment_top10pct, color=CAT, width=0.75)
    ax.axhline(1.0, color=INK, lw=1.4, ls="--")
    ax.text(per_layer.layer.max(), 1.02, " chance (1.0)", color=INK_2, fontsize=8,
            ha="right", va="bottom")
    ax.set_ylim(0, max(2.0, per_layer.enrichment_top10pct.max() * 1.15))

    # b) percentile rank, catalytic vs background
    ax = fig.add_subplot(gs[0, 1])
    style(ax, "b) Mean percentile rank of attribution", "hidden state", "percentile rank")
    ax.plot(per_layer.layer, per_layer.mean_pct_rank_catalytic, color=CAT, lw=2,
            marker="o", ms=4, mec=SURFACE, mew=1, label="catalytic residues")
    ax.plot(per_layer.layer, per_layer.mean_pct_rank_background, color=BG, lw=2,
            marker="s", ms=4, mec=SURFACE, mew=1, label="all other residues")
    ax.axhline(0.5, color=MUTED, lw=1.2, ls="--")
    ax.set_ylim(0.3, 0.8)
    ax.legend(frameon=False, fontsize=8.5)

    # c) layer x position map for one sequence
    ax = fig.add_subplot(gs[1, 0])
    sid = df.id.iloc[0]
    one = df[df.id == sid]
    M = one.pivot(index="layer", columns="position", values="attribution").values
    M = M / (np.abs(M).max(axis=1, keepdims=True) + 1e-12)      # per-layer normalisation
    style(ax, f"c) Attribution map - {sid}", "residue position", "hidden state")
    v = float(np.abs(M).max())
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
    for p in sorted(one[one.is_catalytic].position.unique()):
        ax.axvline(p - 1, color="#008300", lw=1.2, alpha=0.9)
    ax.plot([], [], color="#008300", lw=1.2, label="catalytic site")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label(
        "attribution (normalised per layer)", color=INK_2, fontsize=8)

    # d) rank distribution at the best layer
    ax = fig.add_subplot(gs[1, 1])
    best = summary.get("best_layer", int(per_layer.layer.max()))
    sub = df[df.layer == best].copy()
    sub["pct_rank"] = sub.groupby("id").attribution.rank(pct=True)
    style(ax, f"d) Percentile-rank distribution at hidden state {best}",
          "percentile rank of attribution", "density")
    bins = np.linspace(0, 1, 26)
    ax.hist(sub[~sub.is_catalytic].pct_rank, bins=bins, density=True, color=BG,
            alpha=0.75, label="all other residues")
    ax.hist(sub[sub.is_catalytic].pct_rank, bins=bins, density=True, color=CAT,
            alpha=0.75, label="catalytic residues")
    ax.axvline(0.9, color=INK, lw=1.2, ls="--")
    ax.text(0.9, ax.get_ylim()[1] * 0.95, " top 10%", color=INK_2, fontsize=8, va="top")
    ax.legend(frameon=False, fontsize=8.5)

    # e) the amino-acid control -- within-type comparison
    ax = fig.add_subplot(gs[2, :])
    sub = df[df.layer == best].copy()
    sub["pct_rank"] = sub.groupby("id").attribution.rank(pct=True)
    rows = []
    for aa, g in sub.groupby("residue"):
        c, n = g[g.is_catalytic], g[~g.is_catalytic]
        if len(c) >= 10:
            rows.append({"aa": aa, "n_cat": len(c), "rank_catalytic": c.pct_rank.mean(),
                         "rank_other": n.pct_rank.mean()})
    r = pd.DataFrame(rows).sort_values("rank_catalytic", ascending=False)
    style(ax, "e) THE CONTROL — within the same amino acid, catalytic vs non-catalytic",
          "amino acid (n = catalytic residues of that type)", "mean percentile rank")
    xs = np.arange(len(r))
    ax.bar(xs - 0.2, r.rank_catalytic, width=0.38, color=CAT, label="catalytic")
    ax.bar(xs + 0.2, r.rank_other, width=0.38, color=BG, label="same residue, not catalytic")
    for x, (_, row) in zip(xs, r.iterrows()):
        ax.annotate("", xy=(x + 0.2, row.rank_other), xytext=(x - 0.2, row.rank_catalytic),
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, ls=":"))
        ax.text(x, max(row.rank_catalytic, row.rank_other) + 0.02,
                f"{row.rank_catalytic - row.rank_other:+.2f}", ha="center", fontsize=8, color=INK)
    ax.axhline(0.5, color=MUTED, lw=1.2, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{a}\nn={int(n)}" for a, n in zip(r.aa, r.n_cat)], fontsize=8.5,
                       color=INK_2)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    fig.suptitle(f"Where is the classifier looking?  -  {summary['run']}  .  "
                 f"{summary['n_sequences']} sequences  .  {summary['method']}",
                 color=INK, fontsize=13, x=0.007, y=0.965, ha="left")
    fig.text(0.007, 0.015,
             "Enrichment 1.0 = catalytic residues are no more attributed than any other residue. "
             "Only the final hidden state is an exact decomposition; the others are first-order.",
             fontsize=8.5, color=INK_2)
    out = os.path.join(d, "xai_summary.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"written to {out}")
    print(per_layer.to_string(index=False))


if __name__ == "__main__":
    main()
