#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xai_to_pdb.py

Write per-residue XAI scores into the B-factor column of a structure, plus a PyMOL
script that colours by them and shows the annotated catalytic residues as sticks.

Scores come from explain_residues.py output (xai_<run>/attributions.csv):
  gradcam   grad x activation at --layer (default 33, the last hidden state)
  attn      the AttentionPool weights themselves (attention runs only; no gradient)

B-factor = percentile rank of the score within the protein, 0-100 (100 = most
important). Percentiles make proteins and runs comparable and are robust to the
heavy tail of raw attributions. --raw writes the raw value instead.

The structure is the AlphaFold DB model for the UniProt id (downloaded once into
xai_pdb/) unless --pdb is given. Residues are matched by number and the amino
acid identity is checked; the script refuses to write if they disagree.

Usage:
  python3 xai_to_pdb.py --id A0A0B5L781
  python3 xai_to_pdb.py --id A0A0B5L781 --runs lora_attnpool_full lora_r8_full --layer 33
  pymol xai_pdb/A0A0B5L781/view.pml
"""
import argparse
import os
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# this script lives in xai/; the xai_<run>/ outputs and xai_pdb/ are one level up
ROOT = os.path.dirname(HERE)
THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
         "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
         "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M"}


def fetch_alphafold(uid, dest_dir):
    path = os.path.join(dest_dir, f"AF-{uid}-F1.pdb")
    if os.path.exists(path):
        return path
    import json
    with urllib.request.urlopen(f"https://alphafold.ebi.ac.uk/api/prediction/{uid}", timeout=60) as r:
        url = json.load(r)[0]["pdbUrl"]
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, path)
    return path


def read_residues(pdb_path, chain=None):
    """{resnum: one-letter} for the first chain (or `chain`), ATOM records only."""
    res = {}
    for line in open(pdb_path):
        if line.startswith("ATOM"):
            ch = line[21]
            if chain is None:
                chain = ch
            if ch == chain:
                res[int(line[22:26])] = THREE.get(line[17:20], "X")
    return res, chain


def load_scores(run, uid, layer, kind):
    f = os.path.join(ROOT, f"xai_{run}", "attributions.csv")
    if not os.path.exists(f):
        raise FileNotFoundError(f"{f} -- run explain_residues.py --run {run} first")
    col = "attn_weight" if kind == "attn" else "attribution"
    parts = []
    for chunk in pd.read_csv(f, chunksize=1_000_000):
        if kind == "attn" and "attn_weight" not in chunk:
            raise ValueError(f"{run} has no attn_weight column (not an attention-pooling run)")
        sub = chunk[(chunk.id == uid) & (chunk.layer == layer)]
        if len(sub):
            parts.append(sub)
    if not parts:
        raise ValueError(f"{uid} / layer {layer} not in {f} (only the explained enzymes are there)")
    df = pd.concat(parts).sort_values("position")
    return df[["position", "residue", "is_catalytic", "p_active", col]].rename(columns={col: "score"})


def write_bfactor(pdb_in, pdb_out, chain, bvals):
    with open(pdb_in) as fi, open(pdb_out, "w") as fo:
        for line in fi:
            if line.startswith(("ATOM", "HETATM")) and line[21] == chain:
                b = bvals.get(int(line[22:26]), 0.0)
                line = f"{line[:60]}{b:6.2f}{line[66:]}"
            fo.write(line)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True, help="UniProt accession (must be one of the explained enzymes)")
    p.add_argument("--runs", nargs="+", default=["lora_attnpool_full", "lora_r8_full"])
    p.add_argument("--layer", type=int, default=33)
    p.add_argument("--pdb", default=None, help="structure to use; default: AlphaFold DB model")
    p.add_argument("--chain", default=None)
    p.add_argument("--raw", action="store_true", help="raw score in B instead of percentile 0-100")
    p.add_argument("--out", default=None, help="default: xai_pdb/<id>/")
    a = p.parse_args()

    out = a.out or os.path.join(ROOT, "xai_pdb", a.id)
    os.makedirs(out, exist_ok=True)
    pdb = a.pdb or fetch_alphafold(a.id, out)
    struct, chain = read_residues(pdb, a.chain)

    jobs = [(r, "gradcam") for r in a.runs]
    if any(r for r in a.runs if "attnpool" in r):
        jobs.append((next(r for r in a.runs if "attnpool" in r), "attn"))

    written, catalytic = [], None
    for run, kind in jobs:
        df = load_scores(run, a.id, a.layer, kind)
        # sequence check: the attributions were computed on the UniProt sequence
        mism = [(r.position, r.residue, struct.get(r.position)) for r in df.itertuples()
                if r.position in struct and struct[r.position] != r.residue]
        cover = sum(pos in struct for pos in df.position) / len(df)
        if mism:
            raise SystemExit(f"sequence mismatch between structure and attributions, e.g. {mism[:5]} "
                             f"-- the structure is numbered differently; renumber it or pass another --pdb")
        df["pct"] = df.score.rank(pct=True) * 100
        b = dict(zip(df.position, df.score if a.raw else df.pct))
        name = f"{run}_{'attn' if kind == 'attn' else f'gradcam_L{a.layer}'}"
        path = os.path.join(out, f"{a.id}_{name}.pdb")
        write_bfactor(pdb, path, chain, b)
        written.append((name, os.path.basename(path)))
        catalytic = df[df.is_catalytic].position.tolist()
        top = df.nlargest(8, "score")
        print(f"\n{name}   P(active)={df.p_active.iloc[0]:.3f}   structure coverage {cover:.0%}")
        print("  catalytic: " + ", ".join(f"{r.residue}{r.position} (pct {r.pct:.1f})"
                                          for r in df[df.is_catalytic].itertuples()))
        print("  top 8:     " + ", ".join(f"{r.residue}{r.position}{'*' if r.is_catalytic else ''}"
                                          for r in top.itertuples()))
        print(f"  -> {path}")

    # PyMOL script: one object per score, same colour scale, catalytic residues as sticks
    lo, hi = (0, 100) if not a.raw else ("", "")
    sel_cat = "+".join(map(str, catalytic)) if catalytic else "none"
    lines = ["# generated by xai_to_pdb.py -- B-factor = XAI score "
             + ("percentile within the protein (0-100)" if not a.raw else "(raw)"),
             "reinitialize", "bg_color white", "set cartoon_transparency, 0",
             "set ray_opaque_background, 0"]
    for name, fn in written:
        lines += [f"load {fn}, {name}",
                  f"spectrum b, blue_white_red, {name}"
                  + (f", minimum={lo}, maximum={hi}" if not a.raw else "")]
    lines += [f"select catalytic, resi {sel_cat}",
              "show sticks, catalytic and not name N+C+O",
              "set stick_radius, 0.3",
              "label catalytic and name CA, \"%s%s\" % (resn, resi)",
              "set label_size, 20", "set label_color, black",
              "set label_position, (0, 2.5, 0)",
              f"disable all", f"enable {written[0][0]}",
              "orient catalytic", "zoom catalytic, 15",
              "# toggle objects in the panel to compare; colour: blue = low, red = high"]
    pml = os.path.join(out, "view.pml")
    open(pml, "w").write("\n".join(lines) + "\n")
    print(f"\nPyMOL:  cd {out} && pymol view.pml")


if __name__ == "__main__":
    main()
