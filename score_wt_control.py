#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_wt_control.py

Control for "is the model detecting the catalytic mutation, or just redesign?".

The v2 dataset dropped the 5A/10A/15A_cata_wt decoys. Per protein and shell,
three sequences exist:
  SwissProt    natural enzyme                                   (label active)
  <shell>_wt   shell around the active site redesigned,
               catalytic residues kept                          (never trained on)
  <shell>_mut  same as _wt, only the 1-4 catalytic residues
               mutated                                          (label not_active)
_wt vs _mut is a single-variable comparison: if P(active | _wt) >> P(active | _mut)
the model reads the catalytic residues; if both are ~0 it reads the redesign.

Only proteins whose SwissProt sequence AND <shell>_mut are in the test split,
and with no variant of any kind in the train split, are scored -- so no
protein seen in training leaks in. SwissProt and _mut scores
are taken from <run-dir>/predictions.csv (evaluate.py, same best.pt); only
the _wt sequences get a new forward pass.

Writes <run-dir>/wt_control.json and wt_predictions.csv.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

from data.sequence_dataset import SequenceDataset, get_dataloader
from evaluate import load_run_config
from model.lora_classifier import EsmLoraClassifier
from training.trainer import predict, load_checkpoint

SHELLS = ["5A", "10A", "15A"]


def read_fasta(path):
    """{original_id: sequence}; headers look like >A0A009IHW8|5A|3.2.2.-|20"""
    seqs, header, chunks = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    seqs[header] = "".join(chunks)
                header, chunks = line[1:].split("|")[0].strip(), []
            elif line:
                chunks.append(line)
    if header is not None:
        seqs[header] = "".join(chunks)
    return seqs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--fasta-dir", default="../Decoys_dataset/Decoys_datasets",
                   help="directory with {5A,10A,15A}_cata_wt.fasta and _mut.fasta")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    final_eval_path = os.path.join(args.run_dir, "final_eval.json")
    ckpt_path = os.path.join(args.run_dir, "checkpoints", "best.pt")
    with open(final_eval_path) as fh:
        assert os.path.samefile(json.load(fh)["checkpoint"], ckpt_path), \
            "predictions.csv was not produced from best.pt -- rerun evaluate.py first"
    preds = pd.read_csv(os.path.join(args.run_dir, "predictions.csv"))
    p_by_id = dict(zip(preds["id"], preds["p_active"]))
    test_swissprot = set(preds.loc[preds["source_file"] == "sequencias_swissprot.fasta", "id"])

    cfg = load_run_config(args.run_dir)
    # a protein can have its SwissProt + <shell>_mut in test but another shell's decoy in
    # train (redesign can push a variant into a different cluster) -- exclude those too
    train_ids = pd.read_csv(cfg.data.train_csv, usecols=[cfg.data.id_col])[cfg.data.id_col]
    train_proteins = {i.split("__")[0] for i in train_ids}

    # --- build the _wt set, restricted to test-split proteins ------------------
    rows = []
    for shell in SHELLS:
        wt = read_fasta(os.path.join(args.fasta_dir, f"{shell}_cata_wt.fasta"))
        mut = read_fasta(os.path.join(args.fasta_dir, f"{shell}_cata_mut.fasta"))
        for oid, wt_seq in wt.items():
            mut_id = f"{oid}__{shell}_cata_mut"
            if oid not in test_swissprot or mut_id not in p_by_id or oid in train_proteins:
                continue
            mut_seq = mut.get(oid, "")
            n_diff = sum(a != b for a, b in zip(wt_seq, mut_seq)) if len(mut_seq) == len(wt_seq) else -1
            rows.append({"id": f"{oid}__{shell}_cata_wt", "original_id": oid, "shell": shell,
                         "sequence": wt_seq, "label": "not_active",
                         "source_file": f"{shell}_cata_wt.fasta", "n_catalytic_mut": n_diff})
    wt_df = pd.DataFrame(rows)
    print(f"_wt sequences to score (test-split proteins only): "
          f"{wt_df.groupby('shell').size().to_dict()}")

    input_csv = os.path.join(args.run_dir, "wt_control_input.csv")
    wt_df[["id", "sequence", "label", "source_file"]].to_csv(input_csv, index=False)

    # --- score ------------------------------------------------------------------
    cfg.train.gradient_checkpointing = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = cfg.train.fp16 and device.type == "cuda"

    model = EsmLoraClassifier(cfg).to(device)
    load_checkpoint(ckpt_path, model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.esm.model_name)
    ds = SequenceDataset(input_csv, cfg)
    loader, _ = get_dataloader(ds, tokenizer, cfg, args.batch_size, shuffle=False)
    _, _, probs, idx = predict(model, loader, device, fp16)
    p_wt = dict(zip([ds.ids[i] for i in idx], probs))

    wt_df["p_wt"] = wt_df["id"].map(p_wt)
    wt_df["p_mut"] = [p_by_id[f"{o}__{s}_cata_mut"] for o, s in zip(wt_df["original_id"], wt_df["shell"])]
    wt_df["p_swissprot"] = wt_df["original_id"].map(p_by_id)
    wt_df.drop(columns="sequence").to_csv(os.path.join(args.run_dir, "wt_predictions.csv"), index=False)

    # --- summarize ---------------------------------------------------------------
    def summarize(d):
        delta = d["p_wt"] - d["p_mut"]
        y = np.r_[np.ones(len(d)), np.zeros(len(d))]
        return {
            "n_pairs": int(len(d)),
            "mean_p_swissprot": float(d["p_swissprot"].mean()),
            "mean_p_wt": float(d["p_wt"].mean()),
            "mean_p_mut": float(d["p_mut"].mean()),
            "frac_wt_called_active": float((d["p_wt"] >= 0.5).mean()),
            "frac_mut_called_active": float((d["p_mut"] >= 0.5).mean()),
            "mean_delta_wt_minus_mut": float(delta.mean()),
            "frac_pairs_wt_above_mut": float((delta > 0).mean()),
            # how well P(active) separates _wt from its own _mut (0.5 = can't tell them apart)
            "auc_wt_vs_mut": float(roc_auc_score(y, np.r_[d["p_wt"], d["p_mut"]])),
        }

    report = {}
    print(f"\n{'shell':5s} {'n':>6s} {'P(sp)':>7s} {'P(wt)':>7s} {'P(mut)':>7s} "
          f"{'wt>=.5':>7s} {'mut>=.5':>8s} {'Δ wt-mut':>9s} {'wt>mut':>7s} {'AUC':>6s}")
    for shell in SHELLS:
        d = wt_df[wt_df["shell"] == shell]
        s = summarize(d)
        s["by_n_catalytic_mut"] = {int(k): summarize(g) for k, g in d.groupby("n_catalytic_mut") if len(g) >= 50}
        report[shell] = s
        print(f"{shell:5s} {s['n_pairs']:6d} {s['mean_p_swissprot']:7.3f} {s['mean_p_wt']:7.3f} "
              f"{s['mean_p_mut']:7.3f} {s['frac_wt_called_active']:7.3f} {s['frac_mut_called_active']:8.3f} "
              f"{s['mean_delta_wt_minus_mut']:9.3f} {s['frac_pairs_wt_above_mut']:7.3f} {s['auc_wt_vs_mut']:6.3f}")
        for k, g in s["by_n_catalytic_mut"].items():
            print(f"      {k} catalytic mut: n={g['n_pairs']:6d}  P(wt) {g['mean_p_wt']:.3f}  "
                  f"P(mut) {g['mean_p_mut']:.3f}  AUC {g['auc_wt_vs_mut']:.3f}")

    with open(os.path.join(args.run_dir, "wt_control.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    os.remove(input_csv)
    print(f"\nwritten {args.run_dir}/wt_control.json and wt_predictions.csv")


if __name__ == "__main__":
    main()
