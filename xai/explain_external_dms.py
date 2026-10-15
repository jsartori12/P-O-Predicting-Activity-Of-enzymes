#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explain_external_dms.py

Layer-33 grad x activation ("gradcam", see explain_residues.py) and, for an
attention-pooling run, the AttentionPool weights, for every sequence of the four
DMS (MaveDB) groups of the external benchmark plus each group's reconstructed
wild type.

All variants of a group have the same length, so they batch without padding.
Output, one file per group:
  <out>/<slug>.npz   ids (n,), gradcam (n, L), attn (n, L) [attention runs], logit (n,)
Row 0 is the wild type (id "WT").

The analysis (where the errors fall relative to the active site, and whether the
mutated residue was one the model looked at) is in
../../external_benchmark/dms_site_analysis.py.

Usage:
  python3 explain_external_dms.py --run lora_attnpool_full --out xai_dms_lora_attnpool_full
"""
import argparse
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# this script lives in xai/; config.py, model/, training/ and runs/ are one level up
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from explain_residues import attribution_gradcam, load_model   # noqa: E402

BENCH = os.path.join(ROOT, "..", "..", "external_benchmark", "benchmark.csv")
DMS_GROUPS = ["Parkin", "SARS-CoV-2 PLpro", "BRCA1", "E4B"]
AA3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
       "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
       "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"}


def slug(g):
    return re.sub(r"[^A-Za-z0-9]+", "_", g).strip("_")


def wild_type(sub):
    """WT from the HGVS wt residues; positions never mutated fall back to the consensus."""
    seqs = sub.sequence.tolist()
    L = len(seqs[0])
    wt = {}
    for d in sub.source_detail:
        m = re.fullmatch(r"p\.([A-Z][a-z]{2})(\d+)\S*", d)
        wt[int(m.group(2))] = AA3[m.group(1)]
    cons = [Counter(s[i] for s in seqs).most_common(1)[0][0] for i in range(L)]
    return "".join(wt.get(i + 1, cons[i]) for i in range(L))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="lora_attnpool_full")
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--groups", nargs="+", default=DMS_GROUPS)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    run_dir = next(d for d in (os.path.join(ROOT, a.run), os.path.join(ROOT, "runs", a.run))
                   if os.path.isfile(os.path.join(d, "config.json")))
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(run_dir, a.ckpt, device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.esm.model_name)

    bench = pd.read_csv(BENCH)
    for g in a.groups:
        path = os.path.join(a.out, f"{slug(g)}.npz")
        if os.path.exists(path):
            print(f"have {path}")
            continue
        sub = bench[bench.report_group == g]
        ids = ["WT"] + sub.id.tolist()
        seqs = [wild_type(sub)] + sub.sequence.tolist()
        L = len(seqs[0])
        gc = np.zeros((len(seqs), L), np.float32)
        aw = np.zeros((len(seqs), L), np.float32) if model.attn_pool is not None else None
        logits = np.zeros(len(seqs), np.float32)
        t0 = time.time()
        for s in range(0, len(seqs), a.batch_size):
            enc = tok(seqs[s:s + a.batch_size], return_tensors="pt").to(device)
            attr, logit = attribution_gradcam(model, enc["input_ids"], enc["attention_mask"])
            gc[s:s + len(logit)] = attr[:, 0, 1:L + 1].float().cpu().numpy()
            logits[s:s + len(logit)] = logit.float().cpu().numpy()
            if aw is not None:
                with torch.no_grad():
                    h = model.backbone(input_ids=enc["input_ids"],
                                       attention_mask=enc["attention_mask"]).last_hidden_state
                    w = model.attn_pool.weights(h, enc["attention_mask"])
                aw[s:s + len(logit)] = w[:, 1:L + 1].float().cpu().numpy()
            done = s + len(logit)
            if done % (a.batch_size * 100) < a.batch_size or done == len(seqs):
                el = time.time() - t0
                print(f"  {g}: {done}/{len(seqs)}  {el / 60:.1f} min, "
                      f"eta {el / done * (len(seqs) - done) / 60:.1f} min", flush=True)
        out = {"ids": np.array(ids), "gradcam": gc, "logit": logits}
        if aw is not None:
            out["attn"] = aw
        np.savez_compressed(path, **out)
        print(f"-> {path}")


if __name__ == "__main__":
    main()
