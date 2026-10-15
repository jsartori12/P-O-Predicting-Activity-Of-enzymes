#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data/sequence_dataset.py

Raw-sequence dataset for end-to-end LoRA training (ESM2Classifier works on
cached embeddings instead). Tokenization happens in the collate function so
each batch is padded only to its own longest sequence, and batches are built
from length-sorted chunks so that padding -- wasted backbone compute -- stays
small without losing shuffling.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


class SequenceDataset(Dataset):
    def __init__(self, csv_path, cfg, max_samples=None, seed=42):
        d = cfg.data
        df = pd.read_csv(csv_path, usecols=[d.id_col, d.sequence_col, d.label_col, d.source_col])
        if max_samples is not None and max_samples < len(df):
            # random, not head(): the CSVs are grouped by cluster/source
            df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

        self.ids = df[d.id_col].tolist()
        self.sequences = df[d.sequence_col].tolist()
        self.sources = df[d.source_col].tolist()
        self.labels = (df[d.label_col] == d.positive_label).astype(np.float32).to_numpy()
        self.lengths = np.array([len(s) for s in self.sequences])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {"idx": idx, "sequence": self.sequences[idx], "label": self.labels[idx]}


class LengthGroupedBatchSampler(Sampler):
    """Shuffle, cut into chunks of batch_size * group_factor, sort each chunk by
    length, split into batches, shuffle batch order. Deterministic per epoch
    (seed + epoch) so a resumed run can skip exactly the batches already seen."""

    def __init__(self, lengths, batch_size, shuffle, seed=42, group_factor=50):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.group_factor = group_factor
        self.epoch = 0
        self.skip_batches = 0

    def set_epoch(self, epoch, skip_batches=0):
        self.epoch = epoch
        self.skip_batches = skip_batches

    def _batches(self):
        n = len(self.lengths)
        if not self.shuffle:
            order = np.argsort(-self.lengths, kind="stable")
            return [order[i:i + self.batch_size] for i in range(0, n, self.batch_size)]

        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(n)
        chunk = self.batch_size * self.group_factor
        batches = []
        for start in range(0, n, chunk):
            idx = order[start:start + chunk]
            idx = idx[np.argsort(-self.lengths[idx], kind="stable")]
            batches.extend(idx[i:i + self.batch_size] for i in range(0, len(idx), self.batch_size))
        rng.shuffle(batches)
        return batches

    def __iter__(self):
        for b in self._batches()[self.skip_batches:]:
            yield b.tolist()

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size - self.skip_batches


class Collator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, items):
        enc = self.tokenizer([it["sequence"] for it in items], return_tensors="pt",
                             padding=True, truncation=True, max_length=self.max_length)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor([it["label"] for it in items], dtype=torch.float32),
            "idx": torch.tensor([it["idx"] for it in items], dtype=torch.long),
        }


def get_dataloader(dataset, tokenizer, cfg, batch_size, shuffle):
    sampler = LengthGroupedBatchSampler(dataset.lengths, batch_size, shuffle, seed=cfg.train.seed)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        collate_fn=Collator(tokenizer, cfg.esm.max_seq_len),
                        num_workers=cfg.data.num_workers, pin_memory=True,
                        persistent_workers=cfg.data.num_workers > 0)
    return loader, sampler
