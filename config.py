#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py

LoRA fine-tuning of ESM2 for active/not_active classification. Same data,
same mean-pooled representation and same MLP head as ESM2Classifier's v2
run (frozen backbone, val_loss 0.322 / mcc 0.713) -- the only change is that
low-rank adapters on the attention projections let the backbone itself
adapt, so any difference vs. v2 is attributable to LoRA.

Unlike ESM2Classifier, every step is a forward+backward through the 650M
backbone, so training is step-based: evaluate on a fixed val subset every
`eval_every_steps` optimizer steps, keep best.pt / last.pt, and run
evaluate.py on the full val split at the end.
"""
from dataclasses import dataclass, field


@dataclass
class EsmConfig:
    model_name  : str = "facebook/esm2_t33_650M_UR50D"   # esm2_t6_8M_UR50D for local smoke tests
    max_seq_len : int = 1024                             # tokens incl. <cls>/<eos>; longest sequence in v2 is < 1022 residues


@dataclass
class DataConfig:
    train_csv    : str = "../classifier_dataset/v2_no_wt_decoys/splits/train.csv"
    val_csv      : str = "../classifier_dataset/v2_no_wt_decoys/splits/test.csv"
    sequence_col : str = "sequence"
    label_col    : str = "label"
    id_col       : str = "id"
    source_col   : str = "source_file"
    positive_label : str = "active"
    max_train_samples : int = None      # random subset, for quick trials
    val_eval_samples  : int = 20000     # fixed random val subset for periodic eval; full split in evaluate.py
    num_workers  : int = 4


@dataclass
class LoraSettings:
    r              : int   = 8
    alpha          : int   = 16          # effective scale alpha/r = 2
    dropout        : float = 0.1
    target_modules : list  = field(default_factory=lambda: ["query", "key", "value"])


@dataclass
class HeadConfig:
    pooling    : str   = "mean"         # mean | attention | max | mean_max | cls | cls_mean
                                        # "mean" reproduces every run so far. The others
                                        # exist because attribution showed the catalytic
                                        # sites carry only 0.73% of the logit under 1/N.
    hidden_dim : int   = 256            # same head as ESM2Classifier v2
    n_layers   : int   = 1              # number of HIDDEN layers (1 = the original head)
    layer_norm : bool  = False          # LayerNorm after each hidden Linear; recommended for n_layers >= 3
    dropout    : float = 0.3


@dataclass
class TrainConfig:
    batch_size        : int   = 8       # sequences per forward pass
    grad_accum_steps  : int   = 4       # effective batch = 32
    epochs            : int   = 2
    lr_lora           : float = 2e-4    # adapters: standard LoRA range 1e-4..5e-4
    lr_head           : float = 1e-3    # head starts from scratch, same lr as v2
    weight_decay_lora : float = 0.01
    weight_decay_head : float = 0.05
    warmup_steps      : int   = 500     # linear warmup, then cosine decay to 0
    max_grad_norm     : float = 1.0
    eval_every_steps  : int   = 1000    # optimizer steps (x32 = 32k sequences)
    log_every_steps   : int   = 50
    early_stop_patience : int = 5       # evals without val_loss improvement
    pos_weight        : float = None    # BCE weight on active examples; None = unweighted.
                                        # --pos-weight auto -> n_not_active / n_active of the
                                        # (possibly subsampled) train set, 1.42 on the full split
    fp16              : bool  = True    # V100 has no bf16; autocast fp16 + GradScaler
    gradient_checkpointing : bool = True  # needed for 650M at 1024 tokens x batch 8 on 32GB
    seed              : int   = 42


@dataclass
class Config:
    esm   : EsmConfig    = field(default_factory=EsmConfig)
    data  : DataConfig   = field(default_factory=DataConfig)
    lora  : LoraSettings = field(default_factory=LoraSettings)
    head  : HeadConfig   = field(default_factory=HeadConfig)
    train : TrainConfig  = field(default_factory=TrainConfig)
