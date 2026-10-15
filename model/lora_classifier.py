#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model/lora_classifier.py

ESM2 backbone with LoRA adapters -> mean pool over residues -> MLP head.

Deliberately not EsmForSequenceClassification: that head reads the <cls>
token, while ESM2Classifier's baseline mean-pools residues (excluding
<cls>/<eos>). Keeping the exact same pooling and head (1280 -> 256 -> 1)
means the LoRA vs. frozen comparison changes one thing only.

Trainable: LoRA A/B matrices on query/key/value + the whole head.
Frozen: every original ESM2 weight.
"""
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel


def residue_mask(attention_mask):
    """1 on real residues, 0 on <cls>, <eos> and padding."""
    m = attention_mask.clone()
    m[:, 0] = 0
    eos = attention_mask.sum(dim=1) - 1
    m[torch.arange(m.size(0), device=m.device), eos] = 0
    return m


def mean_pool_residues(hidden, attention_mask):
    """Same pooling as ESM2Classifier/extract_embeddings.py: average over real
    residues, excluding <cls> (position 0) and <eos> (last real token).

    Every residue enters with the SAME weight 1/N. That is the whole problem this
    module's alternatives address: the attribution analysis showed the catalytic
    sites are ranked 7.4x above chance yet carry only 0.73% of the logit, because
    1/N is fixed no matter how informative a residue is.
    """
    m = residue_mask(attention_mask).unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)


def max_pool_residues(hidden, attention_mask):
    """Keeps the most extreme value per dimension instead of averaging it away."""
    m = residue_mask(attention_mask).unsqueeze(-1).bool()
    # a finite floor, not -inf: under fp16 autocast -inf would propagate NaN
    return hidden.masked_fill(~m, torch.finfo(hidden.dtype).min).max(dim=1).values


class AttentionPool(nn.Module):
    """Learned weights instead of the fixed 1/N:  pooled = sum_l w[l] * h[l].

    ~hidden_size+1 extra parameters. If a residue matters, the model can now give
    it weight 0.3 rather than 1/465 -- which mean pooling makes impossible.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, attention_mask):
        m = residue_mask(attention_mask)
        w = self.score(hidden).squeeze(-1)
        w = w.masked_fill(m == 0, torch.finfo(w.dtype).min)
        w = w.softmax(dim=-1).unsqueeze(-1).to(hidden.dtype)
        return (hidden * w).sum(dim=1)

    def weights(self, hidden, attention_mask):
        """The attention distribution itself -- feed it to explain_residues.py to
        see directly which residues the pooling selected."""
        m = residue_mask(attention_mask)
        w = self.score(hidden).squeeze(-1)
        return w.masked_fill(m == 0, torch.finfo(w.dtype).min).softmax(dim=-1)


def cls_token(hidden, attention_mask):
    """The <cls> vector at position 0 -- what HF's EsmForSequenceClassification uses
    (its EsmClassificationHead does features[:, 0, :]).

    Caveat worth knowing before reading the result: unlike BERT, ESM-2 is pretrained
    with masked language modelling ONLY. There is no sequence-level objective (no
    next-sentence prediction) that would train <cls> to summarise the sequence, so
    out of the box it is not a meaningful sequence embedding -- which is why the
    frozen-backbone runs mean-pool instead. With LoRA the backbone CAN learn to route
    information into <cls> through attention, so the mode is informative here and
    much less so for a frozen backbone.
    """
    return hidden[:, 0]


POOLINGS = ("mean", "attention", "max", "mean_max", "cls", "cls_mean")


class EsmLoraClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        backbone = AutoModel.from_pretrained(cfg.esm.model_name, add_pooling_layer=False)
        if cfg.train.gradient_checkpointing:
            # non-reentrant: gradients reach the LoRA weights even though the
            # embedding output (checkpoint input) doesn't require grad
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        lora_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules,
            bias="none",
        )
        self.backbone = get_peft_model(backbone, lora_config)

        hidden_size = backbone.config.hidden_size
        self.pooling = getattr(cfg.head, "pooling", "mean")
        if self.pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {self.pooling!r}")
        self.attn_pool = AttentionPool(hidden_size) if self.pooling == "attention" else None
        # mean_max and cls_mean concatenate two vectors, so the head sees twice the width
        head_in = hidden_size * 2 if self.pooling in ("mean_max", "cls_mean") else hidden_size
        # n_layers = number of HIDDEN layers. With n_layers=1 and layer_norm=False the
        # module indices are 0,1,2,3 -- identical to the original head, so the lora_r8
        # and lora_r8_posw checkpoints still load.
        layers, d = [], head_in
        for _ in range(getattr(cfg.head, "n_layers", 1)):
            layers.append(nn.Linear(d, cfg.head.hidden_dim))
            if getattr(cfg.head, "layer_norm", False):
                layers.append(nn.LayerNorm(cfg.head.hidden_dim))
            layers += [nn.ReLU(), nn.Dropout(cfg.head.dropout)]
            d = cfg.head.hidden_dim
        layers.append(nn.Linear(d, 1))
        self.head = nn.Sequential(*layers)

    def pool(self, hidden, attention_mask):
        if self.pooling == "mean":
            return mean_pool_residues(hidden, attention_mask)
        if self.pooling == "attention":
            return self.attn_pool(hidden, attention_mask)
        if self.pooling == "max":
            return max_pool_residues(hidden, attention_mask)
        if self.pooling == "cls":
            return cls_token(hidden, attention_mask)
        if self.pooling == "cls_mean":
            return torch.cat([cls_token(hidden, attention_mask),
                              mean_pool_residues(hidden, attention_mask)], dim=-1)
        return torch.cat([mean_pool_residues(hidden, attention_mask),
                          max_pool_residues(hidden, attention_mask)], dim=-1)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(self.pool(hidden, attention_mask).float()).squeeze(-1)

    def trainable_state_dict(self):
        """LoRA + head only (a few MB) -- the frozen backbone comes from the Hub."""
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in names}

    def param_groups(self, cfg):
        lora = [p for n, p in self.backbone.named_parameters() if p.requires_grad]
        head = list(self.head.parameters())
        if self.attn_pool is not None:
            head = head + list(self.attn_pool.parameters())   # pooler trains with the head
        return [
            {"params": lora, "lr": cfg.train.lr_lora, "weight_decay": cfg.train.weight_decay_lora},
            {"params": head, "lr": cfg.train.lr_head, "weight_decay": cfg.train.weight_decay_head},
        ]
