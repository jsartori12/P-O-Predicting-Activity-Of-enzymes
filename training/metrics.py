#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training/metrics.py

Same rationale as EncoderClassifier/training/metrics.py: the test split is
~54% not_active / 46% active (v2_no_wt_decoys), so plain accuracy is only
mildly misleading here (unlike the old v1 split's 65.5%/34.5%), but the
same full-imbalance-robust metric set is worth keeping for comparability
across both projects. Metrics are computed once over all predictions (not
batch-averaged) so roc_auc/pr_auc are well-defined.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)


def compute_classification_metrics(labels, probs, threshold=0.5):
    """labels: (N,) 0/1 array. probs: (N,) predicted P(active) in [0, 1]."""
    labels = np.asarray(labels)
    probs  = np.asarray(probs)
    preds  = (probs >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    metrics = {
        "accuracy"         : accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "precision"        : precision,
        "recall"           : recall,
        "f1"               : f1,
        "mcc"              : matthews_corrcoef(labels, preds),
        "majority_baseline": max(labels.mean(), 1 - labels.mean()),
    }

    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = roc_auc_score(labels, probs)
        metrics["pr_auc"]  = average_precision_score(labels, probs)
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"]  = float("nan")

    return {k: float(v) for k, v in metrics.items()}


def format_metrics(metrics, prefix=""):
    order = ["accuracy", "balanced_accuracy", "f1", "precision", "recall",
              "roc_auc", "pr_auc", "mcc", "majority_baseline"]
    return " | ".join(f"{prefix}{k} {metrics[k]:.4f}" for k in order)
