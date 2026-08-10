"""Calibration / uncertainty metrics.

These match the master evaluator exactly — what you see locally is what your
submission is scored on. Read these to understand the scoring; they're short
and standard. Don't change the formulas (the server uses the same ones).

AUROC is implemented by hand here (Mann-Whitney U with average-rank tie
handling) so the student install stays sklearn-free. It is numerically
identical to ``sklearn.metrics.roc_auc_score`` — see
``master/tests/test_student_components.py``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

ECE_BINS = 15
NLL_EPSILON = 1e-12


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    """Top-1 accuracy."""
    return float((probs.argmax(axis=1) == labels).mean())


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """Expected Calibration Error with equal-width bins on max-softmax confidence."""
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Last bin's upper edge is inclusive so confidence == 1.0 doesn't fall off.
        if i == n_bins - 1:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs >= lo) & (confs < hi)
        if not mask.any():
            continue
        total += abs(confs[mask].mean() - correct[mask].mean()) * mask.sum() / n
    return float(total)


def nll(probs: np.ndarray, labels: np.ndarray, eps: float = NLL_EPSILON) -> float:
    """Mean negative log-likelihood with epsilon-clipping for stability."""
    p_true = probs[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(p_true, eps, 1.0)).mean())


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean Brier score: mean_i sum_k (p_ik - onehot_ik)^2."""
    n, _ = probs.shape
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def misclassification_auroc(
    probs: np.ndarray, labels: np.ndarray
) -> Optional[float]:
    """AUROC where score = max-softmax confidence, target = correct ∈ {0,1}.

    Mann-Whitney U formulation. Returns ``None`` when every prediction is
    correct or every prediction is wrong (AUROC is undefined for a
    single-class target vector).
    """
    confs = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(int)
    n_pos = int(correct.sum())
    n_neg = int(len(correct) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(confs).rank(method="average").to_numpy()
    pos_rank_sum = float(ranks[correct == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def compute_all_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "accuracy": accuracy(probs, labels),
        "ece": ece(probs, labels),
        "nll": nll(probs, labels),
        "brier": brier(probs, labels),
        "misclassification_auroc": misclassification_auroc(probs, labels),
    }
