"""Imbalance-aware metrics and the money cost of a decision rule."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from .config import COST_CHARGEBACK_FEE, COST_REVIEW


def recall_at_fpr(y, p, fpr: float = 0.01) -> float:
    f, t, _ = roc_curve(y, p)
    return float(np.interp(fpr, f, t))


def cost(y, amt, flag) -> float:
    """Every flag costs a review; every missed fraud costs its amount plus a fee."""
    y, amt, flag = np.asarray(y), np.asarray(amt), np.asarray(flag, bool)
    missed = (y == 1) & ~flag
    return float(COST_REVIEW * flag.sum() + (amt[missed] + COST_CHARGEBACK_FEE).sum())


def cost_curve(y, p, amt, thresholds=None):
    """Total cost for each global threshold, computed in one sorted pass."""
    y, p, amt = np.asarray(y), np.asarray(p), np.asarray(amt)
    if thresholds is None:
        thresholds = np.unique(np.quantile(p, np.linspace(0, 1, 401)))
    order = np.argsort(-p)
    ps, loss = p[order], np.where(y[order] == 1, amt[order] + COST_CHARGEBACK_FEE, 0.0)
    caught = np.concatenate([[0.0], np.cumsum(loss)])
    total_loss = loss.sum()
    n_flag = np.searchsorted(-ps, -np.asarray(thresholds), side="right")
    costs = COST_REVIEW * n_flag + total_loss - caught[n_flag]
    return np.asarray(thresholds), costs


def amount_aware_flags(p, amt) -> np.ndarray:
    """Flag when the expected loss of approving beats the cost of a review."""
    return np.asarray(p) * (np.asarray(amt) + COST_CHARGEBACK_FEE) > COST_REVIEW


def ece(y, p, bins: int = 15) -> float:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(err)


def summary(y, p) -> dict:
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "recall_at_1pct_fpr": recall_at_fpr(y, p, 0.01),
        "recall_at_5pct_fpr": recall_at_fpr(y, p, 0.05),
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
    }
