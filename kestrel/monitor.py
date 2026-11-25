"""Score drift (PSI) and fraud-rate tracking over time windows."""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(ref, cur, bins: int = 10) -> float:
    """Population stability index of `cur` against `ref` on reference deciles."""
    ref, cur = np.asarray(ref), np.asarray(cur)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_status(v: float) -> str:
    return "stable" if v < 0.1 else "watch" if v < 0.25 else "drift"


def weekly_report(scores: pd.DataFrame, ref_scores, threshold: float) -> list[dict]:
    """scores: columns day, y, p. One row per 7-day window."""
    s = scores.assign(week=(scores["day"] // 7).astype(int))
    rows = []
    for wk, g in s.groupby("week"):
        v = psi(ref_scores, g["p"])
        flagged = g["p"] >= threshold
        rows.append({
            "week": int(wk), "start_day": int(g["day"].min()), "n": int(len(g)),
            "fraud_rate": float(g["y"].mean()), "mean_score": float(g["p"].mean()),
            "flag_rate": float(flagged.mean()),
            "precision": float(g.loc[flagged, "y"].mean()) if flagged.any() else None,
            "psi": v, "status": psi_status(v),
        })
    return rows
