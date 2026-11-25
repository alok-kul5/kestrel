"""SHAP reason codes written the way a fraud analyst reads them.

LightGBM's `pred_contrib=True` returns exact TreeSHAP values, so we get
per-decision attributions without an extra dependency or a slow explainer.
"""
from __future__ import annotations

import math

import numpy as np

REASONS = {
    "TransactionAmt": "Amount ${v:,.2f}",
    "uid_cnt_1h": "{v:.0f} txns on this account in the last hour",
    "uid_cnt_24h": "{v:.0f} txns on this account in 24h",
    "uid_cnt_7d": "{v:.0f} txns on this account in 7 days",
    "uid_amt_24h": "${v:,.0f} spent on this account in 24h",
    "uid_amt_7d": "${v:,.0f} spent on this account in 7 days",
    "card_cnt_24h": "{v:.0f} txns on this card range in 24h",
    "card_amt_24h": "${v:,.0f} on this card range in 24h",
    "dev_cnt_24h": "{v:.0f} txns from this device in 24h",
    "secs_since_last": "{h} since the account's last txn",
    "uid_prior_n": "Account has {v:.0f} prior txns",
    "uid_amt_mean": "Account's usual amount is ${v:,.2f}",
    "amt_z": "Amount is {v:+.1f} std devs from the account's norm",
    "amt_ratio": "Amount is {v:.1f}x the account's average",
    "hour": "Made at {v:04.1f}h",
    "hour_dev": "{v:.1f}h away from the account's usual time of day",
    "new_device": "{yn} device for this account",
    "new_email": "{yn} email domain for this account",
    "dev_uids_30d": "Device used by {v:.0f} accounts in 30 days",
    "card_uids_7d": "Card range used by {v:.0f} accounts in 7 days",
    "uid_devices_30d": "Account used {v:.0f} devices in 30 days",
    "uid_cnt_30d": "{v:.0f} txns on this account in 30 days",
    "amt_cents": "Cents part of amount is {v:.3f}",
    "prod_amt_ratio": "Amount is {v:.1f}x the product line average",
    "g_comp_uids": "Linked to {v:.0f} accounts via shared devices",
    "g_comp_cards": "Device cluster spans {v:.0f} card ranges",
    "g_uid_degree": "Account linked to {v:.0f} devices in the graph",
    "g_dev_degree": "Device linked to {v:.0f} accounts in the graph",
    "dist1": "Billing distance {v:.0f}",
    "P_emaildomain_freq": "Email domain share {v:.2%}",
    "R_emaildomain_freq": "Recipient email domain share {v:.2%}",
    "ProductCD_freq": "Product line share {v:.1%}",
    "card6_freq": "Card type share {v:.1%}",
    "card4_freq": "Card network share {v:.1%}",
    "DeviceType_freq": "Device type share {v:.1%}",
}


def _human_secs(s: float) -> str:
    if s < 120:
        return f"{s:.0f}s"
    if s < 7200:
        return f"{s / 60:.0f} min"
    if s < 172800:
        return f"{s / 3600:.0f}h"
    return f"{s / 86400:.0f} days"


def describe(name: str, value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return f"{name} missing"
    tpl = REASONS.get(name)
    if tpl is None:
        return f"{name} = {value:g}"
    return tpl.format(v=value, h=_human_secs(value), yn="New" if value >= 0.5 else "Known")


def reason_codes(contrib: np.ndarray, x: np.ndarray, names: list[str], k: int = 3) -> list[dict]:
    """Top-k features pushing this decision toward fraud."""
    c = contrib[: len(names)]
    order = np.argsort(-c)[:k]
    return [{"feature": names[i], "value": None if np.isnan(x[i]) else float(x[i]),
             "impact": float(c[i]), "text": describe(names[i], float(x[i]))}
            for i in order if c[i] > 0]
