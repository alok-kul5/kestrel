"""Streaming feature store: the one code path for training and serving.

Offline we replay every transaction through FeatureStore in time order, and the
API calls the very same `process` method per request. Each feature therefore
only ever sees the past (plus the transaction being scored), so there is no
look-ahead leakage and no train/serve skew.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HOUR, DAY = 3600, 86_400

ENGINEERED = [
    "uid_cnt_1h", "uid_cnt_24h", "uid_cnt_7d", "uid_amt_24h", "uid_amt_7d",
    "card_cnt_24h", "card_amt_24h", "dev_cnt_24h", "secs_since_last",
    "uid_prior_n", "uid_amt_mean", "amt_z", "amt_ratio", "hour", "hour_dev",
    "new_device", "new_email", "dev_n_uids", "card_n_uids", "uid_n_devices",
    "prod_amt_ratio",
    "g_comp_uids", "g_comp_cards", "g_uid_degree", "g_dev_degree",
]


class Window:
    """Sliding time window with an O(1) running count and sum."""

    __slots__ = ("span", "q", "total")

    def __init__(self, span: int):
        self.span = span
        self.q: deque = deque()
        self.total = 0.0

    def add(self, ts: int, value: float = 0.0) -> None:
        self.q.append((ts, value))
        self.total += value
        self.expire(ts)

    def expire(self, now: int) -> None:
        q = self.q
        while q and q[0][0] <= now - self.span:
            self.total -= q.popleft()[1]

    def __len__(self) -> int:
        return len(self.q)


@dataclass
class Account:
    w1h: Window = field(default_factory=lambda: Window(HOUR))
    w24h: Window = field(default_factory=lambda: Window(DAY))
    w7d: Window = field(default_factory=lambda: Window(7 * DAY))
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    last_ts: int | None = None
    hour_sin: float = 0.0
    hour_cos: float = 0.0
    devices: set = field(default_factory=set)
    emails: set = field(default_factory=set)


def _nan(x):
    return float("nan") if x is None else x


class FeatureStore:
    def __init__(self) -> None:
        self.accounts: dict[str, Account] = defaultdict(Account)
        self.cards: dict = defaultdict(lambda: Window(DAY))
        self.devices: dict = defaultdict(lambda: Window(DAY))
        self.device_uids: dict = defaultdict(set)
        self.card_uids: dict = defaultdict(set)
        self.product: dict = defaultdict(lambda: [0, 0.0])
        self.graph: dict = {"uid": {}, "dev": {}}

    def set_graph(self, snapshot: dict) -> None:
        self.graph = snapshot

    def process(self, t: dict) -> dict:
        """Compute features for transaction `t`, then fold it into the state."""
        ts, amt, uid = int(t["TransactionDT"]), float(t["TransactionAmt"]), t["uid"]
        dev, email, card, prod = t.get("device_fp"), t.get("P_emaildomain"), t["card1"], t.get("ProductCD")
        if isinstance(email, float):
            email = None
        if isinstance(dev, float):
            dev = None
        a = self.accounts[uid]

        a.w1h.add(ts, amt)
        a.w24h.add(ts, amt)
        a.w7d.add(ts, amt)
        cw = self.cards[card]
        cw.add(ts, amt)
        if dev is not None:
            dw = self.devices[dev]
            dw.add(ts, 1.0)
            self.device_uids[dev].add(uid)
        self.card_uids[card].add(uid)

        std = math.sqrt(a.m2 / (a.n - 1)) if a.n > 1 else None
        hour = (ts % DAY) / HOUR
        ang = 2 * math.pi * hour / 24
        if a.n:
            mean_ang = math.atan2(a.hour_sin, a.hour_cos)
            diff = abs(ang - mean_ang) % (2 * math.pi)
            hour_dev = min(diff, 2 * math.pi - diff) * 24 / (2 * math.pi)
        else:
            hour_dev = None
        pn, pmean = self.product[prod]
        g_uid = self.graph["uid"].get(uid)
        g_dev = self.graph["dev"].get(dev) if dev is not None else None

        f = {
            "uid_cnt_1h": len(a.w1h), "uid_cnt_24h": len(a.w24h), "uid_cnt_7d": len(a.w7d),
            "uid_amt_24h": a.w24h.total, "uid_amt_7d": a.w7d.total,
            "card_cnt_24h": len(cw), "card_amt_24h": cw.total,
            "dev_cnt_24h": len(self.devices[dev]) if dev is not None else float("nan"),
            "secs_since_last": float(ts - a.last_ts) if a.last_ts is not None else float("nan"),
            "uid_prior_n": a.n,
            "uid_amt_mean": a.mean if a.n else float("nan"),
            "amt_z": (amt - a.mean) / std if std else float("nan"),
            "amt_ratio": amt / a.mean if a.n and a.mean > 0 else float("nan"),
            "hour": hour,
            "hour_dev": _nan(hour_dev),
            "new_device": (float(dev not in a.devices) if a.n else 1.0) if dev is not None else float("nan"),
            "new_email": (float(email not in a.emails) if a.n else 1.0) if email is not None else float("nan"),
            "dev_n_uids": len(self.device_uids[dev]) if dev is not None else float("nan"),
            "card_n_uids": len(self.card_uids[card]),
            "uid_n_devices": len(a.devices | ({dev} if dev is not None else set())),
            "prod_amt_ratio": amt / pmean if pn and pmean > 0 else float("nan"),
            "g_comp_uids": g_uid[0] if g_uid else 0,
            "g_comp_cards": g_uid[1] if g_uid else 0,
            "g_uid_degree": g_uid[2] if g_uid else 0,
            "g_dev_degree": g_dev if g_dev else 0,
        }

        # fold the transaction into the account profile
        a.n += 1
        delta = amt - a.mean
        a.mean += delta / a.n
        a.m2 += delta * (amt - a.mean)
        a.last_ts = ts
        a.hour_sin += math.sin(ang)
        a.hour_cos += math.cos(ang)
        if dev is not None:
            a.devices.add(dev)
        if email is not None:
            a.emails.add(email)
        p = self.product[prod]
        p[0] += 1
        p[1] += (amt - p[1]) / p[0]
        return f


STREAM_COLS = ["TransactionDT", "TransactionAmt", "uid", "device_fp", "P_emaildomain", "card1", "ProductCD", "day"]


def build_features(df: pd.DataFrame, snapshots: dict[int, dict] | None = None,
                   store: FeatureStore | None = None) -> tuple[pd.DataFrame, FeatureStore]:
    """Replay `df` (time-sorted) through a store. `snapshots` maps start day -> graph table."""
    store = store or FeatureStore()
    starts = sorted(snapshots or {})
    rows = []
    k = -1
    for t in df[STREAM_COLS].to_dict("records"):
        while k + 1 < len(starts) and t["day"] >= starts[k + 1]:
            k += 1
            store.set_graph(snapshots[starts[k]])
        rows.append(store.process(t))
    out = pd.DataFrame(rows, index=df.index, columns=ENGINEERED).astype(np.float32)
    return out, store
