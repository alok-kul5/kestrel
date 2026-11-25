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
    "uid_amt_mean", "amt_z", "amt_ratio", "hour", "hour_dev",
    "new_device", "new_email", "dev_uids_30d", "card_uids_7d", "uid_devices_30d",
    "uid_cnt_30d", "prod_amt_ratio", "amt_cents",
    "g_comp_uids", "g_comp_cards", "g_uid_degree", "g_dev_degree", "g_comp_known_fraud",
    "uid_known_fraud", "uid_known_rate", "dev_known_fraud", "card_known_rate",
]

# Confirmed fraud labels become visible this long after the transaction
# (chargebacks and analyst confirmations are never instant).
LABEL_DELAY = 14 * DAY


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


class DistinctWindow:
    """Distinct keys seen inside a sliding time window."""

    __slots__ = ("span", "q", "counts")

    def __init__(self, span: int):
        self.span = span
        self.q: deque = deque()
        self.counts: dict = {}

    def add(self, ts: int, key) -> None:
        self.q.append((ts, key))
        self.counts[key] = self.counts.get(key, 0) + 1
        q, c = self.q, self.counts
        while q and q[0][0] <= ts - self.span:
            k = q.popleft()[1]
            c[k] -= 1
            if not c[k]:
                del c[k]

    def __len__(self) -> int:
        return len(self.counts)


@dataclass
class Account:
    w1h: Window = field(default_factory=lambda: Window(HOUR))
    w24h: Window = field(default_factory=lambda: Window(DAY))
    w7d: Window = field(default_factory=lambda: Window(7 * DAY))
    w30d: Window = field(default_factory=lambda: Window(30 * DAY))
    dev30: DistinctWindow = field(default_factory=lambda: DistinctWindow(30 * DAY))
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
        self.device_uids: dict = defaultdict(lambda: DistinctWindow(30 * DAY))
        self.card_uids: dict = defaultdict(lambda: DistinctWindow(7 * DAY))
        self.product: dict = defaultdict(lambda: [0, 0.0])
        self.graph: dict = {"uid": {}, "dev": {}}
        self.label_delay = LABEL_DELAY
        self.pending: deque = deque()       # (visible_at, uid, dev, card, label), time ordered
        self.uid_lab: dict = defaultdict(lambda: [0, 0])   # [labelled, fraud]
        self.dev_lab: dict = defaultdict(lambda: [0, 0])
        self.card_lab: dict = defaultdict(lambda: [0, 0])

    def label(self, t: dict, is_fraud: int, visible_at: int | None = None) -> None:
        """Queue a confirmed label; it only affects features once `visible_at` has passed."""
        ts = int(t["TransactionDT"]) + self.label_delay if visible_at is None else int(visible_at)
        dev = t.get("device_fp")
        self.pending.append((ts, t["uid"], None if isinstance(dev, float) else dev, t["card1"], int(is_fraud)))

    def _release_labels(self, now: int) -> None:
        q = self.pending
        while q and q[0][0] <= now:
            _, uid, dev, card, y = q.popleft()
            for table, key in ((self.uid_lab, uid), (self.dev_lab, dev), (self.card_lab, card)):
                if key is not None:
                    r = table[key]
                    r[0] += 1
                    r[1] += y

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
        self._release_labels(ts)
        a = self.accounts[uid]
        ul = self.uid_lab.get(uid)
        dl = self.dev_lab.get(dev) if dev is not None else None
        cl = self.card_lab.get(card)

        a.w1h.add(ts, amt)
        a.w24h.add(ts, amt)
        a.w7d.add(ts, amt)
        a.w30d.add(ts, amt)
        cw = self.cards[card]
        cw.add(ts, amt)
        if dev is not None:
            dw = self.devices[dev]
            dw.add(ts, 1.0)
            self.device_uids[dev].add(ts, uid)
            a.dev30.add(ts, dev)
        self.card_uids[card].add(ts, uid)

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
            "dev_uids_30d": len(self.device_uids[dev]) if dev is not None else float("nan"),
            "card_uids_7d": len(self.card_uids[card]),
            "uid_devices_30d": len(a.dev30),
            "uid_cnt_30d": len(a.w30d),
            "prod_amt_ratio": amt / pmean if pn and pmean > 0 else float("nan"),
            "amt_cents": round(amt % 1, 3),
            "g_comp_uids": g_uid[0] if g_uid else 0,
            "g_comp_cards": g_uid[1] if g_uid else 0,
            "g_uid_degree": g_uid[2] if g_uid else 0,
            "g_dev_degree": g_dev if g_dev else 0,
            "g_comp_known_fraud": g_uid[3] if g_uid else 0,
            "uid_known_fraud": ul[1] if ul else 0,
            "uid_known_rate": ul[1] / ul[0] if ul else float("nan"),
            "dev_known_fraud": dl[1] if dl else (0 if dev is not None else float("nan")),
            "card_known_rate": (cl[1] + 0.035 * 20) / (cl[0] + 20) if cl else float("nan"),
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
LABEL_COL = "isFraud"


def build_features(df: pd.DataFrame, snapshots: dict[int, dict] | None = None,
                   store: FeatureStore | None = None) -> tuple[pd.DataFrame, FeatureStore]:
    """Replay `df` (time-sorted) through a store. `snapshots` maps start day -> graph table."""
    store = store or FeatureStore()
    starts = sorted(snapshots or {})
    rows = []
    k = -1
    cols = STREAM_COLS + ([LABEL_COL] if LABEL_COL in df else [])
    for t in df[cols].to_dict("records"):
        while k + 1 < len(starts) and t["day"] >= starts[k + 1]:
            k += 1
            store.set_graph(snapshots[starts[k]])
        rows.append(store.process(t))
        if t.get(LABEL_COL) is not None:
            store.label(t, t[LABEL_COL])
    out = pd.DataFrame(rows, index=df.index, columns=ENGINEERED).astype(np.float32)
    return out, store
