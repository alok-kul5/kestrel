"""Load IEEE-CIS, keep the columns we use, and build account keys.

IEEE-CIS has no customer id. Following common practice on this dataset we build
one: card1 + addr1 + the day the account was opened (transaction day minus D1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DAY, RAW

TXN_COLS = (
    ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD"]
    + [f"card{i}" for i in range(1, 7)]
    + ["addr1", "addr2", "dist1", "P_emaildomain", "R_emaildomain"]
    + [f"C{i}" for i in range(1, 15)]
    + ["D1", "D2", "D3", "D4", "D10", "D15"]
    + ["M4", "M5", "M6"]
)
ID_COLS = ["TransactionID", "DeviceType", "DeviceInfo", "id_19", "id_20", "id_30", "id_31", "id_33"]

CATEGORICAL = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
               "M4", "M5", "M6", "DeviceType"]


def load_raw(nrows: int | None = None) -> pd.DataFrame:
    txn = pd.read_csv(RAW / "train_transaction.csv", usecols=TXN_COLS, nrows=nrows)
    ident = pd.read_csv(RAW / "train_identity.csv", usecols=ID_COLS)
    return prepare(txn.merge(ident, on="TransactionID", how="left"))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    df["day"] = (df["TransactionDT"] // DAY).astype(int)
    opened = (df["day"] - df["D1"].fillna(0)).astype(int)
    df["uid"] = (df["card1"].astype(str) + "_" + df["addr1"].fillna(-1).astype(int).astype(str)
                 + "_" + opened.astype(str))
    has_dev = df[DEV_PARTS].notna().any(axis=1)
    fp = df[DEV_PARTS[0]].map(_s)
    for c in DEV_PARTS[1:]:
        fp = fp + "|" + df[c].map(_s)
    df["device_fp"] = fp.where(has_dev, None).astype(object)
    return df


DEV_PARTS = ["DeviceInfo", "id_30", "id_31", "id_33", "id_19", "id_20"]


def _s(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def device_fingerprint(t: dict) -> str | None:
    """Same fingerprint as `prepare`, for one transaction at serving time."""
    vals = [_s(t.get(c)) for c in DEV_PARTS]
    return "|".join(vals) if any(vals) else None


def synthetic(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """A small fake frame with the IEEE-CIS schema, for tests and CI.

    It is random data with a planted fraud pattern (bursts of high amounts on
    shared devices), not a sample of the real dataset.
    """
    rng = np.random.default_rng(seed)
    n_acc = max(n // 20, 10)
    acc = rng.integers(0, n_acc, n)
    dt = np.sort(rng.uniform(DAY, 182 * DAY, n)).astype(int)
    fraud_acc = rng.random(n_acc) < 0.08
    is_fraud = (fraud_acc[acc] & (rng.random(n) < 0.6)).astype(int)
    amt = np.round(rng.lognormal(3.8, 0.9, n) * np.where(is_fraud, 3.0, 1.0), 2)
    devices = np.array([f"dev{i}" for i in range(n_acc // 2)])
    dev = np.where(rng.random(n) < 0.3,
                   np.where(is_fraud == 1, devices[:3][rng.integers(0, 3, n)],
                            devices[rng.integers(0, len(devices), n)]), None)
    df = pd.DataFrame({
        "TransactionID": np.arange(n) + 2_987_000,
        "isFraud": is_fraud,
        "TransactionDT": dt,
        "TransactionAmt": amt,
        "ProductCD": rng.choice(list("WCRHS"), n),
        "card1": 1000 + acc, "card2": rng.integers(100, 600, n).astype(float),
        "card3": 150.0, "card4": rng.choice(["visa", "mastercard"], n),
        "card5": 226.0, "card6": rng.choice(["debit", "credit"], n),
        "addr1": (200 + acc % 50).astype(float), "addr2": 87.0,
        "dist1": rng.exponential(30, n),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com"], n),
        "R_emaildomain": None,
        **{f"C{i}": rng.poisson(1 + 3 * is_fraud).astype(float) for i in range(1, 15)},
        "D1": 0.0, "D2": np.nan, "D3": rng.integers(0, 30, n).astype(float),
        "D4": np.nan, "D10": np.nan, "D15": rng.integers(0, 300, n).astype(float),
        "M4": rng.choice(["M0", "M1", "M2"], n), "M5": "F", "M6": "T",
        "DeviceType": np.where(dev == None, None, "desktop"),  # noqa: E711
        "DeviceInfo": dev, "id_19": np.nan, "id_20": np.nan,
        "id_30": None, "id_31": None, "id_33": None,
    })
    df.loc[df["DeviceInfo"].isna(), "DeviceType"] = None
    # every account opens on its first day so the uid is stable
    first_day = df.groupby("card1")["TransactionDT"].transform("min") // DAY
    df["D1"] = (df["TransactionDT"] // DAY - first_day).astype(float)
    return prepare(df)
