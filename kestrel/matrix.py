"""Turn raw columns plus engineered features into the model matrix."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CATEGORICAL
from .features import ENGINEERED

RAW_NUMERIC = (["TransactionAmt", "card1", "card2", "card3", "card5", "addr1", "addr2", "dist1"]
               + [f"C{i}" for i in range(1, 15)] + ["D1", "D2", "D3", "D4", "D10", "D15"])
FREQ = [f"{c}_freq" for c in CATEGORICAL]
FEATURES = RAW_NUMERIC + FREQ + ENGINEERED


class Encoder:
    """Frequency-encode categoricals using the training window only."""

    def __init__(self) -> None:
        self.maps: dict[str, dict] = {}

    def fit(self, df: pd.DataFrame) -> "Encoder":
        for c in CATEGORICAL:
            vc = df[c].fillna("NA").astype(str).value_counts(normalize=True)
            self.maps[c] = vc.to_dict()
        return self

    def transform(self, df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
        out = df[RAW_NUMERIC].astype(np.float32).copy()
        for c in CATEGORICAL:
            out[f"{c}_freq"] = (df[c].fillna("NA").astype(str).map(self.maps[c])
                                .fillna(0.0).astype(np.float32))
        return pd.concat([out, feats[ENGINEERED]], axis=1)[FEATURES]

    def transform_one(self, t: dict, f: dict) -> np.ndarray:
        row = [np.nan if t.get(c) is None else float(t[c]) for c in RAW_NUMERIC]
        for c in CATEGORICAL:
            v = t.get(c)
            row.append(self.maps[c].get("NA" if v is None else str(v), 0.0))
        row += [f[c] for c in ENGINEERED]
        return np.asarray(row, dtype=np.float32).reshape(1, -1)
