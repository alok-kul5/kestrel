"""Replay all transactions through the feature store and save the result."""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kestrel.config import ARTIFACTS, PROCESSED, VALID_END_DAY  # noqa: E402
from kestrel.data import load_raw  # noqa: E402
from kestrel.features import build_features  # noqa: E402
from kestrel.graph import monthly_snapshots  # noqa: E402

SNAPSHOT_DAYS = [30, 60, 90, 120, VALID_END_DAY]


def main() -> None:
    t0 = time.time()
    df = load_raw()
    print(f"loaded {len(df):,} rows, fraud rate {df.isFraud.mean():.3%}, {time.time() - t0:.0f}s")
    snaps = monthly_snapshots(df, SNAPSHOT_DAYS)
    for d, s in snaps.items():
        print(f"graph before day {d}: {s['nodes']:,} nodes, {s['edges']:,} edges")
    t1 = time.time()
    feats, _ = build_features(df, snaps)
    print(f"features: {len(feats):,} rows in {time.time() - t1:.0f}s "
          f"({(time.time() - t1) / len(feats) * 1e6:.0f} us/txn)")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED / "transactions.parquet")
    feats.to_parquet(PROCESSED / "features.parquet")
    ARTIFACTS.mkdir(exist_ok=True)
    with open(ARTIFACTS / "graph_snapshot.pkl", "wb") as f:
        pickle.dump(snaps[VALID_END_DAY], f)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
