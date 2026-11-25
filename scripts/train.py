"""Train all models on the processed features and write artifacts/."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kestrel.config import ARTIFACTS, PROCESSED  # noqa: E402
from kestrel.train import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="few trees, for a smoke run")
    args = ap.parse_args()
    df = pd.read_parquet(PROCESSED / "transactions.parquet")
    feats = pd.read_parquet(PROCESSED / "features.parquet")
    res = run(df, feats, ARTIFACTS, fast=args.fast)
    print(json.dumps({"champion": res["champion"], "decision": res["decision"],
                      "calibration": res["calibration"]}, indent=2))


if __name__ == "__main__":
    main()
