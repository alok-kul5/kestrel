"""Download the IEEE-CIS Fraud Detection training data (Vesta, 2019).

The files come from a public Hugging Face mirror of the Kaggle competition data,
so no Kaggle login is needed. They land in data/raw/ and are never committed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

MIRROR = "https://huggingface.co/datasets/aliceczr/ieee-fraud-detection/resolve/main"
FILES = ["train_transaction.csv", "train_identity.csv"]
ROOT = Path(__file__).resolve().parents[1]


def download(name: str, out_dir: Path) -> Path:
    dest = out_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"{name}: already here, skipping")
        return dest
    tmp = dest.with_suffix(".part")
    with httpx.stream("GET", f"{MIRROR}/{name}", follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r{name}: {done / total:6.1%}", end="", flush=True)
    print()
    tmp.replace(dest)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        download(name, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
