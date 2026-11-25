"""Stream simulator: replay the held-out month against a running /score endpoint.

    python scripts/replay.py --url http://127.0.0.1:8000 --rate 50 --n 2000

With --serve it starts its own server in-process first (autoplay off) and
writes latency percentiles to artifacts/latency.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kestrel.config import ARTIFACTS, PROCESSED, VALID_END_DAY  # noqa: E402


def test_rows(n: int) -> list[dict]:
    df = pd.read_parquet(PROCESSED / "transactions.parquet")
    test = df[df["day"] >= VALID_END_DAY].head(n).drop(columns=["uid", "device_fp", "day", "isFraud"])
    return test.astype(object).where(test.notna(), None).to_dict("records")


def serve(port: int) -> None:
    os.environ["KESTREL_AUTOPLAY"] = "0"
    import uvicorn
    from kestrel.service import app
    threading.Thread(target=uvicorn.run, args=(app,), kwargs={"port": port, "log_level": "warning"},
                     daemon=True).start()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--rate", type=float, default=0, help="txns per second, 0 = as fast as possible")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--serve", action="store_true")
    args = ap.parse_args()
    rows = test_rows(args.n)
    if args.serve:
        port = int(args.url.rsplit(":", 1)[1])
        serve(port)
    client = httpx.Client(base_url=args.url, timeout=60)
    for _ in range(600):
        try:
            if client.get("/health").status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    rtt, server, flagged = [], [], 0
    t_start = time.perf_counter()
    for i, r in enumerate(rows):
        t0 = time.perf_counter()
        ev = client.post("/score", json=r).json()
        rtt.append((time.perf_counter() - t0) * 1000)
        server.append(ev["latency_ms"])
        flagged += ev["decision"] == "review"
        if args.rate:
            time.sleep(max(0.0, (i + 1) / args.rate - (time.perf_counter() - t_start)))
    wall = time.perf_counter() - t_start
    time.sleep(2)
    srv = client.get("/api/latency").json()
    rtt, server = rtt[20:], server[20:]  # first requests warm caches; not counted
    out = {
        "requests": len(rows), "throughput_rps": round(len(rows) / wall, 1),
        "http_round_trip_ms": {q: round(float(np.percentile(rtt, int(q[1:]))), 2) for q in ("p50", "p95", "p99")},
        "server_compute_ms": {q: round(float(np.percentile(server, int(q[1:]))), 2) for q in ("p50", "p95", "p99")},
        "flag_rate": round(flagged / len(rows), 4),
        "shap_reason_codes_p95_ms": round(srv["explain_p95_ms"], 2),
        "explained_cases": srv["explain_n"],
        "note": "single client, sequential requests, localhost, decision path = feature-store update + model + calibration; SHAP reason codes run on a background worker",
    }
    print(json.dumps(out, indent=2))
    if args.serve:
        (ARTIFACTS / "latency.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
