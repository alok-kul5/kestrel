"""Real-time scoring service and analyst dashboard backend.

Run:  uvicorn kestrel.service:app --port 8000
"""
from __future__ import annotations

import json
import os
import pickle
import threading
import time
from collections import deque
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from .config import ARTIFACTS, PROCESSED, ROOT, VALID_END_DAY
from .data import enrich
from .explain import reason_codes
from .features import FeatureStore
from .matrix import FEATURES
from .metrics import cost
from .monitor import psi, psi_status

DASHBOARD = ROOT / "dashboard"


class Txn(BaseModel):
    """Raw IEEE-CIS fields. Only the first four are required."""
    model_config = ConfigDict(extra="allow")
    TransactionID: int
    TransactionDT: int
    TransactionAmt: float
    card1: int


def threshold_table(test: pd.DataFrame) -> dict:
    """Per-threshold confusion counts and money on the test month, for the slider."""
    y, p, amt = test["y"].to_numpy(), test["p"].to_numpy(), test["amt"].to_numpy()
    grid = np.round(np.linspace(0.01, 0.99, 99), 2)
    rows = {"threshold": grid.tolist(), "flagged": [], "tp": [], "cost": [], "fraud_dollars": []}
    for t in grid:
        f = p >= t
        rows["flagged"].append(int(f.sum()))
        rows["tp"].append(int((f & (y == 1)).sum()))
        rows["cost"].append(cost(y, amt, f))
        rows["fraud_dollars"].append(float(amt[f & (y == 1)].sum()))
    rows.update(n=int(len(y)), n_fraud=int(y.sum()), fraud_total=float(amt[y == 1].sum()),
                approve_all=cost(y, amt, np.zeros(len(y), bool)))
    return rows


class Engine:
    def __init__(self, artifacts: Path = ARTIFACTS, processed: Path = PROCESSED, warm: bool = True):
        self.booster = lgb.Booster(model_file=str(artifacts / "model.txt"))
        self.calibrator = pickle.loads((artifacts / "calibrator.pkl").read_bytes())
        self.encoder = pickle.loads((artifacts / "encoder.pkl").read_bytes())
        self.results = json.loads((artifacts / "results.json").read_text())
        self.curves = json.loads((artifacts / "curves.json").read_text())
        self.threshold = float(self.results["decision"]["threshold"])
        oos = pd.read_parquet(artifacts / "oos_scores.parquet")
        self.ref_scores = oos.loc[oos["day"] < VALID_END_DAY, "p"].to_numpy()
        self.curves["threshold_table"] = threshold_table(oos[oos["day"] >= VALID_END_DAY])
        self.store = FeatureStore()
        snap = artifacts / "graph_snapshot.pkl"
        if snap.exists():
            self.store.set_graph(pickle.loads(snap.read_bytes()))
        self.lock = threading.Lock()
        self.events: deque = deque(maxlen=400)
        self.flagged: deque = deque(maxlen=150)
        self.live_scores: deque = deque(maxlen=5000)
        self.latency_ms: deque = deque(maxlen=20000)
        self.seq = 0
        self.counts = {"scored": 0, "flagged": 0, "flagged_fraud": 0, "fraud": 0}
        self.queue: list[dict] = []
        self.warm_seconds = 0.0
        self.replay_thread: threading.Thread | None = None
        self.replay_stop = threading.Event()
        self.replay_rate = 20.0
        self.replay_pos = 0
        if warm:
            self._warm(processed)

    def _warm(self, processed: Path) -> None:
        """Replay history before the test month so rolling features are live."""
        t0 = time.time()
        df = pd.read_parquet(processed / "transactions.parquet")
        hist = df[df["day"] < VALID_END_DAY]
        cols = ["TransactionDT", "TransactionAmt", "uid", "device_fp", "P_emaildomain", "card1", "ProductCD"]
        for t in hist[cols].to_dict("records"):
            self.store.process(t)
        test = df[df["day"] >= VALID_END_DAY].drop(columns=["uid", "device_fp", "day"])
        self.queue = test.astype(object).where(test.notna(), None).to_dict("records")
        self.warm_seconds = time.time() - t0

    def score(self, raw: dict, label: int | None = None) -> dict:
        t0 = time.perf_counter()
        t = enrich(raw)
        with self.lock:
            f = self.store.process(t)
        x = self.encoder.transform_one(t, f)
        raw_p = float(self.booster.predict(x)[0])
        p = float(self.calibrator.predict([raw_p])[0])
        flag = p >= self.threshold
        reasons = reason_codes(self.booster.predict(x, pred_contrib=True)[0], x[0], FEATURES) if flag else []
        ms = (time.perf_counter() - t0) * 1000
        with self.lock:
            self.seq += 1
            ev = {"seq": self.seq, "id": int(t["TransactionID"]), "dt": int(t["TransactionDT"]),
                  "amount": float(t["TransactionAmt"]), "product": t.get("ProductCD"),
                  "card": t.get("card4"), "email": t.get("P_emaildomain"),
                  "device": t.get("DeviceInfo"), "account": t["uid"],
                  "p": p, "raw_score": raw_p, "decision": "review" if flag else "approve",
                  "reasons": reasons, "latency_ms": round(ms, 3), "label": label,
                  "velocity_24h": f["uid_cnt_24h"], "shared_device": f["dev_n_uids"]}
            self.events.append(ev)
            self.live_scores.append(p)
            self.latency_ms.append(ms)
            self.counts["scored"] += 1
            if label == 1:
                self.counts["fraud"] += 1
            if flag:
                self.counts["flagged"] += 1
                self.counts["flagged_fraud"] += int(label == 1)
                self.flagged.append(ev)
        return ev

    # replay of the test month, in-process
    def _replay(self) -> None:
        while not self.replay_stop.is_set() and self.replay_pos < len(self.queue):
            row = dict(self.queue[self.replay_pos])
            self.replay_pos += 1
            label = row.pop("isFraud", None)
            self.score(row, label=None if label is None else int(label))
            time.sleep(1.0 / self.replay_rate)

    def start_replay(self, rate: float) -> None:
        self.replay_rate = max(0.5, min(rate, 2000.0))
        if self.replay_thread and self.replay_thread.is_alive():
            return
        self.replay_stop.clear()
        self.replay_thread = threading.Thread(target=self._replay, daemon=True)
        self.replay_thread.start()

    def stop_replay(self) -> None:
        self.replay_stop.set()

    def latency(self) -> dict:
        a = np.asarray(self.latency_ms) if self.latency_ms else np.asarray([0.0])
        return {"n": len(self.latency_ms), "p50_ms": float(np.percentile(a, 50)),
                "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99))}


app = FastAPI(title="Kestrel fraud scoring", version="1.0")
engine: Engine | None = None


def get_engine() -> Engine:
    global engine
    if engine is None:
        engine = Engine(warm=os.environ.get("KESTREL_NO_WARM") != "1")
        if os.environ.get("KESTREL_AUTOPLAY", "1") == "1":
            engine.start_replay(float(os.environ.get("KESTREL_RATE", "12")))
    return engine


@app.on_event("startup")
def _startup() -> None:
    get_engine()


@app.get("/health")
def health():
    e = get_engine()
    return {"ok": True, "warm_seconds": round(e.warm_seconds, 1), "accounts": len(e.store.accounts),
            "queue": len(e.queue), "replay_pos": e.replay_pos}


@app.post("/score")
def score(txn: Txn):
    return get_engine().score(txn.model_dump())


@app.get("/api/summary")
def summary():
    e = get_engine()
    r = e.results
    return {"split": r["split"], "models": r["models"], "champion": r["champion"],
            "calibration": r["calibration"], "decision": r["decision"],
            "shap_importance": r["shap_importance"], "threshold": e.threshold}


@app.get("/api/curves")
def curves():
    return get_engine().curves


@app.get("/api/feed")
def feed(since: int = 0, limit: int = 60):
    e = get_engine()
    with e.lock:
        evs = [ev for ev in e.events if ev["seq"] > since][-limit:]
        counts = dict(e.counts)
    return {"events": evs, "counts": counts, "threshold": e.threshold, "latency": e.latency(),
            "replay": {"running": bool(e.replay_thread and e.replay_thread.is_alive()),
                       "pos": e.replay_pos, "total": len(e.queue), "rate": e.replay_rate}}


@app.get("/api/flagged")
def flagged(limit: int = 40):
    e = get_engine()
    with e.lock:
        return {"cases": list(e.flagged)[-limit:][::-1]}


@app.get("/api/monitoring")
def monitoring():
    e = get_engine()
    with e.lock:
        live = np.asarray(e.live_scores)
    v = psi(e.ref_scores, live) if len(live) >= 200 else None
    return {"weekly": e.results["monitoring"],
            "live": {"n": int(len(live)), "psi": v, "status": psi_status(v) if v is not None else "warming up",
                     "mean_score": float(live.mean()) if len(live) else None}}


class ReplayReq(BaseModel):
    rate: float = 12.0


@app.post("/api/replay/start")
def replay_start(req: ReplayReq):
    get_engine().start_replay(req.rate)
    return {"ok": True}


@app.post("/api/replay/stop")
def replay_stop():
    get_engine().stop_replay()
    return {"ok": True}


class ThresholdReq(BaseModel):
    threshold: float


@app.post("/api/threshold")
def set_threshold(req: ThresholdReq):
    if not 0 < req.threshold < 1:
        raise HTTPException(400, "threshold must be in (0, 1)")
    get_engine().threshold = req.threshold
    return {"threshold": req.threshold}


@app.get("/")
def index():
    return FileResponse(DASHBOARD / "index.html")


app.mount("/static", StaticFiles(directory=DASHBOARD), name="static")
