"""End to end on synthetic data: train, then serve through the FastAPI app."""
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from kestrel.data import synthetic
from kestrel.features import build_features
from kestrel.graph import monthly_snapshots
from kestrel.train import run


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("k")
    df = synthetic(6000, seed=11)
    snaps = monthly_snapshots(df, [30, 60, 90, 120, 152])
    feats, _ = build_features(df, snaps)
    proc = root / "processed"
    proc.mkdir()
    df.to_parquet(proc / "transactions.parquet")
    res = run(df, feats, root / "artifacts", fast=True)
    return root, df, res


def test_training_writes_results(trained):
    root, _, res = trained
    assert res["champion"].startswith("lightgbm")
    for name in ["logreg_balanced", "lightgbm_class_weight", "xgboost", "isolation_forest"]:
        assert 0 <= res["models"][name]["test"]["pr_auc"] <= 1
    assert 0 < res["decision"]["threshold"] < 1
    assert res["decision"]["global_threshold"]["cost"] <= res["decision"]["approve_all_cost"]
    assert json.loads((root / "artifacts" / "results.json").read_text())["champion"] == res["champion"]


def test_service_scores_and_explains(trained, monkeypatch):
    root, df, _ = trained
    from kestrel import service
    monkeypatch.setenv("KESTREL_AUTOPLAY", "0")
    service.engine = service.Engine(artifacts=root / "artifacts", processed=root / "processed")
    client = TestClient(service.app)
    assert client.get("/health").json()["ok"]
    test = df[df["day"] >= 152].drop(columns=["uid", "device_fp", "day", "isFraud"])
    row = test.iloc[0].astype(object).where(test.iloc[0].notna(), None).to_dict()
    r = client.post("/score", json=row).json()
    assert 0 <= r["p"] <= 1 and r["decision"] in ("approve", "review")
    client.post("/api/threshold", json={"threshold": 0.01})
    r = client.post("/score", json={**row, "TransactionID": 1, "TransactionDT": row["TransactionDT"] + 5,
                                    "TransactionAmt": 5000.0}).json()
    assert r["decision"] == ("review" if r["p"] >= 0.01 else "approve")
    assert isinstance(r["reasons"], list)
    feed = client.get("/api/feed").json()
    assert feed["counts"]["scored"] == 2
    assert client.get("/api/monitoring").json()["weekly"]
    assert "threshold_table" in client.get("/api/curves").json()
    assert client.get("/").status_code == 200
