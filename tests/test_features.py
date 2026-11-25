import math

import numpy as np
import pandas as pd

from kestrel.data import enrich, synthetic
from kestrel.features import DAY, ENGINEERED, FeatureStore, Window, build_features
from kestrel.graph import snapshot


def _t(ts, amt, uid="a", dev=None, card=1, email="x.com"):
    return {"TransactionDT": ts, "TransactionAmt": amt, "uid": uid, "device_fp": dev,
            "P_emaildomain": email, "card1": card, "ProductCD": "W"}


def test_window_expires_old_events():
    w = Window(10)
    w.add(0, 1.0)
    w.add(5, 2.0)
    w.add(12, 4.0)
    assert len(w) == 2 and w.total == 6.0


def test_velocity_counts_only_past_and_current():
    s = FeatureStore()
    s.process(_t(0, 10))
    s.process(_t(1800, 10))
    f = s.process(_t(2 * 3600, 30))
    assert f["uid_cnt_1h"] == 1          # the 0s and 1800s txns are outside the last hour
    assert f["uid_cnt_24h"] == 3
    assert f["uid_amt_24h"] == 50
    assert f["secs_since_last"] == 2 * 3600 - 1800


def test_deviation_from_account_norm():
    s = FeatureStore()
    for i in range(5):
        s.process(_t(i * 100, 10 + i))
    f = s.process(_t(600, 100))
    assert f["amt_ratio"] == 100 / 12
    assert f["amt_z"] > 10


def test_shared_device_and_new_device():
    s = FeatureStore()
    s.process(_t(0, 5, uid="a", dev="d1"))
    f = s.process(_t(10, 5, uid="b", dev="d1"))
    assert f["dev_uids_30d"] == 2
    assert f["new_device"] == 1.0
    f = s.process(_t(20, 5, uid="b", dev="d1"))
    assert f["new_device"] == 0.0
    f = s.process(_t(40 * DAY, 5, uid="c", dev="d1"))
    assert f["dev_uids_30d"] == 1          # a and b dropped out of the 30-day window


def test_no_lookahead_features_identical_on_prefix():
    """Features of the first n rows must not change when later rows exist."""
    df = synthetic(1500, seed=3)
    full, _ = build_features(df)
    head, _ = build_features(df.iloc[:700])
    pd.testing.assert_frame_equal(full.iloc[:700], head)


def test_graph_snapshot_uses_only_past_edges():
    df = pd.DataFrame({"day": [1, 1, 5], "uid": ["a", "b", "c"], "device_fp": ["d", "d", "d"], "card1": [1, 2, 3]})
    g = snapshot(df, before_day=3)
    assert g["uid"]["a"][0] == 2 and "c" not in g["uid"]
    assert g["dev"]["d"] == 2


def test_online_enrich_matches_offline_keys():
    df = synthetic(300, seed=1)
    raw_cols = [c for c in df.columns if c not in ("uid", "device_fp", "day")]
    for _, r in df.sample(50, random_state=0).iterrows():
        t = enrich({c: r[c] for c in raw_cols})
        assert t["uid"] == r["uid"]
        want = r["device_fp"]
        assert t["device_fp"] == (None if want is None or (isinstance(want, float) and math.isnan(want)) else want)


def test_feature_frame_shape():
    df = synthetic(500)
    f, _ = build_features(df)
    assert list(f.columns) == ENGINEERED and len(f) == len(df)
    assert np.isfinite(f["uid_cnt_24h"]).all()


def test_labels_only_visible_after_delay():
    s = FeatureStore()
    t0 = _t(0, 10, uid="a", dev="d1")
    s.process(t0)
    s.label(t0, 1)
    f = s.process(_t(DAY, 10, uid="a", dev="d1"))
    assert f["uid_known_fraud"] == 0          # label not confirmed yet
    f = s.process(_t(15 * DAY, 10, uid="a", dev="d1"))
    assert f["uid_known_fraud"] == 1 and f["dev_known_fraud"] == 1
