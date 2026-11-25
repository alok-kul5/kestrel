import numpy as np

from kestrel import metrics as M
from kestrel.config import COST_CHARGEBACK_FEE, COST_REVIEW
from kestrel.explain import describe, reason_codes
from kestrel.monitor import psi


def test_cost_counts_reviews_and_missed_fraud():
    y = np.array([1, 0, 1, 0])
    amt = np.array([100.0, 50.0, 30.0, 10.0])
    flag = np.array([True, True, False, False])
    assert M.cost(y, amt, flag) == 2 * COST_REVIEW + 30 + COST_CHARGEBACK_FEE


def test_cost_curve_matches_direct_cost():
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.05
    p = np.clip(y * 0.4 + rng.random(2000) * 0.6, 0, 1)
    amt = rng.lognormal(4, 1, 2000)
    thr, costs = M.cost_curve(y, p, amt, np.array([0.2, 0.5, 0.8]))
    for t, c in zip(thr, costs):
        assert np.isclose(c, M.cost(y, amt, p >= t))


def test_amount_aware_rule():
    flags = M.amount_aware_flags(np.array([0.05, 0.05]), np.array([10.0, 500.0]))
    assert list(flags) == [False, True]


def test_recall_at_fpr_perfect_model():
    y = np.array([0] * 90 + [1] * 10)
    p = np.r_[np.linspace(0, 0.4, 90), np.linspace(0.6, 1, 10)]
    assert M.recall_at_fpr(y, p, 0.01) == 1.0


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(1)
    a = rng.beta(1, 20, 20000)
    assert psi(a, rng.beta(1, 20, 20000)) < 0.02
    assert psi(a, rng.beta(3, 10, 20000)) > 0.25


def test_reason_codes_pick_positive_contributions():
    names = ["uid_cnt_24h", "TransactionAmt", "dev_uids_30d"]
    contrib = np.array([0.9, -0.3, 0.4, -2.0])  # last entry is the bias term
    x = np.array([12.0, 80.0, 5.0])
    rc = reason_codes(contrib, x, names)
    assert [r["feature"] for r in rc] == ["uid_cnt_24h", "dev_uids_30d"]
    assert rc[0]["text"] == "12 txns on this account in 24h"
    assert describe("secs_since_last", 45.0).startswith("45s")
