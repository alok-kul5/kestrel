"""Train, compare, calibrate and pick a cost-optimal threshold.

Split is by time: fit on days 1-119, choose and calibrate on days 120-151,
report once on the untouched test month (days 152-182).
"""
from __future__ import annotations

import gc
import json
import pickle
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from . import metrics as M
from .config import COST_CHARGEBACK_FEE, COST_REVIEW, SEED, TRAIN_END_DAY, VALID_END_DAY
from .features import ENGINEERED
from .matrix import FEATURES, Encoder
from .monitor import weekly_report

FEEDBACK = ["uid_known_fraud", "uid_known_rate", "dev_known_fraud", "card_known_rate", "g_comp_known_fraud"]
GRAPH = [f for f in ENGINEERED if f.startswith("g_") or f in ("dev_uids_30d", "card_uids_7d", "uid_devices_30d")]


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def lgb_params(fast: bool) -> dict:
    return dict(n_estimators=60 if fast else 3000, learning_rate=0.03, num_leaves=63,
                min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                reg_lambda=1.0, random_state=SEED, verbose=-1, n_jobs=-1)


def split(df: pd.DataFrame):
    tr = df["day"] < TRAIN_END_DAY
    va = (df["day"] >= TRAIN_END_DAY) & (df["day"] < VALID_END_DAY)
    te = df["day"] >= VALID_END_DAY
    return tr.values, va.values, te.values


def run(df: pd.DataFrame, feats: pd.DataFrame, out: Path, fast: bool = False,
        cache: Path | None = None) -> dict:
    """`cache` keeps each model's scores so a run interrupted (e.g. by low RAM) can resume."""
    out.mkdir(parents=True, exist_ok=True)
    tr, va, te = split(df)
    enc = Encoder().fit(df[tr])
    X = enc.transform(df, feats)
    y, amt = df["isFraud"].values, df["TransactionAmt"].values
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    ytr, yva, yte = y[tr], y[va], y[te]
    pos_w = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    rng = np.random.default_rng(SEED)
    res: dict = {"split": {
        "train": {"rows": int(tr.sum()), "fraud_rate": float(ytr.mean()), "days": [1, TRAIN_END_DAY - 1]},
        "valid": {"rows": int(va.sum()), "fraud_rate": float(yva.mean()), "days": [TRAIN_END_DAY, VALID_END_DAY - 1]},
        "test": {"rows": int(te.sum()), "fraud_rate": float(yte.mean()), "days": [VALID_END_DAY, int(df.day.max())]},
    }, "models": {}}
    preds = {}

    def record(name, model_fn, cols=FEATURES, note=""):
        gc.collect()
        t0 = time.time()
        hit = cache / f"{name}.npz" if cache else None
        if hit and hit.exists():
            z = np.load(hit)
            pv, pt = z["pv"], z["pt"]
            model_fn.trees = int(z["trees"]) if z["trees"] >= 0 else None
            if (cache / f"{name}.txt").exists():
                model_fn.model = lgb.Booster(model_file=str(cache / f"{name}.txt"))
        else:
            pv, pt = model_fn(cols)
            if hit:
                cache.mkdir(parents=True, exist_ok=True)
                np.savez(hit, pv=pv, pt=pt, trees=getattr(model_fn, "trees", None) or -1)
                if getattr(model_fn, "model", None) is not None:
                    model_fn.model.save_model(str(cache / f"{name}.txt"))
        preds[name] = (pv, pt)
        res["models"][name] = {"trees": getattr(model_fn, "trees", None), "valid": M.summary(yva, pv), "test": M.summary(yte, pt),
                               "fit_seconds": round(time.time() - t0, 1), "note": note}
        print(f"{name:32s} valid PR-AUC {res['models'][name]['valid']['pr_auc']:.4f}  "
              f"test PR-AUC {res['models'][name]['test']['pr_auc']:.4f}  ({time.time() - t0:.0f}s)")

    # 1. logistic regression baseline
    def logreg(cols):
        m = make_pipeline(FunctionTransformer(signed_log), SimpleImputer(strategy="median"),
                          StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))
        m.fit(Xtr[cols], ytr)
        return m.predict_proba(Xva[cols])[:, 1], m.predict_proba(Xte[cols])[:, 1]
    record("logreg_balanced", logreg, note="class_weight=balanced, log + standardise")

    # 2. LightGBM with three ways of handling imbalance
    def lgbm(weighting):
        def fit(cols):
            Xf, yf = Xtr[cols], ytr
            kw = {}
            if weighting == "class_weight":
                kw["scale_pos_weight"] = pos_w
            if weighting == "undersample":
                keep = (yf == 1) | (rng.random(len(yf)) < 5 * (yf == 1).sum() / (yf == 0).sum())
                Xf, yf = Xf[keep], yf[keep]
            m = lgb.LGBMClassifier(**lgb_params(fast), **kw).fit(
                Xf, yf, eval_set=[(Xva[cols], yva)], eval_metric="average_precision",
                callbacks=[lgb.early_stopping(150, first_metric_only=True, verbose=False)])
            fit.model = m.booster_
            fit.trees = int(m.best_iteration_ or m.n_estimators)
            return m.predict_proba(Xva[cols])[:, 1], m.predict_proba(Xte[cols])[:, 1]
        return fit

    fits = {}
    for w, note in [("none", "plain log-loss"), ("class_weight", f"scale_pos_weight={pos_w:.0f}"),
                    ("undersample", "negatives sampled to 1:5")]:
        f = lgbm(w)
        record(f"lightgbm_{w}", f, note=note)
        fits[w] = f.model

    # 3. XGBoost (GPU if available)
    def xgboost(cols):
        params = dict(n_estimators=60 if fast else 3000, learning_rate=0.03, max_depth=8,
                      subsample=0.8, colsample_bytree=0.7, min_child_weight=5, tree_method="hist",
                      eval_metric="aucpr", early_stopping_rounds=150, random_state=SEED)
        fit_kw = dict(eval_set=[(Xva[cols], yva)], verbose=False)
        try:
            m = xgb.XGBClassifier(device="cuda", **params).fit(Xtr[cols], ytr, **fit_kw)
        except Exception:
            m = xgb.XGBClassifier(**params).fit(Xtr[cols], ytr, **fit_kw)
        return m.predict_proba(Xva[cols])[:, 1], m.predict_proba(Xte[cols])[:, 1]
    record("xgboost", xgboost, note="hist, depth 8")

    # 4. isolation forest trained on legitimate traffic only (novelty view)
    def iforest(cols):
        legit = Xtr[cols][ytr == 0]
        legit = legit.sample(min(len(legit), 50_000), random_state=SEED)
        imp = SimpleImputer(strategy="median").fit(legit)
        m = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=2).fit(imp.transform(legit))
        return -m.score_samples(imp.transform(Xva[cols])), -m.score_samples(imp.transform(Xte[cols]))
    record("isolation_forest", iforest, note="unsupervised, legit-only fit")

    # 5. ablations: how much do engineered and graph features add?
    raw_cols = [c for c in FEATURES if c not in ENGINEERED]
    record("ablation_raw_columns_only", lgbm("none"), cols=raw_cols, note="no engineered features")
    record("ablation_no_graph", lgbm("none"), cols=[c for c in FEATURES if c not in GRAPH],
           note="engineered but no shared-entity/graph features")
    record("ablation_no_label_feedback", lgbm("none"), cols=[c for c in FEATURES if c not in FEEDBACK],
           note="engineered but no delayed-label features")

    # champion: best LightGBM variant on validation PR-AUC (TreeSHAP reason codes need a tree model)
    lgb_names = ["lightgbm_none", "lightgbm_class_weight", "lightgbm_undersample"]
    champ = max(lgb_names, key=lambda n: res["models"][n]["valid"]["pr_auc"])
    model = fits[champ.replace("lightgbm_", "")]
    pv, pt = preds[champ]
    res["champion"] = champ

    # calibration on the validation month
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pv, yva)
    cv, ct = iso.predict(pv), iso.predict(pt)
    res["calibration"] = {
        "test_brier_raw": float(M.summary(yte, pt)["brier"]), "test_brier_calibrated": float(M.summary(yte, ct)["brier"]),
        "test_ece_raw": M.ece(yte, pt), "test_ece_calibrated": M.ece(yte, ct),
    }

    # threshold from the cost matrix, chosen on validation, applied to test
    thr_grid = np.unique(np.concatenate([np.linspace(0.005, 0.995, 199), np.quantile(cv, np.linspace(0.5, 1, 101))]))
    _, vcost = M.cost_curve(yva, cv, amt[va], thr_grid)
    best_thr = float(thr_grid[int(np.argmin(vcost))])
    _, tcost = M.cost_curve(yte, ct, amt[te], thr_grid)
    approve_all = M.cost(yte, amt[te], np.zeros(len(yte), bool))
    f_global = ct >= best_thr
    f_aware = M.amount_aware_flags(ct, amt[te])

    def decision_stats(flag):
        tp = int(((yte == 1) & flag).sum())
        return {"cost": M.cost(yte, amt[te], flag), "flag_rate": float(flag.mean()),
                "precision": float(tp / max(flag.sum(), 1)), "recall": float(tp / max((yte == 1).sum(), 1)),
                "fraud_dollars_caught": float(amt[te][(yte == 1) & flag].sum() / amt[te][yte == 1].sum())}

    res["decision"] = {
        "cost_review": COST_REVIEW, "cost_chargeback_fee": COST_CHARGEBACK_FEE,
        "threshold": best_thr, "approve_all_cost": approve_all,
        "global_threshold": decision_stats(f_global),
        "amount_aware": decision_stats(f_aware),
        "naive_0.5": decision_stats(ct >= 0.5),
    }

    # SHAP global importance on a test sample
    idx = rng.choice(np.flatnonzero(te), size=min(20_000, int(te.sum())), replace=False)
    contrib = model.predict(X.iloc[idx].values, pred_contrib=True)
    imp = np.abs(contrib[:, :-1]).mean(0)
    res["shap_importance"] = sorted(({"feature": f, "mean_abs_shap": float(v)} for f, v in zip(FEATURES, imp)),
                                    key=lambda d: -d["mean_abs_shap"])[:20]

    # monitoring over the out-of-sample weeks
    scores = pd.DataFrame({"TransactionID": df["TransactionID"].values, "day": df["day"].values,
                           "y": y, "amt": amt, "p": np.nan})
    scores.loc[va, "p"], scores.loc[te, "p"] = cv, ct
    oos = scores[va | te]
    res["monitoring"] = weekly_report(oos, cv, best_thr)

    # curves for the dashboard
    prec, rec, _ = precision_recall_curve(yte, ct)
    step = max(len(prec) // 300, 1)
    curves = {"pr": {"precision": prec[::step].tolist(), "recall": rec[::step].tolist()},
              "cost": {"threshold": thr_grid.tolist(), "valid": vcost.tolist(), "test": tcost.tolist()},
              "pr_by_model": {}}
    for n in ["logreg_balanced", champ, "xgboost", "ablation_raw_columns_only"]:
        p_, r_, _ = precision_recall_curve(yte, preds[n][1])
        s = max(len(p_) // 200, 1)
        curves["pr_by_model"][n] = {"precision": p_[::s].tolist(), "recall": r_[::s].tolist()}

    model.save_model(str(out / "model.txt"))
    with open(out / "calibrator.pkl", "wb") as f:
        pickle.dump(iso, f)
    with open(out / "encoder.pkl", "wb") as f:
        pickle.dump(enc, f)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    (out / "curves.json").write_text(json.dumps(curves))
    oos.to_parquet(out / "oos_scores.parquet")
    return res
