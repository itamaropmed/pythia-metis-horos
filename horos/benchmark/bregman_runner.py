"""External training benchmark: does the Bregman bound hold where R* is known?

This runner belongs to Horos because it fits candidate models. Pythia only
receives its frozen OOF predictions.

Acceptance criteria (docs/status/PLAN.md, milestone M0):

  * validity     B <= R* on every design
  * estimator    median relative error of Ê against R* under 5%
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict

import numpy as np
from sklearn.ensemble import (ExtraTreesRegressor, ExtraTreesClassifier,
                              HistGradientBoostingRegressor, HistGradientBoostingClassifier)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from pythia.core.bregman import bregman_bound
from horos.models.legacy_crossfit import Learner, cross_fit
from pythia.suite.generators.bregman_designs import Design, suite


def _learners(task: str, seed: int = 0) -> list[Learner]:
    """A deliberately heterogeneous pool: shared inductive bias is the enemy of A-fold."""
    if task == "regression":
        return [
            Learner("hgb", HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.08, max_leaf_nodes=31, random_state=seed), "gbm"),
            Learner("et", ExtraTreesRegressor(
                n_estimators=120, min_samples_leaf=5, random_state=seed, n_jobs=-1), "forest"),
            Learner("ridge", Ridge(alpha=1.0), "linear"),
            Learner("knn", KNeighborsRegressor(n_neighbors=15), "knn"),
        ]
    return [
        Learner("hgb", HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, max_leaf_nodes=31, random_state=seed),
            "gbm", "predict_proba"),
        Learner("et", ExtraTreesClassifier(
            n_estimators=120, min_samples_leaf=5, random_state=seed, n_jobs=-1),
            "forest", "predict_proba"),
        Learner("logit", LogisticRegression(max_iter=1000), "linear", "predict_proba"),
        Learner("knn", KNeighborsClassifier(n_neighbors=25), "knn", "predict_proba"),
    ]


def run_design(d: Design, seed: int = 0, n_splits: int = 3) -> dict:
    loss = "mse" if d.task == "regression" else "logloss"
    fm = cross_fit(d.X, d.y, _learners(d.task, seed), n_splits=n_splits, seed=seed)
    res = bregman_bound(d.y, fm.P_oof, loss,
                        families=fm.families, predictor_names=fm.names)
    B, E, R = res["bound"], res["estimator"], d.r_star
    res.update(
        design_id=d.design_id,
        task=d.task,
        r_star=R,
        valid=bool(B <= R + 1e-12),
        slack=float(R - B),
        tightness=float(B / R) if R > 0 else float("nan"),
        est_rel_err=float((E - R) / R) if R > 0 else float("nan"),
        meta=d.meta,
    )
    return res


def run_suite(n_reg: int = 60, n_clf: int = 40, seed: int = 0,
              out: str | None = None, verbose: bool = True) -> dict:
    t0 = time.time()
    rows = []
    designs = suite(n_reg, n_clf)
    for i, d in enumerate(designs, 1):
        r = run_design(d, seed=seed)
        rows.append(r)
        if verbose:
            print(f"[{i:3d}/{len(designs)}] {d.design_id}  R*={r['r_star']:.4f}  "
                  f"L={r['L']:.4f}  B={r['bound']:.4f}  E={r['estimator']:.4f}  "
                  f"{'ok ' if r['valid'] else 'FAIL'}  tight={r['tightness']:.3f}",
                  flush=True)

    def agg(sel) -> dict:
        s = [r for r in rows if sel(r)]
        if not s:
            return {}
        valid = np.array([r["valid"] for r in s])
        tight = np.array([r["tightness"] for r in s])
        err = np.array([r["est_rel_err"] for r in s])
        return {
            "n_designs": len(s),
            "valid": int(valid.sum()),
            "validity_rate": float(valid.mean()),
            "usable_tightness": float(np.mean(np.where(valid, np.clip(tight, 0, 1), 0.0))),
            "median_abs_est_err": float(np.median(np.abs(err))),
            "median_signed_est_err": float(np.median(err)),
            "est_within_5pct": int(np.sum(np.abs(err) < 0.05)),
        }

    summary = {
        "seconds": round(time.time() - t0, 1),
        "all": agg(lambda r: True),
        "regression": agg(lambda r: r["task"] == "regression"),
        "classification": agg(lambda r: r["task"] == "classification"),
        "failures": [r["design_id"] for r in rows if not r["valid"]],
    }
    payload = {"summary": summary, "rows": rows}
    if out:
        with open(out, "w") as fh:
            json.dump(payload, fh, indent=1, default=float)
    if verbose:
        print("\n" + json.dumps(summary, indent=1))
    return payload
