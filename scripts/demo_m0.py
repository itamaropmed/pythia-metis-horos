"""One-screen demo of the M0 core: route a loss, cross-fit, certify.

    python scripts/demo_m0.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pythia.core.bregman import bregman_bound, cluster_bootstrap   # noqa: E402
from horos.models.legacy_crossfit import Learner, cross_fit, preflight  # noqa: E402
from pythia.core.router import describe, route                     # noqa: E402

LOSS = "mse"

print(describe(LOSS), end="\n\n")
print(describe("auc"), end="\n\n")

rng = np.random.default_rng(0)
n, d = 4000, 6
X = rng.standard_normal((n, d))
m = np.sin(1.5 * X[:, 0]) * 3 + 0.8 * X[:, 1] ** 2
y = m + rng.standard_normal(n) * 0.7          # R* = 0.49
R_STAR = 0.49

learners = [
    Learner("hgb", HistGradientBoostingRegressor(max_iter=200, random_state=0), "gbm"),
    Learner("et", ExtraTreesRegressor(n_estimators=120, min_samples_leaf=5,
                                      random_state=0, n_jobs=-1), "forest"),
    Learner("ridge", Ridge(1.0), "linear"),   # deliberately off the pace -- triggers pool_health
    Learner("knn", KNeighborsRegressor(15), "knn"),
]
for w in preflight(learners, route(LOSS).bayes_act):
    print("PRE-FLIGHT:", w)

fm = cross_fit(X, y, learners, n_splits=3, seed=0)
res = bregman_bound(y, fm.P_oof, LOSS, families=fm.families, predictor_names=fm.names)

print(f"n              {res['n']:,}   predictors {res['n_predictors']}")
print(f"zero-info      {res['zero_info']:.4f}")
print(f"L (best OOF)   {res['L']:.4f}   [{res['best_predictor']}]")
print(f"B (bound)      {res['bound']:.4f}")
print(f"E (estimator)  {res['estimator']:.4f}")
print(f"Gamma          {res['gamma']:.4f}")
print(f"true R*        {R_STAR:.4f}   -> bound {'VALID' if res['bound'] <= R_STAR else 'INVALID'}")
print(f"dispersion removed {res['dispersion_removed']:.1%}")
print(f"per predictor  { {k: round(v, 4) for k, v in res['per_predictor_risk'].items()} }")
for w in res["warnings"]:
    print("WARNING:", w)
print(f"diagnostics    { {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res['diagnostics'].items()} }")

ci = cluster_bootstrap(y, fm.P_oof, None, LOSS, reps=120, seed=0)
for k, v in ci.items():
    print(f"{k:10s} {v['point']:.4f}  [{v['lo']:.4f}, {v['hi']:.4f}]")
