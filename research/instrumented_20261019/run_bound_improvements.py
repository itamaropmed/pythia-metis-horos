"""Tightening the instrumented-information certificate (known-truth simulation).
Compares, at the same alpha and the same EVAL rows:
  A logistic   classifier probabilities of (cell, reader features), LOO-centred (previous default)
  B variational direct ridge estimate of h* = P(Y|X,R) - P(Y|X) (Horos score_learning)
  C portfolio  max over readers of B, alpha/J
  D concat     B on all readers' features concatenated
  E stack      Rayleigh-optimal linear stack of the readers' B-scores (weights from FIT out-of-fold)
Scenario 'complementary': three readers see disjoint concept subsets (+1 junk reader)."""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
from sim_lib import simulate, read, design_matrix
from pythia.core.instrumented import loo_centre_groups, projection_bound, true_spread
from horos.represent.score_learning import variational_scores, oof_scores, stack_weights, stack_scores
ALPHA = .05


def bound(H, y, g, a):
    return projection_bound(loo_centre_groups(H, g), y, a, groups=g).bound


def one(n, rng, scenario):
    d = simulate(n, rng); fit = rng.random(n) < .5; ev = ~fit; g = d["cell"]; y = d["y"]
    if scenario == "complementary":
        R = {"r01": read(d, rng, [0, 1], .1), "r23": read(d, rng, [2, 3], .1), "r45": read(d, rng, [4, 5], .1),
             "junk": read(d, rng, [4], .45, n_hallucinated=5)}
    else:
        R = {f"noisy{j}": read(d, rng, [0, 1, 2, 3, 4, 5], .2) for j in range(5)}
    J = len(R); truth = true_spread(d["P"][ev], g[ev]); out = {}
    # A: logistic on the best single reader (by FIT-OOF alignment would need extra work; report each and the max at alpha/J)
    A = {}
    for k, X in R.items():
        Xd = design_matrix(g, X)
        q = LogisticRegression(C=1., max_iter=1000).fit(Xd[fit], y[fit]).predict_proba(Xd[ev])
        A[k] = bound(q, y[ev], g[ev], ALPHA / J)
    out["A_logistic_portfolio"] = max(A.values())
    B = {k: variational_scores(X[fit], y[fit], g[fit], X[ev], g[ev], 2) for k, X in R.items()}
    out["C_variational_portfolio"] = max(bound(H, y[ev], g[ev], ALPHA / J) for H in B.values())
    Xall = np.column_stack(list(R.values()))
    out["D_variational_concat"] = bound(variational_scores(Xall[fit], y[fit], g[fit], Xall[ev], g[ev], 2), y[ev], g[ev], ALPHA)
    oof = [oof_scores(variational_scores, X[fit], y[fit], g[fit], 2,
                      entity_ids=np.flatnonzero(fit)) for X in R.values()]
    w = stack_weights(oof, y[fit], g[fit])
    out["E_variational_stack"] = bound(stack_scores(list(B.values()), w), y[ev], g[ev], ALPHA)
    return {k: v / truth for k, v in out.items()}, truth


res = {}
for scenario in ("complementary", "five_noisy_reads"):
    for n in (2000, 8000):
        rng = np.random.default_rng(7 + n); rows = [one(n, rng, scenario) for _ in range(30)]
        keys = rows[0][0].keys()
        res[f"{scenario}|n{n}"] = {k: float(np.mean([r[0][k] for r in rows])) for k in keys}
        res[f"{scenario}|n{n}"]["misses(bound>truth)"] = int(sum(any(v > 1 + 1e-9 for v in r[0].values()) for r in rows))
        print(scenario, n, {k: round(v, 3) for k, v in res[f"{scenario}|n{n}"].items()}, flush=True)
(HERE / "results/bound_improvements.json").write_text(json.dumps(res, indent=1))
