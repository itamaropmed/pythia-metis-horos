"""Add nonlinear (gradient-boosted) variational scores to the complementary-reader scenario."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
from sim_lib import simulate, read
from pythia.core.instrumented import loo_centre_groups, projection_bound, true_spread
from horos.represent.score_learning import variational_scores as vs, oof_scores, stack_weights, stack_scores
ALPHA = .05
def bound(H, y, g, a): return projection_bound(loo_centre_groups(H, g), y, a, groups=g).bound
def one(n, rng):
    d = simulate(n, rng); fit = rng.random(n) < .5; ev = ~fit; g = d["cell"]; y = d["y"]
    R = {"r01": read(d, rng, [0, 1], .1), "r23": read(d, rng, [2, 3], .1), "r45": read(d, rng, [4, 5], .1),
         "junk": read(d, rng, [4], .45, n_hallucinated=5)}
    truth = true_spread(d["P"][ev], g[ev]); out = {}
    for model in ("ridge", "gbm"):
        B = [vs(X[fit], y[fit], g[fit], X[ev], g[ev], 2, model=model) for X in R.values()]
        w = stack_weights([oof_scores(vs, X[fit], y[fit], g[fit], 2,
                                      entity_ids=np.arange(fit.sum()), model=model)
                           for X in R.values()], y[fit], g[fit])
        out[f"stack_{model}"] = bound(stack_scores(B, w), y[ev], g[ev], ALPHA) / truth
        Xa = np.column_stack(list(R.values()))
        out[f"concat_{model}"] = bound(vs(Xa[fit], y[fit], g[fit], Xa[ev], g[ev], 2, model=model), y[ev], g[ev], ALPHA) / truth
    return out
res = {}
for n in (2000, 8000):
    rng = np.random.default_rng(7 + n); rows = [one(n, rng) for _ in range(20)]
    res[f"n{n}"] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    res[f"n{n}"]["misses"] = int(sum(v > 1 for r in rows for v in r.values()))
    print(n, {k: round(v, 3) for k, v in res[f"n{n}"].items()}, flush=True)
(HERE / "results/bound_gbm.json").write_text(json.dumps(res, indent=1))
