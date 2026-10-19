"""Stress tests for the four weak points of PERAS-X.
  W1 adversarial non-identifiability  -> test-retest substudy: assumption-free substudy floor + falsification
  W2 choice of gamma                   -> increments test + replicate anchor, gamma chosen by non-rejection
  W3 validity resting on slack         -> Brier target (no Jensen) + expectation-level check from the true laws
  W4 squared loss                      -> variance-adaptive (Maurer/Bernstein) concentration vs Chernoff-KL
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from peras.core.multiscale_pair_extrapolation import (nested_matchings, certificate_v2, expectation_extrapolation,
                                                       increments_test, replicate_anchor, replicate_anchor_v2,
                                                       replicate_anchor_v3)
import runpy
SUITE = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_extrapolation_suite.py"), run_name="lib")
DESIGNS = SUITE["DESIGNS"]
BIN = ["linear2", "smooth5", "sharp2", "wiggle2", "sparse10", "checker_coarse", "checker_fine", "rowwise_random"]
GAMMAS = (1.0, .5, .25)


def classification_run(name, rep, n=4000, m=800):
    m = M_SUB
    r = np.random.default_rng(70_000 + 1000 * BIN.index(name) + rep)
    d, law = DESIGNS[name]; X = r.random((n, d)); P = law(X)
    y = (r.random(n) < P[:, 1]).astype(int)
    sub = np.sort(r.choice(n, m, replace=False)); y2 = (r.random(m) < P[sub, 1]).astype(int)   # retest labels
    mats = nested_matchings(X, 4)
    gold01 = float((1 - P.max(1)).mean()); goldB = float((1 - (P ** 2).sum(1)).mean())
    o = dict(design=name, rep=rep, gold01=gold01, gold_brier=goldB)
    for g in GAMMAS:
        c01 = certificate_v2(mats, y, 2, .05, (g,), "01", conc="kl"); cB = certificate_v2(mats, y, 2, .05, (g,), "brier", conc="kl")
        o[f"x01_g{g}"] = c01["bound"]; o[f"xB_g{g}"] = cB["bound"]
        o[f"exp_overshoot_g{g}"] = expectation_extrapolation(mats, P, dict(a=0, b=3, t=(mats[0][1] / mats[3][1]) ** g))
        o[f"inc_p_g{g}"] = increments_test(mats, y, g)["p"]
        an = replicate_anchor_v3(X, y, sub, y2, 2, g, .025, .05, 4)
        o[f"anchor_p_g{g}"] = an["p_falsify"]; o["substudy_bound"] = an["substudy_bound"]; o[f"tau_g{g}"] = an["tau"]
        o[f"overshoot_rest_g{g}"] = an["extrap"] - float((1 - (P[an["rest"]] ** 2).sum(1)).mean())
    # calibrated pipeline: largest gamma not falsified by either test; certificate at alpha/2, substudy at alpha/2
    chosen = None
    for cap, key in ((1.0, "pipeline_cap1"), (.5, "pipeline_cap_half")):
        chosen = None
        for g in GAMMAS:
            if g <= cap and o[f"anchor_p_g{g}"] >= .05:
                chosen = g; break
        o["gamma_" + key] = chosen
        xb = certificate_v2(mats, y, 2, .025, (chosen,), "01", conc="kl")["bound"] if chosen else 0.0
        o[key] = max(xb, o["substudy_bound"])
    o["fixed_half"] = certificate_v2(mats, y, 2, .05, (.5,), "01", conc="kl")["bound"]
    return o


def mse_run(name, rep, n=4000):
    r = np.random.default_rng(90_000 + 100 * ["mse_uniform", "mse_gauss", "mse_hetero"].index(name) + rep)
    X = r.random((n, 2)); mu = np.sin(3 * X[:, 0]) + X[:, 1]
    if name == "mse_uniform":
        e = r.uniform(-.5, .5, n); s2 = np.full(n, 1 / 12); yr = 3.0
    elif name == "mse_gauss":
        e = r.normal(0, .3, n); s2 = np.full(n, .09); yr = 5.0
    else:
        sd = .1 + .4 * X[:, 1]; e = r.normal(0, 1, n) * sd; s2 = sd ** 2; yr = 5.0
    y = mu + e; mats = nested_matchings(X, 4); gold = float(s2.mean())
    o = dict(design=name, rep=rep, gold=gold, y_range=yr, range_violated=bool(y.max() - y.min() > yr))
    for g in (1.0, .5):
        for conc in ("kl", "bernstein", "betting", "best"):
            o[f"{conc}_g{g}"] = certificate_v2(mats, y, None, .05, (g,), "mse", yr, conc)["bound"]
        o[f"betting_g{g}_range4x"] = certificate_v2(mats, y, None, .05, (g,), "mse", 4 * yr, "betting")["bound"]
    return o


if __name__ == "__main__":
    mode, reps = sys.argv[1], int(sys.argv[2]); M_SUB = int(sys.argv[3]) if len(sys.argv) > 3 else 800; here = os.path.dirname(os.path.abspath(__file__))
    names = BIN if mode == "clf" else ["mse_uniform", "mse_gauss", "mse_hetero"]
    fn = classification_run if mode == "clf" else mse_run
    res = []
    for nm in names:
        for rep in range(reps):
            res.append(fn(nm, rep)); print(json.dumps(res[-1]), flush=True)
    json.dump(res, open(os.path.join(here, f"strengthening_{mode}_{M_SUB}.json"), "w"), indent=1)
