"""Deep multiscale (7 dyadic levels, n = 4096): larger lag ratios shrink t and therefore the weight of the far-level
upper bound (the only range-dependent term), at the price of asking A(gamma) over a wider lag window.
0-1 designs (incl. adversarial) with Chernoff-KL; squared-loss designs with fixed-bet capital bounds at declared
ranges x1 and x4."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from peras.core.multiscale_pair_extrapolation import nested_matchings, certificate_v2
import runpy
DES = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_extrapolation_suite.py"), run_name="lib")["DESIGNS"]
CLF = ["linear2", "smooth5", "sharp2", "wiggle2", "sparse10", "K4_3d", "checker_coarse", "checker_fine", "rowwise_random"]
N, LEV = 4096, 7


def run(name, rep):
    r = np.random.default_rng(123_000 + 1000 * (CLF + ["mse_gauss", "mse_hetero"]).index(name) + rep)
    if name.startswith("mse"):
        X = r.random((N, 2)); mu = np.sin(3 * X[:, 0]) + X[:, 1]
        sd = np.full(N, .3) if name == "mse_gauss" else .1 + .4 * X[:, 1]
        y = mu + r.normal(0, 1, N) * sd; gold = float((sd ** 2).mean()); mats = nested_matchings(X, LEV)
        o = dict(design=name, rep=rep, gold=gold)
        for g in (1.0, .5):
            for k, yr in (("x1", 5.0), ("x4", 20.0)):
                o[f"g{g}_{k}"] = certificate_v2(mats, y, None, .05, (g,), "mse", yr, "betting")["bound"]
        return o
    d, law = DES[name]; X = r.random((N, d)); P = law(X); K = P.shape[1]
    y = np.array([r.choice(K, p=p) for p in P]) if K > 2 else (r.random(N) < P[:, 1]).astype(int)
    mats = nested_matchings(X, LEV); o = dict(design=name, rep=rep, gold=float((1 - P.max(1)).mean()))
    for g in (1.0, .5, .25):
        o[f"g{g}_deep"] = certificate_v2(mats, y, K, .05, (g,), "01", conc="kl")["bound"]
        o[f"g{g}_4lev"] = certificate_v2(mats[:4], y, K, .05, (g,), "01", conc="kl")["bound"]
    return o


if __name__ == "__main__":
    reps = int(sys.argv[1]); names = sys.argv[2].split(",")
    for nm in names:
        for rep in range(reps):
            print(json.dumps(run(nm, rep)), flush=True)
