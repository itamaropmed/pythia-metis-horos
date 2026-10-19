"""Validity/tightness of the model-free sandwich on known laws (incl. adversarial and null designs), against the
accepted model-based upper bound (held-out Brier of a fitted classifier, KL-certified)."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from peras.core.multiscale_pair_extrapolation import nested_matchings, sandwich, kl_bernoulli_upper
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import runpy
DES = runpy.run_path(os.path.join(os.path.dirname(__file__), "../novelty_20261016/run_extrapolation_suite.py"), run_name="lib")["DESIGNS"]
DES = dict(DES); DES["null2"] = (2, lambda X: np.column_stack([np.full(len(X), .6), np.full(len(X), .4)]))
NAMES = ["linear2", "smooth5", "sharp2", "sparse10", "K4_3d", "checker_coarse", "checker_fine", "rowwise_random", "null2"]


def model_upper(Xtr, ytr, Xte, yte, K, alpha):
    best = 1.0
    for mdl in (LogisticRegression(max_iter=2000), HistGradientBoostingClassifier(max_iter=200)):
        pr = mdl.fit(Xtr, ytr).predict_proba(Xte); Y = np.eye(K)[yte]
        b = ((pr - Y) ** 2).sum(1) / 2          # Brier/2 in [0,1]; E = Gbar/2 + E|p-q|^2/2 ... >= Gbar/2
        best = min(best, 2 * kl_bernoulli_upper(float(b.mean()), len(b), alpha / 2))
    return best


def run(name, rep, n=4000):
    r = np.random.default_rng(55_000 + 1000 * NAMES.index(name) + rep)
    d, law = DES[name]; X = r.random((n, d)); P = law(X); K = P.shape[1]
    y = np.array([r.choice(K, p=p) for p in P]) if K > 2 else (r.random(n) < P[:, 1]).astype(int)
    G = float((1 - (P ** 2).sum(1)).mean()); R = float((1 - P.max(1)).mean())
    I = float(((P - P.mean(0)) ** 2).sum(1).mean()); S = 1 - float((P.mean(0) ** 2).sum())
    sw = sandwich(nested_matchings(X, 4), y, K, .05, .5)
    h = n // 2; mu = model_upper(X[:h], y[:h], X[h:], y[h:], K, .05)
    return dict(design=name, rep=rep, G=G, R=R, I=I, BSS=I / S, GL=sw["brier_floor"][0], GU=sw["brier_floor"][1],
                RL=sw["floor01"][0], RU=sw["floor01"][1], IL=sw["info"][0], IU=sw["info"][1], BL=sw["bss"][0],
                BU=sw["bss"][1], model_GU=mu)


if __name__ == "__main__":
    reps = int(sys.argv[1]); names = sys.argv[2].split(",") if len(sys.argv) > 2 else NAMES
    for nm in names:
        for rep in range(reps):
            print(json.dumps(run(nm, rep)), flush=True)
