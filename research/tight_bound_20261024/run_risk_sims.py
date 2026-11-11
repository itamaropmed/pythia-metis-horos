"""Known-truth simulation of the per-case W30 risk labeler: which construction tracks the TRUE risk on held-out rows?
  bins           Metis cell miss rate (table only)
  logit_direct   logistic regression of the binary miss on [cell, reader features]   (learns risk from 0/1 outcomes)
  gbm_direct     gradient-boosted classifier on the same
  resid_gauss    residual forecast: ridge of the continuous residual on reader features (cell-centred), then
                 p = P(|rhat + e| > 30), e ~ N(0, s^2) with s from FIT out-of-fold errors   (the Metis default proposed)
Metrics on TEST vs the exact true risk: MAE, Pearson.  World as in run_bound_sims.world."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import norm, pearsonr
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bound_sims import world, ridge_score, NOISE  # noqa
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


def one(n, s, rho, seed):
    rng = np.random.default_rng(seed); g, mu, y, X = world(n, s, rho, False, rng)
    u = rng.random(n); fit, te = u < .4, u >= .7
    cm = {c: y[fit & (g == c)].mean() for c in np.unique(g)}; f = np.array([cm[c] for c in g]); r = y - f
    miss = (np.abs(r) > 30).astype(int); m = mu - f
    p = norm.cdf((-30 - m) / NOISE) + 1 - norm.cdf((30 - m) / NOISE)
    cells = np.unique(g); oh = (g[:, None] == cells[None]).astype(float)
    out = {}
    br = {c: miss[fit & (g == c)].mean() for c in cells}; out["bins"] = np.array([br[c] for c in g])
    Z = np.column_stack([oh, X])
    out["logit_direct"] = LogisticRegression(C=1., max_iter=2000).fit(Z[fit], miss[fit]).predict_proba(Z)[:, 1]
    out["gbm_direct"] = HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, min_samples_leaf=40).fit(Z[fit], miss[fit]).predict_proba(Z)[:, 1]
    rmean = np.array([r[fit & (g == c)].mean() for c in g])
    h = ridge_score(X[fit], r[fit], g[fit], X, g)
    idx = np.flatnonzero(fit); fo = rng.permutation(len(idx)) % 2; o = np.zeros(n)
    for k in (0, 1):
        a, b = idx[fo != k], idx[fo == k]; o[b] = ridge_score(X[a], r[a], g[a], X[b], g[b])
    sd = np.sqrt(np.mean((r[fit] - rmean[fit] - o[fit]) ** 2)); rh = rmean + h
    out["resid_gauss"] = norm.cdf((-30 - rh) / sd) + 1 - norm.cdf((30 - rh) / sd)
    return {k: (float(np.abs(v[te] - p[te]).mean()), float(pearsonr(v[te], p[te])[0]) if v[te].std() > 0 else 0.) for k, v in out.items()}


def main(reps=40):
    rows = []
    for n in (1500, 3400, 8000, 20000):
        for rho in (.5, 1.5):
            acc = {}
            for rep in range(reps):
                for k, v in one(n, 12., rho, 7000 + rep + n).items():
                    acc.setdefault(k, []).append(v)
            row = {"n": n, "reader_noise": rho, **{k: {"MAE": float(np.mean([a for a, _ in L])), "pearson": float(np.mean([b for _, b in L]))} for k, L in acc.items()}}
            rows.append(row); print(json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 3) for kk, vv in v.items()}) for k, v in row.items()}), flush=True)
    (Path(__file__).parent / "risk_sims.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
