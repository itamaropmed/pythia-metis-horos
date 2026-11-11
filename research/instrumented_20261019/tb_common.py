"""Shared testbed-v2 state for the bound / risk-labeler / agent experiments (known truth).
Readers (all label-free), the fixed FIT/CAL/TEST split, ridge + stacking predictors, risk metrics, certified labeler."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm, spearmanr, pearsonr

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
_g = {"__file__": str(HERE / "cre_eval.py"), "__name__": "cre_eval_header"}
exec(open(HERE / "cre_eval.py").read().split("# ---- (a)")[0], _g)
d, S, cell, y, f, p, miss = (_g[k] for k in ("d", "S", "cell", "y", "f", "p", "miss"))
fit, cal, te, R = _g["fit"], _g["cal"], _g["te"], _g["R"]
_centre = _g["_centre"]
import symbolic_reader as sr
from testbed_v2 import EFFECT
SYM = sr.features(S)
for k in ("rr_shift", "rr_typed", "rr_atoms"):
    R[k] = SYM[k]
OLD = [k for k in R if not k.startswith(("rr_", "oracle"))]
NEW = OLD + ["rr_shift", "rr_typed"]                      # rr_atoms excluded: over-fits (see symbolic_eval)
cellu = np.unique(cell)
_r0 = np.random.default_rng(20261023)
tab = pd.Series(cell).map(pd.Series(_r0.normal(0, 15, len(cellu)), index=cellu)).to_numpy()
mu = 90 + tab + .08 * d["C"]["ebl"] + sum(EFFECT[k] * d["C"][k] for k in EFFECT)
r = y - f; n = len(S); T = d["T"]
rc = _centre(r[:, None], cell)[:, 0]
true_dir = _centre((mu - f)[:, None], cell)[:, 0]


def ridge_fit_pred(X, target, tr, lam=10.):
    Xc = _centre(np.asarray(X, float), cell); sd = Xc[tr].std(0) + 1e-9; Xc = Xc / sd
    tc = _centre(target[:, None], cell)[:, 0]
    b = np.linalg.solve(Xc[tr].T @ Xc[tr] + lam * np.eye(Xc.shape[1]), Xc[tr].T @ tc[tr]); return Xc @ b


def predictor(keys, tr, target=None, seed=0, extra=None):
    """Ridge per reader + stacking weights from 2-fold OOF inside the training rows (predictions for all rows).
    extra: dict name -> feature matrix for readers not in R (e.g. agent concepts)."""
    target = r if target is None else target
    Rx = dict(R); Rx.update(extra or {})
    P = np.column_stack([ridge_fit_pred(Rx[k], target, tr) for k in keys])
    if len(keys) == 1:
        return P[:, 0]
    idx = np.flatnonzero(tr); fo = np.random.default_rng(seed).permutation(len(idx)) % 2; O = np.zeros((len(idx), len(keys)))
    for q in (0, 1):
        m = np.zeros(n, bool); m[idx[fo != q]] = True
        O[fo == q] = np.column_stack([ridge_fit_pred(Rx[k], target, m) for k in keys])[idx[fo == q]]
    w = np.linalg.lstsq(O, _centre(target[:, None], cell)[idx, 0], rcond=None)[0]
    return P @ w


def oof(keys, tr, seed=1, extra=None):
    idx = np.flatnonzero(tr); fo = np.random.default_rng(seed).permutation(len(idx)) % 2; o = np.zeros(n)
    for q in (0, 1):
        m = np.zeros(n, bool); m[idx[fo != q]] = True
        o[idx[fo == q]] = predictor(keys, m, extra=extra)[idx[fo == q]]
    return o


def r2(h, rows):
    return float(1 - np.mean((rc[rows] - h[rows]) ** 2) / np.var(rc[rows]))


def gauss_risk(rhat, s):
    return norm.cdf((-30 - rhat) / s) + 1 - norm.cdf((30 - rhat) / s)


cell_mean_fit = pd.Series(cell).map(pd.Series(r[fit]).groupby(cell[fit]).mean()).fillna(0).to_numpy()


def risk_forecast(keys, extra=None):
    """Residual-forecast risk: p = P(|rhat + e| > 30), e ~ N(0, s^2), s from FIT out-of-fold errors."""
    if not keys:
        rh = ro = cell_mean_fit
    else:
        rh = cell_mean_fit + predictor(keys, fit, extra=extra); ro = cell_mean_fit + oof(keys, fit, extra=extra)
    s = np.sqrt(np.mean((r[fit] - ro[fit]) ** 2))
    return gauss_risk(rh, s), rh


def metrics(ph, rows=None):
    rows = te if rows is None else rows; t = p[rows]; q = ph[rows]
    return {"MAE_vs_true": float(np.abs(q - t).mean()), "pearson": float(pearsonr(q, t)[0]) if q.std() > 0 else 0.,
            "spearman": float(spearmanr(q, t)[0]) if q.std() > 0 else 0., "brier": float(((q - miss[rows]) ** 2).mean()),
            "calib_slope": float(np.polyfit(q, t, 1)[0]) if q.std() > 0 else 0.}


def certified_labeler(ph, m=250):
    from metis.reader_gain import refine_and_certify
    edges = np.unique(np.quantile(ph[fit], [1 / 3, 2 / 3])); bc, bt = np.digitize(ph[cal], edges), np.digitize(ph[te], edges)
    res = refine_and_certify({"cell": cell[cal]}, bc, S[cal].astype(str), miss[cal].astype(float), min_entities=m)
    cert = wrong = 0; glob_t = p[te].mean(); dirs = []
    for key in np.unique(res["region_of_row"]):
        node = res["registry"].nodes.get(f"model|region|{key}"); parts = key.split("|")
        if node is None or not node.eligible or node.contrast_lower is None or parts[0] != "reader":
            continue
        rb = int(parts[2][1:]) if parts[1] == "cell" else int(parts[1][1:]); sel = bt == rb
        if parts[1] == "cell":
            sel &= cell[te] == "|".join(parts[3:])
        rows = res["region_of_row"] == key; t = p[te][sel].mean() if sel.any() else glob_t
        if node.contrast_lower > 0:
            cert += rows.sum(); wrong += rows.sum() * (t <= glob_t); dirs.append((key, "higher", int(rows.sum()), float(t)))
        elif node.contrast_upper < 0:
            cert += rows.sum(); wrong += rows.sum() * (t >= glob_t); dirs.append((key, "lower", int(rows.sum()), float(t)))
    return {"certified_share": float(res["certified_share"]), "wrong_on_test_truth": int(wrong), "directions": dirs,
            "test_global_true_risk": float(glob_t)}
