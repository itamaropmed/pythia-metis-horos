"""Testbed v2 (known truth): the table-conditioned, embedding-free symbolic reader vs the previous readers, for
(a) explained residual variance and duration MAE (TEST), (b) the certified duration information (CAL; zero-shot also
on FIT u CAL) with the old (Maurer-Pontil) and new (betting) inequalities, (c) the per-case W30 risk labeler:
accuracy and held-out correlation with the TRUE risk, (d) the certified labeler.
FIT trains/tunes, CAL certifies, TEST audits.  The declared outcome cap for certificates is 60 min."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm, spearmanr, pearsonr
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
__file__ = str(HERE / "cre_eval.py")
exec(open(HERE / "cre_eval.py").read().split("# ---- (a)")[0])           # d, S, cell, y, f, p, miss, R, real, splits
import symbolic_reader as sr
from pythia.core.tight_bounds import scalar_cert, tune_cert, tame, centre_within, crossfit_certificate
from metis.reader_gain import refine_and_certify

SYM = sr.features(S)
for k in ("rr_shift", "rr_typed", "rr_atoms"):
    R[k] = SYM[k]
OLD = [k for k in real if not k.startswith("rr_")]
NEW = OLD + ["rr_shift", "rr_typed", "rr_atoms"]
cellu = np.unique(cell); r0 = np.random.default_rng(20261023)
tab = pd.Series(cell).map(pd.Series(r0.normal(0, 15, len(cellu)), index=cellu)).to_numpy()
mu = 90 + tab + .08 * d["C"]["ebl"] + sum(EFFECT[k] * d["C"][k] for k in EFFECT)
r = y - f; CAP = 60.; n = len(S); out = {}


def ridge_fit_pred(X, target, tr, lam=10.):
    Xc = _centre(X.astype(float), cell); sd = Xc[tr].std(0) + 1e-9; Xc = Xc / sd
    tc = _centre(target[:, None], cell)[:, 0]
    b = np.linalg.solve(Xc[tr].T @ Xc[tr] + lam * np.eye(Xc.shape[1]), Xc[tr].T @ tc[tr]); return Xc @ b


def predictor(keys, tr, target=r, seed=0):
    """Ridge per reader + stacking weights from 2-fold OOF inside the training rows. Returns predictions for all rows."""
    P = np.column_stack([ridge_fit_pred(R[k], target, tr) for k in keys])
    if len(keys) == 1:
        return P[:, 0]
    idx = np.flatnonzero(tr); fo = np.random.default_rng(seed).permutation(len(idx)) % 2; O = np.zeros((len(idx), len(keys)))
    for q in (0, 1):
        m = np.zeros(n, bool); m[idx[fo != q]] = True
        O[fo == q] = np.column_stack([ridge_fit_pred(R[k], target, m) for k in keys])[idx[fo == q]]
    w = np.linalg.lstsq(O, _centre(target[:, None], cell)[idx, 0], rcond=None)[0]
    return P @ w


def oof(keys, tr, seed=1):
    idx = np.flatnonzero(tr); fo = np.random.default_rng(seed).permutation(len(idx)) % 2; o = np.zeros(n)
    for q in (0, 1):
        m = np.zeros(n, bool); m[idx[fo != q]] = True
        o[idx[fo == q]] = predictor(keys, m)[idx[fo == q]]
    return o


rc = _centre(r[:, None], cell)[:, 0]
truth_share = float(np.var(_centre((mu - f)[:, None], cell)[te]) / np.var(rc[te]))
cands = {k: [k] for k in ["rr_shift", "rr_typed", "rr_atoms", "llm_extract", "emb_joint(pca64)", "openmed_disease"]}
cands.update({"STACK_old(real readers)": OLD, "STACK_new(+symbolic)": NEW, "oracle(reference)": ["oracle(reference)"]})
PRED = {k: predictor(v, fit) for k, v in cands.items()}
PRED["rr_shift(zero-shot, untrained)"] = SYM["rr_shift"][:, 0] - pd.Series(SYM["rr_shift"][:, 0]).groupby(cell).transform("mean").to_numpy()
# (a) variance explained + MAE (TEST)
out["a"] = {"_true_explainable_share": truth_share, "_table_only_MAE": float(np.abs(r[te]).mean()),
            "_true_mean_MAE": float(np.abs((y - mu)[te]).mean())}
for k, h in PRED.items():
    out["a"][k] = {"R2_within_cell": float(1 - np.mean((rc[te] - h[te]) ** 2) / np.var(rc[te])),
                   "MAE_min": float(np.abs(y[te] - f[te] - h[te]).mean()),
                   "corr_with_true_residual_mean": float(np.corrcoef(h[te], _centre((mu - f)[:, None], cell)[te, 0])[0, 1])}
    print(f"(a) {k:34s} R2 {out['a'][k]['R2_within_cell']:6.3f}  MAE {out['a'][k]['MAE_min']:6.2f}  corr(true) {out['a'][k]['corr_with_true_residual_mean']:.3f}")
print("    table-only MAE %.2f, true-mean MAE %.2f, explainable share %.3f" % (out["a"]["_table_only_MAE"], out["a"]["_true_mean_MAE"], truth_share))

# (b) certificates (declared cap 60).  Truth for the capped residual via Monte Carlo over the planted noise.
eps = np.random.default_rng(3).normal(0, 20, (3000, 1)); m_cap = np.clip(mu - f + eps, -CAP, CAP).mean(0)
def V(rows): return float(np.mean(centre_within(m_cap[rows], cell[rows]) ** 2))
out["b"] = {"_truth_CAL": V(cal), "_truth_FITuCAL": V(fit | cal)}
J = len(cands) + 1; a = .05 / J
for k in list(cands) + ["rr_shift(zero-shot, untrained)"]:
    zs = k.startswith("rr_shift(zero")
    res = {}
    if zs:      # untrained: tuning of q, lam on FIT, certify on CAL; and (no training at all) on FIT u CAL
        hraw = SYM["rr_shift"][:, 0]
        cfg = tune_cert(centre_within(hraw[fit], cell[fit]), r[fit], cell[fit], a, int(cal.sum()), caps=(CAP,))
        lab = fit | cal
        for meth in ("mp", "betting"):
            h = tame(hraw[cal], cell[cal], cfg["q"])
            res[f"CAL_{meth}"] = scalar_cert(h, r[cal], CAP, a, method=meth, lam=cfg["lam"] if meth == "betting" else None, groups=cell[cal]).bound
            h2 = tame(hraw[lab], cell[lab], .05)          # no outcome-based tuning on FIT u CAL: q, lam declared
            lam2 = None if meth != "betting" else .5
            res[f"FITuCAL_{meth}"] = scalar_cert(h2, r[lab], CAP, a, method=meth, lam=lam2, groups=cell[lab]).bound
    else:
        o = oof(cands[k], fit)
        cfg = tune_cert(centre_within(o[fit], cell[fit]), r[fit], cell[fit], a, int(cal.sum()), caps=(CAP,))
        for meth in ("mp", "betting"):
            h = tame(PRED[k][cal], cell[cal], cfg["q"])
            res[f"CAL_{meth}"] = scalar_cert(h, r[cal], CAP, a, method=meth, lam=cfg["lam"] if meth == "betting" else None, groups=cell[cal]).bound
        hp = tame(PRED[k][cal], cell[cal], cfg["q"]); res["plug_in_CAL"] = float(max(np.mean(hp * np.clip(r[cal], -CAP, CAP)), 0) ** 2 / np.mean(hp ** 2))
    out["b"][k] = res
    print(f"(b) {k:34s} " + "  ".join(f"{m} {v:6.1f}" for m, v in res.items()) + f"   [truth CAL {out['b']['_truth_CAL']:.1f}]")

# (c) risk labeler on TEST vs the true risk
def metrics(ph):
    t = p[te]; q = ph[te]
    slope = float(np.polyfit(q, t, 1)[0]) if q.std() > 0 else 0.
    return {"MAE_vs_true": float(np.abs(q - t).mean()), "pearson": float(pearsonr(q, t)[0]) if q.std() > 0 else 0.,
            "spearman": float(spearmanr(q, t)[0]) if q.std() > 0 else 0., "brier": float(((q - miss[te]) ** 2).mean()),
            "calib_slope(true~pred)": slope}

def gauss_risk(rhat, s):
    return norm.cdf((-30 - rhat) / s) + 1 - norm.cdf((30 - rhat) / s)

out["c"] = {}
bins = pd.Series(cell).map(pd.Series(miss[fit]).groupby(cell[fit]).mean()).to_numpy()
out["c"]["metis_bins"] = metrics(bins)
cm = pd.Series(r[fit]).groupby(cell[fit]).mean(); base = pd.Series(cell).map(cm).fillna(0).to_numpy()
oh = (cell[:, None] == cellu[None]).astype(float)
FORE = {}
for k in ["none", "rr_typed", "rr_atoms", "llm_extract", "STACK_old(real readers)", "STACK_new(+symbolic)", "oracle(reference)"]:
    rh = base + (0 if k == "none" else PRED[k])
    ro = base + (0 if k == "none" else oof(cands[k], fit))                    # FIT out-of-fold forecast
    s = np.sqrt(np.mean((r[fit] - ro[fit]) ** 2))                              # homoscedastic, OOF
    FORE[f"gauss[{k}]"] = gauss_risk(rh, s)
    # heteroscedastic scale: log squared OOF error ~ cell + |shift| + LLM tail prob (all label-free features)
    Z = np.column_stack([oh, np.abs(SYM["rr_shift"][:, 0]) / 30, SYM["_p_long"] + SYM["_p_short"]])
    e2 = np.log((r - ro) ** 2 + 1)
    b = np.linalg.lstsq(Z[fit], e2[fit], rcond=None)[0]
    s_i = np.sqrt(np.exp(Z @ b)); s_i *= s / np.sqrt(np.mean(s_i[fit] ** 2))
    FORE[f"gauss_hetero[{k}]"] = gauss_risk(rh, s_i)
# zero-shot LLM tail probabilities, logistic recalibration on FIT (2 parameters + bin offset)
from sklearn.linear_model import LogisticRegression
lg = lambda x: np.log(np.clip(x, 1e-3, 1 - 1e-3) / (1 - np.clip(x, 1e-3, 1 - 1e-3)))
pz = np.clip(SYM["_p_long"] + SYM["_p_short"], 1e-3, .999)
Zl = np.column_stack([lg(pz), lg(bins)])
FORE["llm_tail_prob(recalibrated)"] = LogisticRegression(C=1e3).fit(Zl[fit], miss[fit]).predict_proba(Zl)[:, 1]
FORE["llm_tail_prob(raw)"] = pz
for k, ph in FORE.items():
    out["c"][k] = metrics(ph)
for k, v in out["c"].items():
    print(f"(c) {k:42s} MAE {v['MAE_vs_true']:.4f}  pearson {v['pearson']:.3f}  spearman {v['spearman']:.3f}  brier {v['brier']:.4f}  slope {v['calib_slope(true~pred)']:.2f}")
print("    reference: true risk Brier %.4f" % float(((p[te] - miss[te]) ** 2).mean()))
out["c"]["_true_risk_brier"] = float(((p[te] - miss[te]) ** 2).mean())

# (d) certified labeler (forecast tertile bins from FIT, m = 250 declared), audited on TEST truth
out["d"] = {}
for k in ["gauss[STACK_old(real readers)]", "gauss_hetero[STACK_new(+symbolic)]", "gauss[STACK_new(+symbolic)]", "gauss[oracle(reference)]"]:
    ph = FORE[k]; edges = np.unique(np.quantile(ph[fit], [1 / 3, 2 / 3])); bc, bt = np.digitize(ph[cal], edges), np.digitize(ph[te], edges)
    res = refine_and_certify({"cell": cell[cal]}, bc, S[cal].astype(str), miss[cal].astype(float), min_entities=250)
    cert = wrong = 0; glob_t = p[te].mean()
    for key in np.unique(res["region_of_row"]):
        node = res["registry"].nodes.get(f"model|region|{key}"); parts = key.split("|")
        if node is None or not node.eligible or node.contrast_lower is None or parts[0] != "reader":
            continue
        rb = int(parts[2][1:]) if parts[1] == "cell" else int(parts[1][1:]); sel = bt == rb
        if parts[1] == "cell":
            sel &= cell[te] == "|".join(parts[3:])
        rows = res["region_of_row"] == key; t = p[te][sel].mean() if sel.any() else glob_t
        if node.contrast_lower > 0:
            cert += rows.sum(); wrong += rows.sum() * (t <= glob_t)
        elif node.contrast_upper < 0:
            cert += rows.sum(); wrong += rows.sum() * (t >= glob_t)
    out["d"][k] = {"certified_share": float(res["certified_share"]), "wrong_on_test_truth": int(wrong)}
    print(f"(d) {k:42s} certified share {res['certified_share']:.3f}  wrong {wrong}")
(HERE / "results/symbolic_eval.json").write_text(json.dumps(out, indent=1, default=float))
