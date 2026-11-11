"""From the residual-explaining readers to (i) a certified duration-information bound and (ii) a risk labeler.
(i)  Pythia scalar projection bound on the capped signed duration residual r = clip(y - f, -120, 120):
     certified V = Var_within-cell(E[r | table, note]) in min^2, per reader and for the FIT-stacked readers.
(ii) Risk from the residual forecast: p_hat = P(|r_hat + e| > 30), e ~ N(0, s^2), s from FIT out-of-fold errors.
     This turns explained duration variance into per-case W30 risk; audited against the exact true risk on TEST,
     then certified with forecast bins (m = 250, declared in advance)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
exec(open(HERE / "cre_eval.py").read().split("# ---- (a)")[0])          # same data, readers and split
from split_safety import center_by_training_groups
from pythia.core.instrumented import projection_bound_scalar, tame_scores
from metis.reader_gain import refine_and_certify
cellu = np.unique(cell); r0 = np.random.default_rng(20261023)
tab = pd.Series(cell).map(pd.Series(r0.normal(0, 15, len(cellu)), index=cellu)).to_numpy()
mu = 90 + tab + .08 * d["C"]["ebl"] + sum(EFFECT[k] * d["C"][k] for k in EFFECT)
CAP = 120.; rr = np.clip(y - f, -CAP, CAP); z = rr + CAP                 # z in [0, 2 CAP]
out = {"duration_bound_min2": {}, "risk": {}, "certified": {}}
truth_V = float(np.var(_centre((mu - f)[:, None], cell)[cal]))           # within-cell variance of the true mean residual
out["duration_bound_min2"]["_truth_CAL"] = truth_V


def ridge_dir(X, target, rows_tr, lam=10.):
    Xc = center_by_training_groups(X.astype(float), cell, rows_tr)
    sd = Xc[rows_tr].std(0) + 1e-9; Xc = Xc / sd
    tc = center_by_training_groups(np.asarray(target)[:, None], cell, rows_tr)[:, 0]
    b = np.linalg.solve(Xc[rows_tr].T @ Xc[rows_tr] + lam * np.eye(Xc.shape[1]), Xc[rows_tr].T @ tc[rows_tr])
    return Xc @ b


def stacked_pred(keys, target):
    preds = [ridge_dir(R[k], target, fit) for k in keys]
    if len(keys) == 1:
        return preds[0]
    idx = np.flatnonzero(fit); fo = np.random.default_rng(0).permutation(len(idx)) % 2; P = np.zeros((len(idx), len(keys)))
    for j, k in enumerate(keys):
        for q in (0, 1):
            m = np.zeros(len(S), bool); m[idx[fo != q]] = True
            P[fo == q, j] = ridge_dir(R[k], target, m)[idx[fo == q]]
    target_fit_centered = center_by_training_groups(np.asarray(target)[:, None], cell, fit)[:, 0]
    w = np.linalg.lstsq(P, target_fit_centered[idx], rcond=None)[0]
    return np.column_stack(preds) @ w


J = len(R) + 1; preds = {}
for k in list(R) + ["STACK(all real readers)"]:
    h = stacked_pred(real if k.startswith("STACK") else [k], rr); preds[k] = h
    b = projection_bound_scalar(tame_scores(h[cal], cell[cal]), z[cal], 2 * CAP, .05 / J).bound  # superseded by cre_bound_final.py (eb_scalar_bound)
    out["duration_bound_min2"][k] = b
print("certified within-cell variance of E[duration residual | note] (min^2, CAL; truth %.1f):" % truth_V)
for k, v in out["duration_bound_min2"].items():
    if not k.startswith("_"):
        print(f"  {k:32s} {v:8.2f}  ({v / truth_V:.2f} of truth)")
# ---- (ii) risk from residual forecast
base_err = rr[fit] - pd.Series(rr[fit]).groupby(cell[fit]).transform("mean").to_numpy()
cell_mean = pd.Series(rr[fit]).groupby(cell[fit]).mean()
for k in ["none"] + list(preds):
    rhat = pd.Series(cell).map(cell_mean).fillna(0).to_numpy() + (0 if k == "none" else preds[k])
    s = np.sqrt(np.mean((rr[fit] - rhat[fit]) ** 2))                    # (in-sample on FIT; conservative-ish)
    ph = norm.cdf((-30 - rhat) / s) + 1 - norm.cdf((30 - rhat) / s)
    out["risk"][k] = {"MAE_vs_true_risk": float(np.abs(ph[te] - p[te]).mean()), "brier": float(((ph[te] - miss[te]) ** 2).mean())}
    print(f"risk from residual forecast [{k:30s}] MAE vs true risk {out['risk'][k]['MAE_vs_true_risk']:.4f}  Brier {out['risk'][k]['brier']:.4f}")
    if k in ("none", "STACK(all real readers)", "llm_extract", "emb_joint(pca64)", "oracle(reference)"):
        edges = np.unique(np.quantile(ph[fit], [1 / 3, 2 / 3])); bc, bt = np.digitize(ph[cal], edges), np.digitize(ph[te], edges)
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
        out["certified"][k] = {"regions": res["regions"], "certified_share": res["certified_share"], "wrong_on_test_truth": int(wrong)}
        print(f"    certified labeler: regions {res['regions']} share {res['certified_share']:.3f} wrong {wrong}/{cert}")
(HERE / "results/cre_eval2.json").write_text(json.dumps(out, indent=1, default=float))
