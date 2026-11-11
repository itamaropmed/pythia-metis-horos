"""Risk-predictor development on testbed v2 (known truth): does each reader improve the per-case risk forecast,
and does the certified layer stay correct?  FIT trains, CAL certifies, TEST audits against the exact P(miss).

Forecasts of P(miss | case):
  bins_table       Metis baseline: FIT miss rate of the case's table cell
  gbm[readers]     gradient-boosted classifier of miss on (cell one-hot + reader features), FIT only
Certified layer (declared before outcomes: 3 forecast bins with edges from FIT forecasts, m = 250):
  local-risk v2 on CAL (support partition, eligible ledger, contrast); every certified direction audited on TEST truth.
Pythia gate: certified risk-information per reader and for the Rayleigh-stacked readers (variational scores)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
from testbed_v2 import load
import reader_features as rf
from metis.reader_gain import refine_and_certify
from pythia.core.instrumented import loo_centre_groups, projection_bound, true_spread
from horos.represent.score_learning import variational_scores_cv as vs, oof_scores, stack_weights, stack_scores
from sklearn.ensemble import HistGradientBoostingClassifier

d = load(); S, cell, miss, p, split = d["S"], d["cell"], d["miss"], d["p_miss"], d["split"]
fit, cal, te = split == "fit", split == "cal", split == "test"
oh = pd.get_dummies(cell).to_numpy(float)
R = {"tfidf_svd200": rf.tfidf(d["T"], fit_rows=fit), "llm_extract(gpt-6-luna)": rf.llm_extract(S)}
for k in ("disease", "pharma", "anatomy"):
    R[f"openmed_{k}"] = rf.ner_bag(S, k, fit_rows=fit)
for q in ("q0", "q1", "q2", "q3"):
    R[f"jev_{q}"] = rf.jev(S, qset=q)
R["oracle_concepts(reference)"] = np.column_stack([d["C"][k] for k in d["C"]])
R = {k: v for k, v in R.items() if v is not None}
print("readers:", list(R), flush=True)

out = {"n": int(d["n"]), "rows": {s: int((split == s).sum()) for s in ("fit", "cal", "test")}, "forecast": {}, "gate": {}, "certified": {}}
base_bins = pd.Series(cell).map(pd.Series(miss[fit]).groupby(cell[fit]).mean()).fillna(miss[fit].mean()).to_numpy()


def report(name, phat):
    out["forecast"][name] = {"test_MAE_vs_true_risk": float(np.abs(phat[te] - p[te]).mean()),
                             "test_brier": float(((phat[te] - miss[te]) ** 2).mean())}
    print(f"{name:34s} MAE vs truth {out['forecast'][name]['test_MAE_vs_true_risk']:.4f}  Brier {out['forecast'][name]['test_brier']:.4f}", flush=True)


def gbm(X):
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=40,
                                       l2_regularization=1.).fit(X[fit], miss[fit])
    return m.predict_proba(X)[:, 1]


report("bins_table (Metis baseline)", base_bins)
report("gbm[table]", gbm(oh))
F = {}
real = [k for k in R if not k.startswith("oracle")]
if "--gbm" in sys.argv:
    for k, X in R.items():
        report(f"gbm[table+{k}]", gbm(np.column_stack([oh, X])))

# ---- residual-ridge forecasts: p_hat = cell rate (FIT) + a * stacked variational score (low variance)
from horos.represent.score_learning import _centre


def residual_forecast(keys):
    Xs = [R[k] for k in keys]
    rows_all = np.ones(len(miss), bool)
    from horos.represent.score_learning import variational_scores as vr       # fixed ridge: scale-stable for forecasting
    H_all = [vr(X[fit], miss[fit], cell[fit], X, cell, 2) for X in Xs]          # FIT-trained, applied to all rows
    # This text testbed is one synthetic case per row; production clinical data
    # must supply patient/entity IDs rather than row IDs here.
    oof = [oof_scores(vr, X[fit], miss[fit], cell[fit], 2,
                      entity_ids=d["S"][fit]) for X in Xs]
    w = stack_weights(oof, miss[fit], cell[fit]) if len(Xs) > 1 else np.array([1.])
    h_oof = stack_scores(oof, w)[:, 1]; h = stack_scores(H_all, w)[:, 1]
    r_fit = miss[fit] - base_bins[fit]
    a = float((h_oof * r_fit).sum() / max((h_oof ** 2).sum(), 1e-12))          # scale fitted on FIT out-of-fold
    return np.clip(base_bins + a * h, 0, 1)


for k in R:
    F[k] = residual_forecast([k]); report(f"residual_ridge[{k}]", F[k])
F["all_real"] = residual_forecast(real); report("residual_ridge[STACK all real readers]", F["all_real"])

# ---- Pythia gate on CAL (risk-information beyond the table cells), one alpha ledger
J = len(R) + 1; gate = {}
B = {k: vs(X[fit], miss[fit], cell[fit], X[cal], cell[cal], 2) for k, X in R.items()}
for k, H in B.items():
    gate[k] = projection_bound(loo_centre_groups(H, cell[cal]), miss[cal], .05 / J, groups=cell[cal]).bound
w = stack_weights([oof_scores(vs, R[k][fit], miss[fit], cell[fit], 2,
                              entity_ids=d["S"][fit]) for k in real], miss[fit], cell[fit])
gate["STACK(all real readers)"] = projection_bound(loo_centre_groups(stack_scores([B[k] for k in real], w), cell[cal]),
                                                   miss[cal], .05 / J, groups=cell[cal]).bound
gate["_truth_on_CAL"] = true_spread(np.column_stack([1 - p[cal], p[cal]]), cell[cal])
gate["_stack_weights"] = dict(zip(real, map(float, w)))
out["gate"] = gate
print("gate:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in gate.items()}, flush=True)

# ---- certified layer on CAL with forecast bins, audited on TEST truth
glob_true_test = p[te].mean()
for name in ["bins_table (no reader)", "all_real"] + real + ["oracle_concepts(reference)"]:
    if name.startswith("bins_table"):
        bins_cal = None; ph = None
    else:
        ph = F[name]; edges = np.unique(np.quantile(ph[fit], [1 / 3, 2 / 3])); bins_cal = np.digitize(ph[cal], edges)
    res = refine_and_certify({"cell": cell[cal]}, bins_cal, S[cal].astype(str), miss[cal].astype(float), min_entities=250)
    reg = res["registry"]; cert_rows = wrong = 0; regions = res["region_of_row"]
    te_bins = None if ph is None else np.digitize(ph[te], edges)
    for key in np.unique(regions):
        node = reg.nodes.get(f"model|region|{key}")
        if node is None or not node.eligible or node.contrast_lower is None:
            continue
        rows_cal = regions == key
        # TEST rows routed to the same region (reader bin r* and/or cell)
        parts = key.split("|")
        if ph is not None and parts[0] == "reader":
            rb = int(parts[2][1:]) if parts[1] == "cell" else int(parts[1][1:])
            sel = te_bins == rb
            if parts[1] == "cell":
                sel &= cell[te] == "|".join(parts[3:])
        else:
            sel = cell[te] == "|".join(parts[1:]) if parts[0] == "cell" else np.ones(te.sum(), bool)
        if not sel.any():
            continue
        t = p[te][sel].mean()
        if node.contrast_lower > 0:
            cert_rows += rows_cal.sum(); wrong += rows_cal.sum() * (t <= glob_true_test)
        elif node.contrast_upper < 0:
            cert_rows += rows_cal.sum(); wrong += rows_cal.sum() * (t >= glob_true_test)
    out["certified"][name] = {"regions": res["regions"], "certified_share_cal": res["certified_share"],
                              "certified_rows": int(cert_rows), "wrong_direction_on_test_truth": int(wrong)}
    print(f"certified {name:30s} regions {res['regions']:2d} share {res['certified_share']:.3f} wrong {wrong}/{cert_rows}", flush=True)
(HERE / "results/metis_risk_v2.json").write_text(json.dumps(out, indent=1, default=float))
