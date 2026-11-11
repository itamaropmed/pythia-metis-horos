"""Binary miss vs continuous capped error as the risk target: certified risk information per reader on CAL
(FIT trains every score; known truth E[|y-f| capped | note] computed by Monte Carlo)."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
from testbed_v2 import load, EFFECT
import reader_features as rf
from pythia.core.instrumented import projection_bound_scalar, loo_centre_scalar, tame_scores, projection_bound, loo_centre_groups, true_spread
from horos.represent.score_learning import variational_scores_scalar as vss, variational_scores as vs
d = load(); S, cell, split = d["S"], d["cell"], d["split"]; fit, cal = split == "fit", split == "cal"
CAP = 120.; z = np.minimum(np.abs(d["y"] - d["f"]), CAP) / CAP
# exact-ish truth for E[z | table, note]: mu - f is known up to the noise N(0, 20^2) -> Monte Carlo
mu = d["y"] - (d["y"] - d["f"]) + 0     # placeholder (recomputed below)
rng = np.random.default_rng(0)
tab_mu = d["f"] + 0
# recompute mu exactly as in testbed_v2 (same seed): mean duration given table + note
import pandas as pd
cellu = np.unique(d["cell"]); r0 = np.random.default_rng(20261023)
tab = pd.Series(d["cell"]).map(pd.Series(r0.normal(0, 15, len(cellu)), index=cellu)).to_numpy()
mu = 90 + tab + .08 * d["C"]["ebl"] + sum(EFFECT[k] * d["C"][k] for k in EFFECT)
eps = rng.normal(0, 20, (4000, 1)); m_true = (np.minimum(np.abs(mu - d["f"] + eps), CAP) / CAP).mean(0)
R = {"tfidf_svd200": rf.tfidf(d["T"], fit_rows=fit), "llm_extract": rf.llm_extract(S), "openmed_disease": rf.ner_bag(S, "disease", fit_rows=fit),
     "openmed_pharma": rf.ner_bag(S, "pharma", fit_rows=fit), "openmed_anatomy": rf.ner_bag(S, "anatomy", fit_rows=fit),
     "oracle(reference)": np.column_stack([d["C"][k] for k in d["C"]])}
q2 = rf.llm_qa(S[fit], "q2")
J = len(R); out = {"truth_continuous_V_cal": float(true_spread(m_true[cal][:, None], cell[cal])),
                   "truth_binary_V_cal": float(true_spread(np.column_stack([1 - d["p_miss"][cal], d["p_miss"][cal]]), cell[cal]))}
for k, X in R.items():
    raw = vss(X[fit], z[fit], cell[fit], X[cal], cell[cal])
    bc = projection_bound_scalar(loo_centre_scalar(raw, cell[cal]), z[cal], 1.0, .05 / (2 * J),
                                 groups=cell[cal]).bound
    bct = projection_bound_scalar(tame_scores(raw, cell[cal]), z[cal], 1.0, .05 / (2 * J),
                                  groups=cell[cal]).bound
    Hb = vs(X[fit], d["miss"][fit], cell[fit], X[cal], cell[cal], 2)
    bb = projection_bound(loo_centre_groups(Hb, cell[cal]), d["miss"][cal], .05 / (2 * J), groups=cell[cal]).bound
    bbt = projection_bound(tame_scores(Hb, cell[cal]), d["miss"][cal], .05 / (2 * J), groups=cell[cal]).bound
    tc, tb = out["truth_continuous_V_cal"], out["truth_binary_V_cal"]
    out[k] = {"continuous": bc / tc, "continuous_tamed": bct / tc, "binary": bb / tb, "binary_tamed": bbt / tb}
    print(f"{k:20s} share of truth: continuous {bc/tc:.2f} tamed {bct/tc:.2f} | binary {bb/tb:.2f} tamed {bbt/tb:.2f}", flush=True)
print("truth V: continuous", round(out["truth_continuous_V_cal"], 5), "binary", round(out["truth_binary_V_cal"], 5))
(HERE / "results/continuous_risk_gate.json").write_text(json.dumps(out, indent=1, default=float))
