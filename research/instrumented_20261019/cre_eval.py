"""Conditional residual embeddings (CRE) on testbed v2 (known truth): do LLM embeddings of the note -- alone or
jointly with the serialised table -- explain the variance the table leaves, and do they sharpen the bound and the
risk labeler?  FIT trains every projection (PCA, ridge, stack weights); CAL certifies; TEST audits.

Readers (all label-free on CAL/TEST):
  emb_note        text-embedding-3-small(note), PCA-64 fitted on FIT
  emb_joint       text-embedding-3-small("specialty=..., table prediction=... min" + note), PCA-64
  cre_note        conditional residual embedding: [E64, E16 x table-cell one-hot]  (the residual direction may
                  depend on the table context; shrunk to the shared direction by the ridge)
  llm_extract, openmed_*, oracle (reference) for comparison
Outcomes: (a) duration residual r = y - f(table): variance explained on TEST vs the true explainable share;
          (b) W30 miss: Pythia certified risk information (CAL), residual-ridge forecast MAE vs true risk (TEST),
              certified risk labeler with forecast bins (m = 250 declared in advance), audited on TEST truth."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
from testbed_v2 import load, EFFECT
import reader_features as rf
from metis.reader_gain import refine_and_certify
from pythia.core.instrumented import loo_centre_groups, projection_bound, true_spread, tame_scores
from horos.represent.score_learning import variational_scores as vr, oof_scores, stack_weights, stack_scores, _centre
from split_safety import center_by_training_groups

d = load(); S, cell, miss, p, split, y, f = d["S"], d["cell"], d["miss"], d["p_miss"], d["split"], d["y"], d["f"]
fit, cal, te = split == "fit", split == "cal", split == "test"
U = Path("/mnt/user-data/uploads/PycharmProjects/pythia-metis-horos/research/instrumented_20261019/results/reader_cache")


def emb(variant):
    E = {}
    for l in open(U / f"emb_{variant}.jsonl"):
        r = json.loads(l); E[r["i"]] = r["e"]
    return np.array([E[int(i)] for i in S], np.float32)


def pca_fit(X, k):
    mu = X[fit].mean(0); _, _, Vt = np.linalg.svd(X[fit] - mu, full_matrices=False); return (X - mu) @ Vt[:k].T


En, Ej = emb("note"), emb("joint")
Pn, Pj = pca_fit(En, 64), pca_fit(Ej, 64)
cells = sorted(set(cell)); oh = (cell[:, None] == np.array(cells)[None]).astype(float)
cre = np.column_stack([Pn, (Pn[:, :16, None] * oh[:, None, :]).reshape(len(S), -1)])
R = {"emb_note(pca64)": Pn, "emb_joint(pca64)": Pj, "cre_note(table-conditioned)": cre,
     "llm_extract": rf.llm_extract(S), "openmed_disease": rf.ner_bag(S, "disease", fit_rows=fit), "openmed_pharma": rf.ner_bag(S, "pharma", fit_rows=fit),
     "tfidf_svd200": rf.tfidf(d["T"], fit_rows=fit), "oracle(reference)": np.column_stack([d["C"][k] for k in d["C"]])}
real = [k for k in R if not k.startswith("oracle")]
out = {"variance": {}, "gate": {}, "forecast": {}, "certified": {}}

# ---- (a) variance of the duration residual explained by the note (TEST), vs truth
cellu = np.unique(cell); r0 = np.random.default_rng(20261023)
tab = pd.Series(cell).map(pd.Series(r0.normal(0, 15, len(cellu)), index=cellu)).to_numpy()
mu = 90 + tab + .08 * d["C"]["ebl"] + sum(EFFECT[k] * d["C"][k] for k in EFFECT)
r = y - f
true_expl = np.var(_centre((mu - f)[:, None], cell)[te]) / np.var(_centre(r[:, None], cell)[te])
out["variance"]["true_explainable_share_of_within_cell_residual_variance"] = float(true_expl)


def r2(keys, lam=10.):
    Xs = [R[k] for k in keys]
    rc = center_by_training_groups(r[:, None], cell, fit)[:, 0]
    preds = []
    for X in Xs:
        Xc = center_by_training_groups(X.astype(float), cell, fit); sd = Xc[fit].std(0) + 1e-9; Xc = Xc / sd
        b = np.linalg.solve(Xc[fit].T @ Xc[fit] + lam * np.eye(Xc.shape[1]), Xc[fit].T @ rc[fit]); preds.append(Xc @ b)
    if len(preds) > 1:                                            # stack on FIT out-of-fold (2-fold)
        fo = np.random.default_rng(0).permutation(fit.sum()) % 2; P_oof = np.zeros((fit.sum(), len(Xs)))
        idx = np.flatnonzero(fit)
        for j, X in enumerate(Xs):
            for k in (0, 1):
                tr, ev = idx[fo != k], idx[fo == k]
                Xc = center_by_training_groups(X.astype(float), cell, tr)
                rc_fold = center_by_training_groups(r[:, None], cell, tr)[:, 0]
                sd = Xc[tr].std(0) + 1e-9; Xc = Xc / sd
                b = np.linalg.solve(Xc[tr].T @ Xc[tr] + lam * np.eye(Xc.shape[1]), Xc[tr].T @ rc_fold[tr]); P_oof[fo == k, j] = Xc[ev] @ b
        w = np.linalg.lstsq(P_oof, rc[fit], rcond=None)[0]; pred = np.column_stack(preds) @ w
    else:
        pred = preds[0]
    return float(1 - np.mean((rc[te] - pred[te]) ** 2) / np.var(rc[te]))


for k in R:
    out["variance"][k] = r2([k])
out["variance"]["STACK(all real readers)"] = r2(real)
print("share of within-cell duration-residual variance explained on TEST (truth max %.3f):" % true_expl)
for k, v in out["variance"].items():
    if not k.startswith("true"):
        print(f"  {k:32s} {v:+.3f}")

# ---- (b) miss: gate, forecasts, certified labeler
J = len(R) + 1
H = {k: vr(X[fit], miss[fit], cell[fit], X[cal], cell[cal], 2) for k, X in R.items()}
for k in R:
    out["gate"][k] = projection_bound(tame_scores(H[k], cell[cal]), miss[cal], .05 / J, groups=cell[cal], entity_ids=S[cal]).bound
w = stack_weights([oof_scores(vr, R[k][fit], miss[fit], cell[fit], 2, entity_ids=S[fit]) for k in real], miss[fit], cell[fit])
out["gate"]["STACK(all real readers)"] = projection_bound(tame_scores(stack_scores([H[k] for k in real], w), cell[cal]), miss[cal], .05 / J, groups=cell[cal], entity_ids=S[cal]).bound
out["gate"]["_truth_CAL"] = float(true_spread(np.column_stack([1 - p[cal], p[cal]]), cell[cal]))
print("gate (certified risk information, CAL; truth %.4f):" % out["gate"]["_truth_CAL"],
      {k: round(v, 5) for k, v in out["gate"].items() if not k.startswith("_")})
base = pd.Series(cell).map(pd.Series(miss[fit]).groupby(cell[fit]).mean()).fillna(miss[fit].mean()).to_numpy()


def forecast(keys):
    Xs = [R[k] for k in keys]
    Hall = [vr(X[fit], miss[fit], cell[fit], X, cell, 2) for X in Xs]
    oof = [oof_scores(vr, X[fit], miss[fit], cell[fit], 2, entity_ids=S[fit]) for X in Xs]
    ww = stack_weights(oof, miss[fit], cell[fit]) if len(Xs) > 1 else np.array([1.])
    ho, h = stack_scores(oof, ww)[:, 1], stack_scores(Hall, ww)[:, 1]
    a = float((ho * (miss[fit] - base[fit])).sum() / max((ho ** 2).sum(), 1e-12))
    return np.clip(base + a * h, 0, 1)


F = {"bins_table": base}
F.update({k: forecast([k]) for k in R}); F["STACK(all real readers)"] = forecast(real)
for k, ph in F.items():
    out["forecast"][k] = {"MAE_vs_true_risk": float(np.abs(ph[te] - p[te]).mean()), "brier": float(((ph[te] - miss[te]) ** 2).mean())}
    print(f"forecast {k:32s} MAE vs truth {out['forecast'][k]['MAE_vs_true_risk']:.4f}  Brier {out['forecast'][k]['brier']:.4f}")
glob_t = p[te].mean()
for k in ["STACK(all real readers)", "emb_note(pca64)", "emb_joint(pca64)", "cre_note(table-conditioned)", "llm_extract", "oracle(reference)"]:
    ph = F[k]; edges = np.unique(np.quantile(ph[fit], [1 / 3, 2 / 3])); bc, bt = np.digitize(ph[cal], edges), np.digitize(ph[te], edges)
    res = refine_and_certify({"cell": cell[cal]}, bc, S[cal].astype(str), miss[cal].astype(float), min_entities=250)
    cert = wrong = 0
    for key in np.unique(res["region_of_row"]):
        node = res["registry"].nodes.get(f"model|region|{key}")
        if node is None or not node.eligible or node.contrast_lower is None:
            continue
        parts = key.split("|"); rows = res["region_of_row"] == key
        if parts[0] != "reader":
            continue
        rb = int(parts[2][1:]) if parts[1] == "cell" else int(parts[1][1:])
        sel = bt == rb
        if parts[1] == "cell":
            sel &= cell[te] == "|".join(parts[3:])
        if not sel.any():
            continue
        t = p[te][sel].mean()
        if node.contrast_lower > 0:
            cert += rows.sum(); wrong += rows.sum() * (t <= glob_t)
        elif node.contrast_upper < 0:
            cert += rows.sum(); wrong += rows.sum() * (t >= glob_t)
    out["certified"][k] = {"regions": res["regions"], "certified_share": res["certified_share"], "wrong_on_test_truth": int(wrong), "certified_rows": int(cert)}
    print(f"certified {k:30s} regions {res['regions']} share {res['certified_share']:.3f} wrong {wrong}/{cert}")
(HERE / "results/cre_eval.json").write_text(json.dumps(out, indent=1, default=float))
