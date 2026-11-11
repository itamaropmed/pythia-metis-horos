"""Bound-guided risk-labeler development on REAL MTSamples notes with a planted duration outcome (known truth).

Deployed model = table-only (specialty-cell means fitted on FIT notes).  Risk target = W30 miss of that model.
Readers (all already computed, no new spend): TF-IDF, medspaCy rules, each OpenMed NER model, all OpenMed,
Jev surgical questions.  Rows = the notes Jev answered (same rows for every reader).
Steps: GATE (certified risk-information gain per reader) -> REFINE the labeler with the best accepted reader
-> ACCOUNT (certified resolution before/after; every certified direction checked against the known truth).
"""
import json, sys
from pathlib import Path

import numpy as np, pandas as pd
from scipy.stats import norm

HERE = Path(__file__).parent; ROOT = HERE.parents[1]; sys.path.insert(0, str(ROOT))
from horos.represent.jev_reader import to_features  # noqa
from metis.reader_gain import gate_readers, miss_probabilities, reader_bins, refine_and_certify  # noqa
from pythia.core.instrumented import _groups, true_spread  # noqa

src = (HERE / "mtsamples_certify.py").read_text()
g = {"__file__": str(HERE / "mtsamples_certify.py")}
exec(compile(src[: src.index("def crossfit")], "head", "exec"), g)
T, cell, anes, ebl, comp = g["T"], g["cell"], g["anes"], g["ebl"], g["comp"]
CACHE = HERE / "results/reader_cache"
ans = {}
for l in (open(CACHE / "jev_mtsamples.jsonl") if (CACHE / "jev_mtsamples.jsonl").exists() else []):
    d = json.loads(l)
    if d.get("ok"):
        ans[d["i"]] = d["answers"]
ROWS = HERE / "results/risk_rows.json"
S = np.array(sorted(ans)) if ans else np.array(json.load(open(ROWS)))   # same evaluation rows either way
n = len(S)
(HERE / "results/risk_rows.json").write_text(json.dumps(S.tolist()))     # the rows an LLM reader must cover
rng = np.random.default_rng(20261020)

# ---- planted duration-like outcome, table-only deployed model, miss indicator with exact truth
cells = cell[S]
tab = pd.Series(cells).map(pd.Series(rng.normal(0, 15, len(np.unique(cell))), index=np.unique(cell))).to_numpy()
mu = 90 + tab + 25 * anes[S] + .08 * np.minimum(ebl[S], 500) + 30 * comp[S]
y = mu + rng.normal(0, 20, n)
fit = rng.permutation(n) < n // 2; cal = ~fit
f_cell = pd.Series(y[fit]).groupby(cells[fit]).mean()
f = pd.Series(cells).map(f_cell).fillna(y[fit].mean()).to_numpy()
miss = (np.abs(y - f) > 30).astype(int)
d = mu - f
p_true = norm.cdf((-30 - d) / 20) + 1 - norm.cdf((30 - d) / 20)          # exact P(miss | table, note)

# ---- readers (label-free features of the note)
from sklearn.decomposition import TruncatedSVD  # noqa
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa
tf = TruncatedSVD(200, random_state=0).fit_transform(
    TfidfVectorizer(min_df=5, ngram_range=(1, 2), max_features=20000, sublinear_tf=True).fit_transform(T))
feats = {"tfidf_svd200": tf[S],
         "oracle_concepts (reference ceiling)": np.column_stack([anes[S], ebl[S] >= 100, comp[S]]).astype(float)}
if (CACHE / "fast_rules.npy").exists():
    feats["medspacy_rules"] = np.load(CACHE / "fast_rules.npy")[S]
if ans:
    feats["jev_surgical"] = np.vstack([to_features(ans[i]) for i in S])
for p in sorted(CACHE.glob("openmed_ner_*.npy")):
    feats[p.stem] = np.load(p)[S]
# LLM readers produced on the Mac through the opmed LiteLLM (llm_reader_mtsamples.py -> readers.jsonl)
rj = HERE / "readers.jsonl"
if rj.exists():
    by = {}
    for l in open(rj):
        r = json.loads(l)
        if r.get("ok"):
            by.setdefault(r["reader"], {})[r["i"]] = r.get("json") or {}
    for name, dd in by.items():
        cover = np.mean([int(i) in dd for i in S])
        if cover < .99:
            print("skip (incomplete on the evaluation rows):", name, cover); continue
        print(f"{name}: {cover:.4f} of evaluation rows read; unread rows get an uninformative (empty) read")
        def num(v, default):
            try:
                return float(v)
            except Exception:
                return default

        def val(j, k, opts):
            return [float(str(j.get(k, "")).lower() == o) for o in opts]
        J_ = [dd.get(int(i), {}) for i in S]
        if "judge" in name:
            feats["llm:" + name] = np.array([[num(j.get("p_long"), 50.) / 100] for j in J_])
        else:
            feats["llm:" + name] = np.array([val(j, "anesthesia", ["general", "regional", "local", "mac", "none"]) +
                                             [min(num(j.get("blood_loss_ml"), 0.), 2000) / 500] +
                                             val(j, "complications", ["yes"]) + val(j, "emergency", ["yes"]) +
                                             [num(j.get("complexity_1to5"), 3.) / 5] for j in J_])
_ner = [k for k in feats if k.startswith("openmed_ner_")]
if _ner:
    feats["openmed_ner_all"] = np.column_stack([feats[k] for k in _ner])

cf, cc = cells[fit], cells[cal]
probs, probs_fit = {}, {}
for k, X in feats.items():
    q, qf = miss_probabilities(X[fit], miss[fit], X[cal], cf, cc, C=.5)
    probs[k], probs_fit[k] = q, qf

# 1. GATE
gate = gate_readers(cc, miss[cal], probs)
truth_gain = true_spread(np.column_stack([1 - p_true[cal], p_true[cal]]), cc)
print("true risk information beyond the labeler cells:", round(truth_gain, 5))
for r in gate:
    print(f"  {r.reader:28s} certified gain {r.certified_gain:.5f}  accepted={r.accepted}")

# 2-3. REFINE + ACCOUNT
ent = np.arange(n)[cal].astype(str)
coarse = np.where(pd.Series(cc).isin(pd.Series(cc).value_counts().index[:5]), cc, "other")
base = {"cell": cc, "coarse": coarse}
res = {"n_rows": int(n), "cal_rows": int(cal.sum()), "true_risk_information": truth_gain,
       "gate": [r.__dict__ for r in gate], "labelers": {}}


def audit(out):
    reg_rows = out["region_of_row"]; glob_true = p_true[cal].mean(); wrong = 0; cert = 0
    for key in np.unique(reg_rows):
        rows = reg_rows == key
        node = out["registry"].nodes.get(f"model|region|{key}")
        if node is None or not node.eligible or node.contrast_lower is None:
            continue
        t = p_true[cal][rows].mean()
        if node.contrast_lower > 0:
            cert += rows.sum(); wrong += rows.sum() * (t <= glob_true)
        elif node.contrast_upper < 0:
            cert += rows.sum(); wrong += rows.sum() * (t >= glob_true)
    return int(cert), int(wrong)


for name in [None] + [r.reader for r in gate if r.accepted][:3]:
    bins = None if name is None else reader_bins(probs_fit[name][:, 1], probs[name][:, 1], 3)
    out = refine_and_certify(base, bins, ent, miss[cal].astype(float), min_entities=100)
    c, w = audit(out)
    tag = "baseline (table cells only)" if name is None else f"+ {name}"
    res["labelers"][tag] = {"regions": out["regions"], "certified_share": out["certified_share"],
                            "relation_counts": out["relation_counts"], "certified_rows": c,
                            "certified_rows_wrong_vs_truth": w}
    print(f"{tag:34s} regions {out['regions']:3d}  certified share {out['certified_share']:.3f}  wrong {w}/{c}")
(HERE / "results/risk_labeler_gain.json").write_text(json.dumps(res, indent=1, default=float))

if "--debug" in sys.argv:
    name = gate[0].reader
    bins = reader_bins(probs_fit[name][:, 1], probs[name][:, 1], 3)
    out = refine_and_certify(base, bins, ent, miss[cal].astype(float), min_entities=100)
    for k, nd in out["registry"].nodes.items():
        if nd.eligible:
            print(k, nd.n_entities, round(nd.estimate, 3), round(nd.lower, 3), round(nd.upper, 3), nd.contrast_lower and round(nd.contrast_lower, 3), nd.contrast_upper and round(nd.contrast_upper, 3))
    print("bin sizes", np.bincount(bins), "true P(miss) by bin", [round(p_true[cal][bins == b].mean(), 3) for b in range(3)],
          "alpha per tail", out["registry"].alpha_per_tail_node_model)

if "--variants" in sys.argv:
    for name in [k for k in probs if k != "medspacy_rules"]:
        bins = reader_bins(probs_fit[name][:, 1], probs[name][:, 1], 3)
        for m in (100, 300, 500):
            for lab, b in (("cells+reader", base), ("reader-first (coarse only)", {"coarse": coarse})):
                out = refine_and_certify(b, bins, ent, miss[cal].astype(float), min_entities=m)
                c, w = audit(out)
                print(f"{name:14s} m={m:3d} {lab:28s} regions {out['regions']:2d} certified {out['certified_share']:.3f} wrong {w}/{c}")

if "--baseline-grid" in sys.argv:
    for m in (100, 300, 500):
        out = refine_and_certify(base, None, ent, miss[cal].astype(float), min_entities=m); c, w = audit(out)
        print(f"baseline m={m} regions {out['regions']} certified {out['certified_share']:.3f} wrong {w}/{c}")
