"""Planted-target certificate on the notes Jev answered (run stopped at the user's request; no further spend).
Subset = the first notes in file order that Jev answered (label-free selection), same rows for every reader."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parents[1]))
import runpy
from horos.represent.jev_reader import to_features
from pythia.core.instrumented import _groups, loo_centre_groups, projection_bound, true_spread
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ns = runpy.run_path(str(HERE / "mtsamples_certify.py"), run_name="lib") if False else None
# rebuild the planted design exactly as mtsamples_certify.py does (same seed and order)
src = (HERE / "mtsamples_certify.py").read_text()
head = src[: src.index("def crossfit")]
g = {"__file__": str(HERE / "mtsamples_certify.py")}; exec(compile(head, "mtsamples_certify_head", "exec"), g)
T, P, y, cell, anes, ebl, comp = g["T"], g["P"], g["y"], g["cell"], g["anes"], g["ebl"], g["comp"]
ans = {}
for l in open(HERE / "results/reader_cache/jev_mtsamples.jsonl"):
    d = json.loads(l)
    if d.get("ok"):
        ans[d["i"]] = d["answers"]
S = np.array(sorted(ans)); n = len(S)
CACHE = HERE / "results/reader_cache"
tf = TruncatedSVD(200, random_state=0).fit_transform(
    TfidfVectorizer(min_df=5, ngram_range=(1, 2), max_features=20000, sublinear_tf=True).fit_transform(T))
feats = {"jev_surgical_questions": np.vstack([to_features(ans[i]) for i in S]),
         "tfidf_svd200": tf[S],
         "oracle_concepts": np.column_stack([anes, ebl >= 100, comp]).astype(float)[S],
         "openmed_ner_all": np.column_stack([np.load(f) for f in sorted(CACHE.glob("openmed_ner_*.npy"))])[S]}
feats["jev_plus_openmed"] = np.column_stack([feats["jev_surgical_questions"], feats["openmed_ner_all"]])
yS, cS, PS = y[S], cell[S], P[S]
folds = np.random.default_rng(11).permutation(n) % 2; J = len(feats)
res = {"n_answered": int(n), "true_nli_subset": true_spread(PS, cS), "alpha": .05, "J": J, "readers": {}}
for k, X in feats.items():
    Xc = np.column_stack([pd.get_dummies(cS).to_numpy(float), X]); Q = np.zeros((n, 2)); L = 0.
    for f in (0, 1):
        tr, te = folds != f, folds == f
        Q[te] = LogisticRegression(C=1., max_iter=3000).fit(Xc[tr], yS[tr]).predict_proba(Xc[te])
    for f in (0, 1):
        I = np.flatnonzero(folds == f)
        group_i = _groups(cS[I])
        L += len(I) / n * projection_bound(loo_centre_groups(Q[I], group_i), yS[I],
                                            .05 / (2 * J), groups=group_i).bound
    res["readers"][k] = L; print(k, round(L, 5), flush=True)
# agreement of Jev's anesthesia answer with the planted ground truth
pa = feats["jev_surgical_questions"][:, 0]
res["jev_general_anesthesia_vs_planted"] = {"auc_like_mean_p_if_true": float(pa[anes[S] == 1].mean()),
                                            "mean_p_if_false": float(pa[anes[S] == 0].mean())}
(HERE / "results/jev_subset_certify.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
