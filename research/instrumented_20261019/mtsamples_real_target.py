"""Real target (no planted truth): which reader certifies the most information about the note's medical
specialty (6 classes) beyond a weak table (note-length decile)?  All bounds are valid lower bounds, so the
readers can be ranked; there is no 'truth' column here.  Readers: tfidf, medspaCy rules, each OpenMed NER model,
all OpenMed NER together.  Public MTSamples only."""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parents[1]))
from pythia.core.instrumented import _groups, loo_centre_groups, projection_bound  # noqa

ALPHA = .05
df = pd.read_csv(os.environ.get("PERAS_DATA", "/home/claude/data") + "/mtsamples.csv").fillna("")
spec = df["medical_specialty"].str.strip()
top = ["Cardiovascular / Pulmonary", "Orthopedic", "Gastroenterology", "Neurology", "Urology", "Obstetrics / Gynecology"]
keep = np.flatnonzero(spec.isin(top).to_numpy() & (df["transcription"].str.len() > 50).to_numpy())
T = df["transcription"].astype(str).to_numpy()[keep]
y = pd.Categorical(spec.to_numpy()[keep], categories=top).codes.astype(int); K = len(top); n = len(y)
cell = pd.qcut(pd.Series([len(t) for t in T]), 10, labels=False, duplicates="drop").to_numpy()
folds = np.random.default_rng(7).permutation(n) % 2
CACHE = HERE / "results" / "reader_cache"
feats = {"tfidf_svd200": TruncatedSVD(200, random_state=0).fit_transform(
    TfidfVectorizer(min_df=5, ngram_range=(1, 2), max_features=20000, sublinear_tf=True).fit_transform(T))}
for f in ["fast_rules"] + [p.stem for p in sorted(CACHE.glob("openmed_ner_*.npy"))]:
    x = np.load(CACHE / f"{f}.npy")
    feats["medspacy_context_rules" if f == "fast_rules" else f] = x[keep]
ner = [k for k in feats if k.startswith("openmed_ner_")]
feats["openmed_ner_all"] = np.column_stack([feats[k] for k in ner])
J = len(feats)


def crossfit(X):
    Q = np.zeros((n, K)); Xc = np.column_stack([np.eye(cell.max() + 1)[cell], X])
    for f in (0, 1):
        tr, te = folds != f, folds == f
        Q[te] = LogisticRegression(C=.5, max_iter=3000).fit(Xc[tr], y[tr]).predict_proba(Xc[te])
    return Q


res = {}
for k, X in feats.items():
    Q = crossfit(X); L = 0.
    for f in (0, 1):
        I = np.flatnonzero(folds == f)
        group_i = _groups(cell[I])
        L += len(I) / n * projection_bound(loo_centre_groups(Q[I], group_i), y[I],
                                            ALPHA / (2 * J), groups=group_i).bound
    res[k] = {"certified_info_beyond_length": L, "n_features": int(X.shape[1])}
    print(k, round(L, 4), flush=True)
out = {"n": n, "classes": top, "table": "note-length decile", "alpha": ALPHA, "J": J, "readers": res,
       "units": "Brier/Gini; max possible = within-cell Gini of the specialty labels"}
(HERE / "results" / "mtsamples_specialty_readers.json").write_text(json.dumps(out, indent=1))
