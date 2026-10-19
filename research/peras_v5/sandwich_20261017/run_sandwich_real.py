"""Model-free sandwich on REAL labels (no gold needed for the assumption-free sides).
MTSamples: unique notes, specialty (top-10 + other, K = 11; and all 40), several outcome-blind geometries incl. a
null control (random geometry) and a leakage geometry (keywords field).  PMLB: pharynx Status / Grade and
lowbwt LOW from covariates (birth weight itself excluded: LOW is derived from it).
Accepted comparator for the upper side: held-out Brier of fitted classifiers (logistic regression, gradient
boosting), KL-certified."""
import sys, os, json, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from peras.core.multiscale_pair_extrapolation import nested_matchings, sandwich, kl_bernoulli_upper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


def brier_upper(pr, y, K, alpha):
    b = ((pr - np.eye(K)[y]) ** 2).sum(1) / 2
    return 2 * kl_bernoulli_upper(float(b.mean()), len(b), alpha)


def unit(X):
    X = np.asarray(X, float); return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def mtsamples():
    df = pd.read_csv(__import__("os").environ.get("PERAS_DATA", "/home/claude/data") + "/mtsamples.csv").fillna("")
    key = df.transcription.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[(key != "") & ~key.map(lambda s: hashlib.md5(s.encode()).hexdigest()).duplicated()].reset_index(drop=True)
    # the CSV is SORTED BY SPECIALTY: shuffle rows (label-independent seed) so that truncation, folds and any
    # tie-breaking in the matchings are outcome-blind; matchings additionally use perm_seed
    df = df.sample(frac=1, random_state=20261017).reset_index(drop=True)
    n = (len(df) // 16) * 16; df = df.iloc[:n]; T = df.transcription; low = T.str.lower()
    spec = df.medical_specialty.str.strip()
    TERMS = ["diabet", "hypertens", "cancer|carcinoma|malignan", "fractur", "infect", "pain", "fever", "nausea", "chest",
             "abdom", "knee", "hip", "spine|lumbar|cervical", "cardiac|heart", "renal|kidney", "lung|pulmonary", "stroke",
             "seizure", "depress", "asthma", "copd", "anemia", "obes", "allerg", "pregnan", "surgery|operative", "biopsy",
             "mri", "ct scan|cat scan", "x-ray|xray|radiograph"]
    svd = lambda M: unit(TruncatedSVD(100, random_state=0).fit_transform(M))
    geoms = {
        "word_tfidf_svd": svd(TfidfVectorizer(min_df=3, sublinear_tf=True).fit_transform(T)),
        "char4_tfidf_svd": svd(TfidfVectorizer(analyzer="char", ngram_range=(4, 4), min_df=3, sublinear_tf=True, max_features=60000).fit_transform(T)),
        "concept_dictionary_30": np.column_stack([low.str.contains(t).to_numpy(float) for t in TERMS]),
        "log_length_only": np.log(T.str.len().to_numpy(float))[:, None],
        "random_geometry_null": np.random.default_rng(0).normal(size=(n, 10)),
        "keywords_field_LEAK": svd(TfidfVectorizer(min_df=2).fit_transform(df.keywords + " ")),
    }
    out = []
    for labname in ("top10_plus_other", "all_40"):
        lab = spec.where(spec.isin(spec.value_counts().index[:10]), "other") if labname.startswith("top") else spec
        y = pd.factorize(lab)[0]; K = int(y.max() + 1)
        # accepted upper: best of three fitted classifiers on word TF-IDF (2-fold cross-fitting -> held-out Brier on
        # every row): LR with C tuned by inner 3-fold CV, LR C=4, distance-weighted 15-NN (cosine); alpha/3 each
        from sklearn.model_selection import GridSearchCV
        from sklearn.neighbors import KNeighborsClassifier
        Xw = TfidfVectorizer(min_df=3, sublinear_tf=True).fit_transform(T); folds = np.arange(n) % 2
        mods = {"lr_tuned": lambda: GridSearchCV(LogisticRegression(max_iter=3000), {"C": [.5, 2, 8, 32]}, cv=3, scoring="neg_log_loss"),
                "lr_C4": lambda: LogisticRegression(C=4, max_iter=3000),
                "knn15": lambda: KNeighborsClassifier(15, weights="distance", metric="cosine")}
        mU = {}
        for mname, mk in mods.items():
            pr = np.zeros((n, K))
            for f in (0, 1):
                tr, te = folds != f, folds == f
                m = mk().fit(Xw[tr], y[tr]); cls = m.classes_ if hasattr(m, "classes_") else m.best_estimator_.classes_
                pr_f = np.zeros((te.sum(), K)); pr_f[:, cls] = m.predict_proba(Xw[te]); pr[te] = pr_f
            mU[mname] = brier_upper(pr, y, K, .05 / 3)
        mU_best = min(mU.values())
        # model-GUIDED pairing (new): geometry = cross-fitted predicted probability vectors.  Rows of fold f are paired
        # using a model trained on the other fold only (outcome-blind for fold f), so the upper bound stays
        # assumption-free and needs no calibration of the model; folds certified separately at alpha/2 and averaged.
        for mname in ("lr_tuned", "knn15"):
            GUs, ILs = [], []
            for f in (0, 1):
                tr, te = folds != f, folds == f
                m = mods[mname]().fit(Xw[tr], y[tr]); cls = m.classes_ if hasattr(m, "classes_") else m.best_estimator_.classes_
                q = np.zeros((te.sum(), K)); q[:, cls] = m.predict_proba(Xw[te])
                nf = (te.sum() // 16) * 16; sw = sandwich(nested_matchings(q[:nf], 4, perm_seed=1), y[te][:nf], K, .025, .5)
                GUs.append(sw["brier_floor"][1]); ILs.append(sw["info"][0])
            out.append(dict(data="MTSamples", label=labname, K=K, geometry=f"model_guided_{mname}", n=n, GL=None,
                            GU=float(np.mean(GUs)), RU=float(min(np.mean(GUs), (K - 1) / K)), IL=float(np.mean(ILs)),
                            model_brier_upper=mU_best, model_uppers=mU))
            print(json.dumps(out[-1]), flush=True)
        for gname, X in geoms.items():
            sw = sandwich(nested_matchings(X, 4, perm_seed=1), y, K, .05, .5)
            out.append(dict(data="MTSamples", label=labname, K=K, geometry=gname, n=n, GL=sw["brier_floor"][0],
                            GU=sw["brier_floor"][1], RL=sw["floor01"][0], RU=sw["floor01"][1], IL=sw["info"][0],
                            IU=sw["info"][1], BSS_L=sw["bss"][0], BSS_U=sw["bss"][1], Q_all=sw["Q_all"],
                            Q_levels=sw["Q_levels"], model_brier_upper=mU_best, model_uppers=mU))
            print(json.dumps(out[-1]), flush=True)
    return out


def pmlb(reps=10, n=4096):
    out = []
    ph = pd.read_csv(__import__("os").environ.get("PERAS_DATA", "/home/claude/data") + "/1196_BNG_pharynx.tsv.gz", sep="\t")
    lb = pd.read_csv(__import__("os").environ.get("PERAS_DATA", "/home/claude/data") + "/1193_BNG_lowbwt.tsv.gz", sep="\t")
    tasks = [("pharynx", ph, "Status", ["Inst", "sex", "Treatment", "Grade", "Condition", "Site", "T", "N"], ["Age"]),
             ("pharynx", ph, "Grade", ["Inst", "sex", "Treatment", "Condition", "Site", "T", "N", "Status"], ["Age"]),
             ("lowbwt", lb, "LOW", ["RACE", "SMOKE", "PTL", "HT", "UI", "FTV"], ["AGE", "LWT"])]
    for data, D, target, cats, nums in tasks:
        y_all = pd.factorize(D[target])[0]; K = int(y_all.max() + 1)
        Z = np.column_stack([pd.get_dummies(D[c].astype(str)).to_numpy(float) for c in cats] +
                            [((D[c] - D[c].mean()) / D[c].std()).to_numpy()[:, None] for c in nums])
        for rep in range(reps):
            r = np.random.default_rng(800 + rep); idx = r.choice(len(D), n, replace=False)
            tr = r.choice(np.setdiff1d(np.arange(len(D)), idx), min(20000, len(D) - n), replace=False)
            X, y = Z[idx], y_all[idx]
            sw = sandwich(nested_matchings(X, 4, perm_seed=1), y, K, .05, .5)
            mus = []
            for mdl in (LogisticRegression(max_iter=3000), HistGradientBoostingClassifier(max_iter=300)):
                mdl.fit(Z[tr], y_all[tr]); mus.append(brier_upper(mdl.predict_proba(X), y, K, .025))
            out.append(dict(data=data, label=target, K=K, geometry="covariates", rep=rep, n=n, GL=sw["brier_floor"][0],
                            GU=sw["brier_floor"][1], RL=sw["floor01"][0], RU=sw["floor01"][1], IL=sw["info"][0],
                            IU=sw["info"][1], BSS_L=sw["bss"][0], BSS_U=sw["bss"][1], Q_all=sw["Q_all"],
                            Q_levels=sw["Q_levels"], model_brier_upper=min(mus)))
            print(json.dumps(out[-1]), flush=True)
    return out


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = (mtsamples() if which in ("all", "mts") else []) + (pmlb() if which in ("all", "pmlb") else [])
    json.dump(res, open(os.path.join(here, f"sandwich_real_{which}.json"), "w"), indent=1)
