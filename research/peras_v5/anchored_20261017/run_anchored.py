"""Plasmode on real clinical notes (MTSamples, 2,352 unique notes) joined to a synthetic tabular record:
  age band (from the note's 'NN-year-old' when present, else random), sex (pronouns when present, else random),
  lab_high (more likely when the note mentions diabetes) -> 5 x 2 x 2 = 20 table cells.
Planted label law p(X, T) (known), four regimes of how the text relates to the table:
  S0 table_only  : Y depends on X only                      -> DeltaI(T|X) = 0 (false-positive check)
  S1 redundant   : Y depends on 'diabet' (mostly mirrored by lab_high in the table) + X
  S2 unique      : Y depends on X + 'smok' + dose mentions (not in the table)
  S3 synergy     : Y depends on (sex x 'pain') and (age band x note length) interactions
Certified (assumption-free) lower bound on the text information beyond the table vs the accepted model-based
'added value' (cross-fitted logistic regression Brier: table vs table+text), which is an estimate."""
import sys, os, json, re, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from peras.core.anchored_information import anchored_information
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix

df = pd.read_csv(__import__("os").environ.get("PERAS_DATA", "/home/claude/data") + "/mtsamples.csv").fillna("")
key = df.transcription.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
df = df[(key != "") & ~key.map(lambda s: hashlib.md5(s.encode()).hexdigest()).duplicated()]
df = df.sample(frac=1, random_state=7).reset_index(drop=True)          # file is sorted by specialty
T = df.transcription; low = T.str.lower(); n = len(df)
Xs = TfidfVectorizer(min_df=3, sublinear_tf=True).fit_transform(T)
E = TruncatedSVD(100, random_state=0).fit_transform(Xs); E /= np.linalg.norm(E, axis=1, keepdims=True)
age_txt = low.str.extract(r"(\d{1,3})[- ]year[- ]old")[0].astype(float)
she = low.str.count(r"\b(she|her)\b"); he = low.str.count(r"\b(he|his|him)\b")
f_diab = low.str.contains("diabet").to_numpy(float); f_smok = low.str.contains("smok").to_numpy(float)
f_dose = (low.str.count(r"\d+\s?mg").clip(upper=5) / 5).to_numpy(); f_pain = low.str.contains("pain").to_numpy(float)
loglen = (np.log(T.str.len().to_numpy(float)) - 7.7)


def make_table(r):
    age = np.where(age_txt.notna(), age_txt.fillna(0), r.integers(1, 90, n))
    ageb = np.digitize(age, [18, 40, 60, 75])
    sex = np.where(she > he, 1, np.where(he > she, 0, r.integers(0, 2, n)))
    lab = (r.random(n) < np.where(f_diab > 0, .85, .2)).astype(int)
    return ageb, sex, lab


def law(scen, ageb, sex, lab):
    base = .5 * (ageb - 2) + .6 * sex - .3 + 1.0 * lab
    if scen == "S0_table_only":
        lin = base
    elif scen == "S1_redundant":
        lin = base + 1.2 * f_diab
    elif scen == "S2_unique":
        lin = base + 1.6 * f_smok + 2.0 * f_dose
    elif scen == "S3_synergy":
        lin = base + 2.0 * (2 * sex - 1) * (f_pain - .5) + 1.5 * (ageb - 2) * loglen
    else:   # S4_strong_unique: text dominates
        lin = .3 * base + 4.0 * (f_pain - .5) + 3.0 * f_smok + 3.0 * f_dose - 1.0
    return 1 / (1 + np.exp(-lin))


def model_added_value(onehot, y):
    folds = np.arange(n) % 2; out = {}
    for name, M in (("table", onehot), ("table+text", hstack([csr_matrix(onehot), Xs]).tocsr())):
        pr = np.zeros(n)
        for f in (0, 1):
            tr, te = folds != f, folds == f
            pr[te] = LogisticRegression(C=2, max_iter=3000).fit(M[tr], y[tr]).predict_proba(M[te])[:, 1]
        out[name] = float((2 * (pr - y) ** 2).mean())       # multiclass Brier for K = 2
    return out["table"] - out["table+text"]


if __name__ == "__main__":
    reps = int(sys.argv[1]); here = os.path.dirname(os.path.abspath(__file__))
    from sklearn.neighbors import NearestNeighbors
    def knn_score(cells, y, folds, f, k=15):
        sc = np.zeros(n)
        for c in np.unique(cells):
            tr = np.flatnonzero((cells == c) & (folds != f)); te = np.flatnonzero((cells == c) & (folds == f))
            if len(te) == 0 or len(tr) == 0:
                continue
            idx = NearestNeighbors(n_neighbors=min(k, len(tr))).fit(E[tr]).kneighbors(E[te], return_distance=False)
            sc[te] = y[tr][idx].mean(1)
        return sc
    for scen in ("S0_table_only", "S1_redundant", "S2_unique", "S3_synergy", "S4_strong_unique"):
        for rep in range(reps):
            r = np.random.default_rng(4242 + rep + 1000 * len(scen))
            ageb, sex, lab = make_table(r); cells = ageb * 4 + sex * 2 + lab
            q = law(scen, ageb, sex, lab); y = (r.random(n) < q).astype(int)
            G_XT = float((2 * q * (1 - q)).mean())
            qbar = pd.Series(q).groupby(cells).transform("mean").to_numpy(); G_X = float((2 * qbar * (1 - qbar)).mean())
            dI = G_X - G_XT
            res = anchored_information(cells, E, y, 2, .05, seed=rep)
            folds = np.random.default_rng(rep).integers(0, 2, n); gl, gp = [], []
            for f in (0, 1):
                te = folds == f; sc = knn_score(cells, y, folds, f)
                g = anchored_information(cells[te], sc[te][:, None], y[te], 2, .025, seed=rep)
                gl.append(g["dI_lower"] * te.mean()); gp.append(g["dI_point"] * te.mean())
            onehot = pd.get_dummies(pd.DataFrame(dict(a=ageb, s=sex, l=lab)).astype(str)).to_numpy(float)
            o = dict(scen=scen, rep=rep, dI_true=dI, G_X=G_X, G_XT=G_XT, dI_lower=res["dI_lower"], dI_point=res["dI_point"],
                     GX_L=res["GX"][0], GX_U=res["GX"][1], GXT_U=res["GXT_upper"], model_added_value=model_added_value(onehot, y),
                     cells_used=res["cells_used"], n_used=res["n_used"],
                     guided_dI_lower=float(sum(gl)), guided_dI_point=float(sum(gp)))
            print(json.dumps(o), flush=True)
