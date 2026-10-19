"""REAL data: MTSamples (2,357 unique notes, real specialty labels).  The 'table' is a structured extraction from
each note (what a clinical NLP pipeline would populate): note type (from section headers), age band, sex, length
band.  New task -- EXTRACTION-COMPLETENESS CERTIFICATION: how much outcome information does the free text still
carry beyond the structured fields, certified without any model, along a ladder of increasingly rich tables."""
import sys, os, json, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from peras.core.agreement_field import afi_certificate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix

df = pd.read_csv(__import__("os").environ.get("PERAS_DATA", "/home/claude/data") + "/mtsamples.csv").fillna("")
key = df.transcription.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
df = df[(key != "") & ~key.map(lambda s: hashlib.md5(s.encode()).hexdigest()).duplicated()]
df = df.sample(frac=1, random_state=20261017).reset_index(drop=True)          # file is sorted by specialty
T = df.transcription; up = T.str.upper(); low = T.str.lower(); n = len(df)
age = low.str.extract(r"(\d{1,3})[- ]year[- ]old")[0].astype(float)
ageb = np.select([age.isna(), age < 18, age < 40, age < 65], [0, 1, 2, 3], 4)
she = low.str.count(r"\b(she|her|female|woman)\b"); he = low.str.count(r"\b(he|his|him|male|man)\b")
sex = np.select([she > he, he > she], [0, 1], 2)
ntype = np.select([up.str.contains("PROCEDURE|OPERATION|ANESTHESIA|PREOPERATIVE"),
                   up.str.contains("DISCHARGE"),
                   up.str.contains("HISTORY OF PRESENT ILLNESS|CHIEF COMPLAINT"),
                   up.str.contains("IMPRESSION|FINDINGS")], [0, 1, 2, 3], 4)
lenb = pd.qcut(T.str.len(), 3, labels=False).to_numpy()
LADDER = {"T0_no_table": np.zeros(n, int), "T1_note_type": ntype, "T2_type_age_sex": ntype * 15 + ageb * 3 + sex,
          "T3_type_age_sex_len": (ntype * 15 + ageb * 3 + sex) * 3 + lenb}
Xs = TfidfVectorizer(min_df=3, sublinear_tf=True).fit_transform(T)
E = TruncatedSVD(100, random_state=0).fit_transform(Xs); E /= np.linalg.norm(E, axis=1, keepdims=True)
spec = df.medical_specialty.str.strip()
LABELS = {"specialty_top10_other": pd.factorize(spec.where(spec.isin(spec.value_counts().index[:10]), "other"))[0],
          "surgery_vs_not": (spec == "Surgery").astype(int).to_numpy()}


def crossfit_brier(M, y, K):
    folds = np.arange(n) % 2; pr = np.zeros((n, K))
    for f in (0, 1):
        tr, te = folds != f, folds == f
        m = LogisticRegression(C=2, max_iter=3000).fit(M[tr], y[tr]); pr[np.ix_(te, m.classes_)] = m.predict_proba(M[te])
    return float(((pr - np.eye(K)[y]) ** 2).sum(1).mean())


if __name__ == "__main__":
    for lname, y in LABELS.items():
        K = int(y.max() + 1)
        for tname, cell in LADDER.items():
            for seed in range(5):
                c = afi_certificate(cell, E, y, K, .05, levels=2, seed=seed)
                oh = pd.get_dummies(pd.Series(cell).astype(str)).to_numpy(float)
                av = crossfit_brier(oh, y, K) - crossfit_brier(hstack([csr_matrix(oh), Xs]).tocsr(), y, K) if seed == 0 else None
                o = dict(label=lname, K=K, table=tname, seed=seed, cells_used=c["cells_used"], n_used=c["n_used"],
                         IX=c["I_X"], dI_L=c["dI_lower"], GX=c["G_X"], GXT_U=c["G_XT_upper"], IT_L=c["I_T_lower"],
                         p_text_nothing=c["p_text_adds_nothing"], spectrum=[round(v, 4) for v in c["spectrum"]],
                         coinfo_point=c["design_coinfo_point"], model_added_value=av)
                print(json.dumps(o), flush=True)
