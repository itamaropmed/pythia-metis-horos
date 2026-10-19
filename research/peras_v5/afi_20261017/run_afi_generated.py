import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from peras.core.agreement_field import afi_certificate
from generator import make, MECH
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix


def brier(pr, y, K):
    return float(((pr - np.eye(K)[y]) ** 2).sum(1).mean())


def crossfit(M, y, K):
    n = len(y); folds = np.arange(n) % 2; pr = np.zeros((n, K))
    for f in (0, 1):
        tr, te = folds != f, folds == f
        m = LogisticRegression(C=2, max_iter=3000).fit(M[tr], y[tr]); pr[np.ix_(te, m.classes_)] = m.predict_proba(M[te])
    return pr


def run(name, rep, n=8000):
    d = make(name, n, 10_000 * (MECH.index(name) + 1) + rep); y, P, K, cell = d["y"], d["P"], d["K"], d["cell"]
    Xs = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(d["notes"])
    E = TruncatedSVD(64, random_state=0).fit_transform(Xs); E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    c = afi_certificate(cell, E, y, K, .05, levels=3, seed=rep)
    k = c["keep"]; Pk = P[k]; ck = cell[k]
    S = 1 - float((Pk.mean(0) ** 2).sum()); GXT = float((1 - (Pk ** 2).sum(1)).mean())
    pc = pd.DataFrame(Pk).groupby(ck).transform("mean").to_numpy(); GX = float((1 - (pc ** 2).sum(1)).mean())
    onehot = pd.get_dummies(pd.Series(cell).astype(str)).to_numpy(float)
    pr_tab = crossfit(onehot, y, K); pr_all = crossfit(hstack([csr_matrix(onehot), Xs]).tocsr(), y, K)
    return dict(dataset=name, rep=rep, K=K, true_IX=S - GX, true_dI=GX - GXT, true_GX=GX, true_GXT=GXT,
                IX_L=c["I_X"][0], IX_U=c["I_X"][1], dI_L=c["dI_lower"], GX_L=c["G_X"][0], GX_U=c["G_X"][1],
                GXT_U=c["G_XT_upper"], IT_L=c["I_T_lower"], p_text_nothing=c["p_text_adds_nothing"],
                spectrum=[round(v, 4) for v in c["spectrum"]], coinfo_point=c["design_coinfo_point"],
                model_added_value=brier(pr_tab, y, K) - brier(pr_all, y, K))


if __name__ == "__main__":
    reps = int(sys.argv[1]); names = sys.argv[2].split(",") if len(sys.argv) > 2 else MECH
    for nm in names:
        for rep in range(reps):
            o = run(nm, rep); print(json.dumps(o), flush=True)
