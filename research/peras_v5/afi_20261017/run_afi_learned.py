"""Geometry portfolio for Delta I(T|X): unsupervised (TF-IDF -> SVD) vs CROSS-FITTED LEARNED METRIC (a model trained on
the other fold selects and weights the n-gram coordinates; it is outcome-blind for the rows it pairs).  Each geometry at
alpha/2; the learned geometry certifies each fold at alpha/4 and the fold bounds are summed with fold weights
(sum_f w_f DeltaI_f <= DeltaI because within-fold cell means only remove variance)."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from peras.core.agreement_field import afi_certificate
from generator import make, MECH
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression


def learned_dI(Xs, cell, y, K, alpha, seed, top=300, levels=3):
    n = len(y); folds = np.random.default_rng(seed).permutation(n) % 2; tot = 0.0; parts = []
    for f in (0, 1):
        tr, te = folds != f, folds == f
        m = LogisticRegression(C=2, max_iter=3000).fit(Xs[tr], y[tr])
        imp = np.abs(m.coef_).sum(0); sel = np.argsort(-imp)[:top]
        E = (Xs[te][:, sel].multiply(imp[sel][None, :])).toarray()
        E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        E = E + 1e-9 * np.random.default_rng(seed + f).normal(size=E.shape)       # break exact ties at random
        c = afi_certificate(cell[te], E, y[te], K, alpha / 2, levels=levels, seed=seed + f)
        tot += te.mean() * c["dI_lower"]; parts.append(c["spectrum"])
    return tot, parts


def run(name, rep, n=8000):
    d = make(name, n, 10_000 * (MECH.index(name) + 1) + rep); y, P, K, cell = d["y"], d["P"], d["K"], d["cell"]
    Xs = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(d["notes"])
    E = TruncatedSVD(64, random_state=0).fit_transform(Xs); E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    c = afi_certificate(cell, E, y, K, .025, levels=3, seed=rep)
    pc = pd.DataFrame(P).groupby(cell).transform("mean").to_numpy()
    true_dI = float((1 - (pc ** 2).sum(1)).mean() - (1 - (P ** 2).sum(1)).mean())
    lv, _ = learned_dI(Xs, cell, y, K, .025, rep)
    return dict(dataset=name, rep=rep, true_dI=true_dI, svd_dI_L=c["dI_lower"], learned_dI_L=lv,
                portfolio_dI_L=max(c["dI_lower"], lv))


if __name__ == "__main__":
    reps = int(sys.argv[1]); names = sys.argv[2].split(",") if len(sys.argv) > 2 else MECH
    for nm in names:
        for rep in range(reps):
            print(json.dumps(run(nm, rep)), flush=True)
