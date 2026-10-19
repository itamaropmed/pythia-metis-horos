"""Representation, cross-fitting, accepted baselines and the new estimator."""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.special import digamma, gammaln
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

import core
from core import LOSSES, erm_risk, loo_risk, pointloss

Z95 = 1.6448536269514722


# ------------------------------------------------------------------ representation
def represent(d, n_svd=16, seed=0):
    """Tabular (standardised) + TF-IDF(1-2 gram) -> SVD.  Unsupervised: never sees y.
    Returns (R, TF, variants) where variants are alternative unsupervised
    representations offered to feature-space baselines (best one reported)."""
    tab = d["tab"]
    tab = (tab - tab.mean(0)) / (tab.std(0) + 1e-9)
    TF = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True).fit_transform(d["notes"])
    sv = TruncatedSVD(n_svd, random_state=seed).fit_transform(TF)
    sv = (sv - sv.mean(0)) / (sv.std(0) + 1e-9)
    R = np.column_stack([tab, sv])
    variants = {"tab": tab, "tab+svd4": np.column_stack([tab, sv[:, :4]]), "tab+svd16": R}
    return R, TF, variants


# ------------------------------------------------------------------ cross-fitting
MODELS = {"mean": dict(loss="squared_error"), "q50": dict(loss="quantile", quantile=0.5),
          "q10": dict(loss="quantile", quantile=0.1), "q90": dict(loss="quantile", quantile=0.9)}


def _hgb(kw, seed):
    return HGB(max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
               min_samples_leaf=20, random_state=seed, **kw)


def text_stack(TF, y, tr, te, seed=0):
    """Nested cross-fitted ridge text scores for y and for |y - ridge(y)|.
    Returns (train_rows_scores, test_rows_scores); train-row scores are inner-OOF."""
    from sklearn.linear_model import Ridge
    def fit_pred(A, b, B):
        return Ridge(alpha=3.0).fit(A, b).predict(B)
    Str = np.zeros((len(tr), 2)); Ste = np.zeros((len(te), 2))
    for itr, ite in KFold(5, shuffle=True, random_state=seed).split(tr):
        a, b = tr[itr], tr[ite]
        Str[ite, 0] = fit_pred(TF[a], y[a], TF[b])
        inner = np.zeros(len(a))
        for i2, j2 in KFold(3, shuffle=True, random_state=seed).split(a):
            inner[j2] = fit_pred(TF[a[i2]], y[a[i2]], TF[a[j2]])
        Str[ite, 1] = fit_pred(TF[a], np.abs(y[a] - inner), TF[b])
    Ste[:, 0] = fit_pred(TF[tr], y[tr], TF[te])
    Ste[:, 1] = fit_pred(TF[tr], np.abs(y[tr] - Str[:, 0]), TF[te])
    return Str, Ste


def crossfit(R, y, K=5, seed=0, TF=None, frac=1.0):
    """K-fold cross-fitted loss-matched GBMs.  If TF (sparse TF-IDF) is given,
    nested cross-fitted ridge text scores are appended (fitted on training rows only).
    frac < 1 subsamples each training fold (for learning curves)."""
    out = {m: np.empty(len(y)) for m in MODELS}
    tr_risk = {m: [] for m in MODELS}
    rng = np.random.default_rng(seed)
    for tr, te in KFold(K, shuffle=True, random_state=seed).split(R):
        if frac < 1.0:
            tr = np.sort(rng.choice(tr, int(frac * len(tr)), replace=False))
        Rtr, Rte = R[tr], R[te]
        if TF is not None:
            Str, Ste = text_stack(TF, y, tr, te, seed)
            Rtr, Rte = np.column_stack([Rtr, Str]), np.column_stack([Rte, Ste])
        for m, kw in MODELS.items():
            g = _hgb(kw, seed).fit(Rtr, y[tr])
            out[m][te] = g.predict(Rte)
            tr_risk[m].append((tr, g.predict(Rtr)))
    out["_train"] = tr_risk
    return out


ACTION = {"MSE": "mean", "MAE": "q50", "Q90": "q90", "W30": "q50", "W60": "q50"}


# ------------------------------------------------------------------ baselines
def _curve_point(y, P):
    te = {L: float(pointloss(L, y - P[ACTION[L]]).mean()) for L in LOSSES}
    trr = {}
    for L in LOSSES:
        m = ACTION[L]
        trr[L] = float(np.mean([pointloss(L, y[tr] - p).mean() for tr, p in P["_train"][m]]))
    return te, trr


def cortes_curve(R, y, TF, seed, P_full, fracs=(0.3, 0.6), K=5):
    """Cortes, Jackel, Solla, Vapnik & Denker (1994): fit test and training
    learning curves with a common asymptote, E_test = a + b n^-alpha,
    E_train = a - c n^-alpha; the asymptote a estimates the irreducible error."""
    pts = []
    cortes_curve.last_P = {1.0: P_full}
    for f in fracs:
        P = crossfit(R, y, K=K, seed=seed, TF=TF, frac=f)
        cortes_curve.last_P[f] = P
        pts.append((f, *_curve_point(y, P)))
    pts.append((1.0, *_curve_point(y, P_full)))
    n0 = len(y) * (K - 1) / K
    out = {}
    for L in LOSSES:
        best = None
        for al in (0.5, 0.75, 1.0):
            A, b = [], []
            for f, te, trr in pts:
                x = (f * n0) ** -al
                A.append([1, x, 0]); b.append(te[L])
                A.append([1, 0, -x]); b.append(trr[L])
            sol, res, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
            r = float(np.sum((np.array(A) @ sol - b) ** 2))
            if best is None or r < best[0]:
                best = (r, sol[0])
        out[L] = float(max(best[1], 0.0))
    return out

def oof_risk(y, P):
    """Cross-fitted plug-in risk of a loss-matched GBM (Williamson et al. style)."""
    return {L: float(pointloss(L, y - P[ACTION[L]]).mean()) for L in LOSSES}


def knn_loo(R, y, k=20):
    """Classical leave-one-out kNN plug-in (local ERM action from k neighbours)."""
    nb = NearestNeighbors(n_neighbors=k).fit(R).kneighbors(return_distance=False)
    return {L: float(loo_risk(L, y, y[nb]).mean()) for L in LOSSES}


def rice_1nn(R, y):
    """1-NN residual-variance estimator (Devroye et al. 2018 / Rice-type)."""
    nb = NearestNeighbors(n_neighbors=1).fit(R).kneighbors(return_distance=False)[:, 0]
    return {"MSE": float(0.5 * np.mean((y - y[nb]) ** 2))}


def gamma_test(R, y, p=10):
    """Gamma test (Stefansson, Koncar & Jones 1997): regress gamma(k) on delta(k)."""
    dist, nb = NearestNeighbors(n_neighbors=p).fit(R).kneighbors()
    g = [0.5 * np.mean((y - y[nb[:, j]]) ** 2) for j in range(p)]
    dl = [np.mean(dist[:, j] ** 2) for j in range(p)]
    lr = LinearRegression().fit(np.array(dl)[:, None], g)
    return {"MSE": float(max(lr.intercept_, 0.0))}


def ab_lp(R, y, k=10):
    """Anderson & Bjarnadottir (2024) omniscient-model LP for MAE.
    min sum|P_i - y_i|  s.t. |P_i - P_j| <= beta_max * sd(y) * D_ij on a kNN graph,
    beta_max = largest |coef| of a standardised linear regression."""
    n = len(y)
    ys = (y - y.mean()) / y.std()
    beta = np.abs(LinearRegression().fit(R, ys).coef_).max() * y.std()
    dist, nb = NearestNeighbors(n_neighbors=k).fit(R).kneighbors()
    I = np.repeat(np.arange(n), k); J = nb.ravel(); D = dist.ravel()
    keep = I < J
    I, J, D = I[keep], J[keep], D[keep]
    m = len(I)
    # variables: P (n), t (n)
    rows = np.arange(m)
    A1 = sparse.csr_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[rows, rows], np.r_[I, J])), shape=(m, 2 * n))
    eye = sparse.identity(n, format="csr")
    A2 = sparse.hstack([eye, -eye]); A3 = sparse.hstack([-eye, -eye])
    A = sparse.vstack([A1, -A1, A2, A3]).tocsr()
    b = np.r_[beta * D, beta * D, y, -y]
    c = np.r_[np.zeros(n), np.ones(n)]
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * n + [(0, None)] * n, method="highs")
    return {"MAE": float(res.fun / n) if res.status == 0 else np.nan}


def _kl_entropy(X, k=3):
    """Kozachenko-Leonenko differential entropy (nats)."""
    n, d = X.shape
    r = NearestNeighbors(n_neighbors=k).fit(X).kneighbors()[0][:, -1]
    logc = d / 2 * np.log(np.pi) - gammaln(d / 2 + 1)
    return digamma(n) - digamma(k) + logc + d * np.mean(np.log(r + 1e-12))


def fano_bounds(R, y, dim=6):
    """Estimation-theoretic Fano bounds from h(Y|X) = h(X,Y) - h(X):
    MSE >= exp(2h)/(2 pi e),  MAE >= exp(h)/(2e)."""
    Z = PCA(dim, random_state=0).fit_transform(R)
    Z = Z / Z.std(0)
    h = _kl_entropy(np.column_stack([Z, y])) - _kl_entropy(Z)
    return {"MSE": float(np.exp(2 * h) / (2 * np.pi * np.e)), "MAE": float(np.exp(h) / (2 * np.e))}


def tuples(G, e, k_max, metric_nb=None):
    """Neighbour tuples in geometry G (self excluded from neighbour list)."""
    dist, nb = NearestNeighbors(n_neighbors=k_max).fit(G).kneighbors()
    return dist, nb


def bracket_midpoint(R, y, k=5):
    """Previous Peras (Codex) expectation bracket on raw outcomes in feature space:
    midpoint of in-tuple ERM risk and leave-one-out risk."""
    _, nb = NearestNeighbors(n_neighbors=k).fit(R).kneighbors()
    out = {}
    for L in LOSSES:
        T = np.column_stack([y, y[nb[:, :k - 1]]])
        out[L] = float(0.5 * (erm_risk(L, T).mean() + loo_risk(L, y, y[nb[:, :k]]).mean()))
    return out


def paired_witness_lcb(R, y):
    """Previous Peras paired witness (k=2 in-tuple ERM on raw 1-NN pairs), 95% LCB."""
    nb = NearestNeighbors(n_neighbors=1).fit(R).kneighbors(return_distance=False)[:, 0]
    out = {}
    for L in LOSSES:
        w = erm_risk(L, np.column_stack([y, y[nb]]))
        out[L] = float(w.mean() - Z95 * w.std(ddof=1) / np.sqrt(len(w)))
    return out


# ------------------------------------------------------------------ new: RTX
def law_embedding(P):
    """Cross-fitted conditional-law coordinates: quantile vector (q10,q50,q90) and mean.
    RMS distance between these vectors approximates the W2 distance between the
    conditional laws of two cases (Theorem 4)."""
    Q = np.column_stack([P["q10"], P["q50"], P["q90"], P["mean"]])
    return Q / Q[:, 1].std()


def rtx(R, y, P, ks=(2, 3, 4, 6, 8, 12, 16), geometry="law", mode="extrap"):
    """Residualised law-space Tuple eXtrapolation.
    For each loss: residualise with the loss-matched cross-fitted action, form
    tuples in law space, compute the in-tuple ERM risk r(k) and LOO risk r~(k),
    midpoint m(k) = (r+r~)/2 cancels the 1/k optimism, then extrapolate m against
    the tuple's law-space spread delta(k) to delta -> 0 (Gamma-test style)."""
    G = law_embedding(P) if geometry == "law" else R
    kmax = max(ks)
    dist, nb = NearestNeighbors(n_neighbors=kmax).fit(G).kneighbors()
    out, diag = {}, {}
    for L in LOSSES:
        e = y - P[ACTION[L]]
        ms, ds, rs, rts = [], [], [], []
        for k in ks:
            T = np.column_stack([e, e[nb[:, :k - 1]]])
            r = erm_risk(L, T).mean()
            rt = loo_risk(L, e, e[nb[:, :k]]).mean()
            ms.append(0.5 * (r + rt)); rs.append(r); rts.append(rt)
            ds.append(np.mean(dist[:, :k] ** 2))
        ms, ds = np.array(ms), np.array(ds)
        if mode == "extrap":
            lr = LinearRegression().fit(ds[:, None], ms)
            est = lr.intercept_
            est = float(np.clip(est, min(rs), max(rts)))
        else:
            est = float(ms[ks.index(8)])
        out[L] = est
        diag[L] = dict(m=ms.tolist(), d=ds.tolist(), r=rs, rt=rts)
    return out, diag


def _window_R_emp(zs, c):
    """1 - max_a fraction of sorted sample zs inside [a-c, a+c]  (exact)."""
    j = np.searchsorted(zs, zs + 2 * c, side="right")
    return 1.0 - (j - np.arange(len(zs))).max() / len(zs)


def calibrate_window_weight(z, k, cgrid, B=3000, seed=0):
    """Self-calibrated combination weight w(c) for window losses.
    Resample tuples from the pooled standardised residual law, compute the
    in-tuple ERM risk r_k(c), the LOO risk r~_k(c) and the exact risk R(c) of that
    law; w(c) solves w r + (1-w) r~ = R.  Uses no gold information."""
    rng = np.random.default_rng(seed)
    zs = np.sort(z)
    T = z[rng.integers(0, len(z), (B, k + 1))]
    w = np.empty(len(cgrid))
    for i, c in enumerate(cgrid):
        L = f"W{float(c)!r}"
        r = core.erm_risk(L, T[:, :k]).mean()
        rt = core.loo_risk(L, T[:, k], T[:, :k]).mean()
        Rc = _window_R_emp(zs, c)
        w[i] = np.clip((rt - Rc) / (rt - r), 0.0, 1.0) if rt - r > 1e-9 else 0.5
    return w


def rtx2(R, y, P, k=8, ks_extrap=None):
    """RTX v2 (frozen before confirmation):
    * residualise with the loss-matched cross-fitted action,
    * tuples of size k in cross-fitted law space,
    * regular losses (MSE, MAE, pinball): midpoint of in-tuple ERM and LOO risk
      (Theorem 3: equal first-order optimism and pessimism),
    * window losses (non-regular, cube-root): per-case combination weight
      w(h / s_i) self-calibrated on the pooled standardised residual law."""
    G = law_embedding(P)
    _, nb = NearestNeighbors(n_neighbors=k).fit(G).kneighbors()
    s = np.maximum((P["q90"] - P["q10"]) / 2.5631, 1e-6)
    out = {}
    for L in LOSSES:
        e = y - P[ACTION[L]]
        T = np.column_stack([e, e[nb[:, :k - 1]]])
        r = erm_risk(L, T)
        rt = loo_risk(L, e, e[nb[:, :k]])
        if L.startswith("W"):
            h = float(L[1:])
            z = e / s
            cgrid = np.geomspace(0.05, 40, 40)
            wg = calibrate_window_weight(z, k, cgrid)
            wi = np.interp(np.log(h / s), np.log(cgrid), wg)
            out[L] = float(np.mean(wi * r + (1 - wi) * rt))
        else:
            out[L] = float(np.mean(0.5 * (r + rt)))
    return out


def rtx3(R, y, P, ks=(3, 4, 6, 8, 12, 16), window_weights=True, extrap=True):
    """RTX v3: loss-aware combination inside spread extrapolation.
    For every tuple size k: c(k) = w r(k) + (1-w) r~(k) with w = 1/2 for regular
    losses and the self-calibrated w_k(h/s_i) for window losses; then c(k) is
    regressed on the law-space spread delta(k) and extrapolated to delta = 0."""
    G = law_embedding(P)
    kmax = max(ks)
    dist, nb = NearestNeighbors(n_neighbors=kmax).fit(G).kneighbors()
    s = np.maximum((P["q90"] - P["q10"]) / 2.5631, 1e-6)
    cgrid = np.geomspace(0.05, 40, 40)
    out = {}
    for L in LOSSES:
        e = y - P[ACTION[L]]
        cs, ds, rs, rts = [], [], [], []
        for k in ks:
            r = erm_risk(L, np.column_stack([e, e[nb[:, :k - 1]]]))
            rt = loo_risk(L, e, e[nb[:, :k]])
            if L.startswith("W") and window_weights:
                wg = calibrate_window_weight(e / s, k, cgrid, seed=k)
                wi = np.interp(np.log(float(L[1:]) / s), np.log(cgrid), wg)
            else:
                wi = 0.5
            cs.append(float(np.mean(wi * r + (1 - wi) * rt)))
            rs.append(r.mean()); rts.append(rt.mean())
            ds.append(float(np.mean(dist[:, :k] ** 2)))
        if extrap:
            est = LinearRegression().fit(np.array(ds)[:, None], cs).intercept_
        else:
            est = cs[ks.index(8)]
        out[L] = float(np.clip(est, min(rs), max(rts)))
    return out


# ------------------------------------------------------------------ FROZEN (dev-selected, 80 draws)
LCB_K = {"MSE": 3, "MAE": 3, "Q90": 6, "W30": 6, "W60": 3}


def rtx_final(y, P):
    """Frozen RTX estimator.  Configuration chosen on dev seeds 100-107 only."""
    a = rtx3(None, y, P)                       # MSE, MAE, Q90
    b, _ = rtx(None, y, P)                     # W30 (midpoint + spread extrapolation)
    c = rtx3(None, y, P, extrap=False)         # W60 (self-calibrated weights, k = 8)
    return {"MSE": a["MSE"], "MAE": a["MAE"], "Q90": a["Q90"], "W30": b["W30"], "W60": c["W60"]}


def rtx_lcb_final(y, P):
    """Frozen RTX lower confidence bound: in-tuple ERM of residuals in law space,
    tuple size per loss from LCB_K (largest k whose next size was also valid on dev)."""
    G = law_embedding(P)
    _, nb = NearestNeighbors(n_neighbors=max(LCB_K.values())).fit(G).kneighbors()
    out = {}
    for L in LOSSES:
        k = LCB_K[L]
        e = y - P[ACTION[L]]
        w = erm_risk(L, np.column_stack([e, e[nb[:, :k - 1]]]))
        out[L] = float(w.mean() - Z95 * w.std(ddof=1) / np.sqrt(len(w)))
    return out


def rtx_lcb(R, y, P, k=3):
    """New lower bound: in-tuple ERM risk of residuals in law space (k-tuple),
    one-sided 95% CLT lower confidence bound."""
    G = law_embedding(P)
    _, nb = NearestNeighbors(n_neighbors=k).fit(G).kneighbors()
    out = {}
    for L in LOSSES:
        e = y - P[ACTION[L]]
        w = erm_risk(L, np.column_stack([e, e[nb[:, :k - 1]]]))
        out[L] = float(w.mean() - Z95 * w.std(ddof=1) / np.sqrt(len(w)))
    return out


# ------------------------------------------------------------------ FROZEN v2 (dev-selected on seeds 100-107)
def rtx_lc(y, Ps, N0=1600.0):
    """RTX-LC: double extrapolation.  RTX is computed with cross-fitted nuisances trained
    on fractions f of each training fold and extrapolated along the learning curve,
    RTX(f) = a + b / (f N0), to f -> infinity (model error -> 0; Theorem 4).
    Q90 keeps plain RTX (learning-curve extrapolation hurt it on dev)."""
    fr = sorted(Ps)
    vals = {f: rtx_final(y, Ps[f]) for f in fr}
    out = {}
    for L in LOSSES:
        if L == "Q90":
            out[L] = vals[1.0][L]; continue
        A = np.array([[1.0, 1.0 / (f * N0)] for f in fr]); b = np.array([vals[f][L] for f in fr])
        out[L] = float(max(np.linalg.lstsq(A, b, rcond=None)[0][0], 0.0))
    return out
