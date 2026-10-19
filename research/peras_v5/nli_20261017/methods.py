"""NLI estimators/tests and accepted baselines for 'does the text add information beyond the table?'"""
import sys, math, numpy as np
sys.path.insert(0, __import__("os").environ.get("PERAS_REPO", __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), "../../.."))))
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack, csr_matrix
from pythia.core.dynamic_geometry import diffusion_coords, _within_cell_stats
from horos.models.pythia_adapters import dynamic_certificate


# ---------------------------------------------------------------- NLI (ours)
def nli_certificate(cell, E, Xs, y, alpha=.05, seed=0):
    """Portfolio lower bound on NLI (Brier units), alpha/4 each: static SVD, diffusion t=4, predictable posterior,
    model-assisted ceiling."""
    a = alpha / 4
    st = dynamic_certificate(cell, E, y, 2, a, B=1, learner="static", seed=seed)["dI_lower"]
    D = diffusion_coords(E, ts=(4,), seed=seed)[4]
    di = dynamic_certificate(cell, D, y, 2, a, B=1, learner="static", seed=seed)["dI_lower"]
    po = dynamic_certificate(cell, E, y, 2, a, B=3, learner="posterior", F=Xs, seed=seed, fracs=(.2, .4, .4))["dI_lower"]
    mc, _ = nli_model_ceiling(cell, Xs, y, a, seed)
    return max(st, di, po, mc), dict(static=st, diffusion=di, posterior=po, model_ceiling=mc), D


def _cell_pairs(cell, G, seed):
    from peras.core.anchored_information import _greedy_pairs
    rng = np.random.default_rng(seed); A, B = [], []
    for c in np.unique(cell):
        I = np.flatnonzero(cell == c)
        if len(I) < 4:
            continue
        I = I[rng.permutation(len(I))]; GI = G[I] + 1e-9 * rng.normal(size=G[I].shape)
        for a, b in _greedy_pairs(GI):
            A.append(I[a]); B.append(I[b])
    return np.array(A), np.array(B)


def nli_permutation_test(cell, geoms, y, B=199, seed=0):
    """Exact within-cell permutation test of H0: NLI = 0 with the pair statistic Q_cell - Q_text (Bonferroni over
    geometries).  Q_cell is invariant under within-cell permutation, so the statistic is -Q_text; pairs are fixed
    (outcome-blind), only labels move."""
    rng = np.random.default_rng(seed); groups = [np.flatnonzero(cell == c) for c in np.unique(cell)]
    pairs = [_cell_pairs(cell, G, seed) for G in geoms]
    s0 = [-(y[a] != y[b]).mean() for a, b in pairs]; ge = np.ones(len(geoms))
    for _ in range(B):
        yp = y.copy()
        for I in groups:
            yp[I] = y[I][rng.permutation(len(I))]
        ge += np.array([-(yp[a] != yp[b]).mean() >= s for (a, b), s in zip(pairs, s0)])
    return float(min(1.0, len(geoms) * (ge / (B + 1)).min()))


def nli_model_ceiling(cell, Xs, y, alpha=.05, seed=0, frac=.5):
    """Model-assisted NLI lower bound (assumption-free):  NLI = G_X - G_XT,  G_X >= table-cell U-statistic lower bound
    (exact replicates), G_XT <= held-out Brier of ANY predictor trained on other rows.  Union bound alpha/2 + alpha/2."""
    from peras.core.multiscale_pair_extrapolation import kl_bernoulli_upper, betting_upper
    n = len(y); rng = np.random.default_rng(seed); A = rng.random(n) < frac; Bm = ~A
    oh = csr_matrix(np.eye(cell.max() + 1)[cell]); F = hstack([oh, Xs]).tocsr()
    q = LogisticRegression(C=2, max_iter=3000).fit(F[A], y[A]).predict_proba(F[Bm])[:, 1]
    z = np.clip((q - y[Bm]) ** 2, 0, 1)                  # Brier/2 in [0,1]
    BU = 2 * min(kl_bernoulli_upper(float(z.mean()), len(z), alpha / 4), betting_upper(z, alpha / 4))
    cb, yb = cell[Bm], y[Bm]; w2, Q, nb = 0.0, 0.0, 0
    groups = [np.flatnonzero(cb == c) for c in np.unique(cb)]; groups = [g for g in groups if len(g) >= 4]
    nb = sum(len(g) for g in groups)
    for I in groups:
        m = len(I); k = yb[I].sum(); w = m / nb
        Q += w * (1 - (k * (k - 1) + (m - k) * (m - k - 1)) / (m * (m - 1))); w2 += w ** 2 / (m // 2)
    mstar = min(len(g) for g in groups)
    GXL = Q - math.sqrt(w2 * math.log(2 / alpha) / 2) - .5 / (mstar - 1)
    # rows of B outside the used cells: their model Brier is still bounded by BU only jointly; restrict to used cells
    used = np.concatenate(groups); z2 = np.clip((q[used] - yb[used]) ** 2, 0, 1)
    BU2 = 2 * min(kl_bernoulli_upper(float(z2.mean()), len(z2), alpha / 4), betting_upper(z2, alpha / 4))
    return max(GXL - BU2, 0.0), dict(GX_lower=GXL, model_brier_upper=BU2)


# ---------------------------------------------------------------- accepted baselines
def _brier(q, y):
    return 2 * (q - y) ** 2           # binary multiclass Brier (sum over two classes)


def model_delta_bootstrap(cell, Xs, y, B=1000, seed=0):
    """Common practice: cross-fitted logistic regression on table vs table+text; per-row Brier differences;
    percentile bootstrap CI of the mean improvement.  Returns (estimate, lower 2.5%, reject)."""
    n = len(y); folds = np.random.default_rng(seed).permutation(n) % 2
    oh = csr_matrix(np.eye(cell.max() + 1)[cell]); F = hstack([oh, Xs]).tocsr()
    qt, qa = np.zeros(n), np.zeros(n)
    for f in (0, 1):
        tr, te = folds != f, folds == f
        qt[te] = LogisticRegression(C=2, max_iter=3000).fit(oh[tr], y[tr]).predict_proba(oh[te])[:, 1]
        qa[te] = LogisticRegression(C=2, max_iter=3000).fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
    d = _brier(qt, y) - _brier(qa, y); rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    lo = float(np.quantile(bs, .025)); return float(d.mean()), lo, lo > 0


def model_local_permutation(cell, Xs, y, B=199, seed=0):
    """Accepted valid test (local permutation for discrete conditioning, Kim et al. 2022 style): fit table and
    table+text models on half A; on half B the statistic is the Brier improvement; permute labels within cells on
    half B (table-model Brier is invariant) -> exact p-value."""
    n = len(y); rng = np.random.default_rng(seed); A = rng.permutation(n) % 2 == 0; Bm = ~A
    oh = csr_matrix(np.eye(cell.max() + 1)[cell]); F = hstack([oh, Xs]).tocsr()
    mt = LogisticRegression(C=2, max_iter=3000).fit(oh[A], y[A]); ma = LogisticRegression(C=2, max_iter=3000).fit(F[A], y[A])
    qt = mt.predict_proba(oh[Bm])[:, 1]; qa = ma.predict_proba(F[Bm])[:, 1]; yb = y[Bm]; cb = cell[Bm]
    stat = lambda yy: float((_brier(qt, yy) - _brier(qa, yy)).mean())
    s0 = stat(yb); groups = [np.flatnonzero(cb == c) for c in np.unique(cb)]; ge = 1
    for b in range(B):
        yp = yb.copy()
        for I in groups:
            yp[I] = yb[I][rng.permutation(len(I))]
        ge += stat(yp) >= s0
    return ge / (B + 1), s0


def codec_within_cell(cell, Z, y, rng):
    """Azadkia-Chatterjee conditional dependence coefficient T(Y, Z | X) for discrete X: N(i) = random other row of
    the same cell (NN in X with random tie-break), M(i) = nearest row in Z within the same cell."""
    n = len(y); R = np.array([(y <= v).sum() for v in y], float)
    Ni = np.zeros(n, int); Mi = np.zeros(n, int)
    for c in np.unique(cell):
        I = np.flatnonzero(cell == c)
        if len(I) < 2:
            Ni[I] = I; Mi[I] = I; continue
        sh = rng.permutation(len(I)); Ni[I[sh]] = I[np.roll(sh, 1)]
        idx = NearestNeighbors(n_neighbors=2).fit(Z[I] + 1e-9 * rng.normal(size=Z[I].shape)).kneighbors(Z[I], return_distance=False)
        Mi[I] = I[idx[:, 1]]
    num = (np.minimum(R, R[Mi]) - np.minimum(R, R[Ni])).sum(); den = (R - np.minimum(R, R[Ni])).sum()
    return float(num / den) if den > 0 else 0.0


def codec_permutation_test(cell, Z, y, B=199, seed=0):
    """Within-cell permutation test with the CODEC statistic; the neighbour structure (N, M) is outcome-blind and fixed."""
    rng = np.random.default_rng(seed); n = len(y)
    Ni = np.zeros(n, int); Mi = np.zeros(n, int)
    for c in np.unique(cell):
        I = np.flatnonzero(cell == c)
        if len(I) < 2:
            Ni[I] = I; Mi[I] = I; continue
        sh = rng.permutation(len(I)); Ni[I[sh]] = I[np.roll(sh, 1)]
        idx = NearestNeighbors(n_neighbors=2).fit(Z[I] + 1e-9 * rng.normal(size=Z[I].shape)).kneighbors(Z[I], return_distance=False)
        Mi[I] = I[idx[:, 1]]
    def T(yy):
        R = np.searchsorted(np.sort(yy), yy, side="right").astype(float)
        num = (np.minimum(R, R[Mi]) - np.minimum(R, R[Ni])).sum(); den = (R - np.minimum(R, R[Ni])).sum()
        return float(num / den) if den > 0 else 0.0
    s0 = T(y); ge = 1; groups = [np.flatnonzero(cell == c) for c in np.unique(cell)]
    for _ in range(B):
        yp = y.copy()
        for I in groups:
            yp[I] = y[I][rng.permutation(len(I))]
        ge += T(yp) >= s0
    return ge / (B + 1), s0


def nli_perm_learned(cell, Xs, y, B=199, seed=0):
    """NLI pair statistic with a LEARNED text metric: on half A fit a table+text model; on half B pair rows inside each
    cell by closeness of the model score (outcome-blind for half B); exact within-cell permutation on half B."""
    n = len(y); rng = np.random.default_rng(seed); A = rng.permutation(n) % 2 == 0; Bm = ~A
    oh = csr_matrix(np.eye(cell.max() + 1)[cell]); F = hstack([oh, Xs]).tocsr()
    q = LogisticRegression(C=2, max_iter=3000).fit(F[A], y[A]).predict_proba(F[Bm])[:, 1:2]
    return nli_permutation_test(cell[Bm], [q], y[Bm], B, seed)


# ---------------------------------------------------------------- NLI projection (variational) certificate and test
def _loo_centred(cell, q):
    """h_i = q_i - mean_{j != i, same cell} q_j  (label-free).  Sums to exactly 0 inside every cell."""
    h = np.zeros(len(q))
    for c in np.unique(cell):
        I = np.flatnonzero(cell == c); m = len(I)
        if m < 2:
            continue
        S = q[I].sum(); h[I] = q[I] - (S - q[I]) / (m - 1)
    return h


def _fit_score(cell, Xs, y, tr, te):
    oh = csr_matrix(np.eye(cell.max() + 1)[cell]); F = hstack([oh, Xs]).tocsr()
    return LogisticRegression(C=2, max_iter=3000).fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]


def nli_projection(cell, Xs, y, alpha=.05, seed=0, B=199, p_true=None):
    """Variational NLI (Theorem V):  NLI_n >= 2 * rho_n^2,  rho_n = (1/n) sum (p_i - pbar_c) h_i / sqrt(H),
    H = (1/n) sum h_i^2, for ANY label-free-on-the-evaluation-half direction h with zero within-cell sums.
    Certificate: conditional on (X, T) the y_i are independent, so Hoeffding with ranges |h_i| bounds the numerator;
    Chernoff-KL lifts the in-sample bound to the population NLI.  Two folds, union bound alpha/4 per event.
    Test: exact within-cell permutation of the same linear statistic on fold B (single split, like the baseline),
    and a cross-fitted Bonferroni version.  Also returns the weighted-GCM normal-approximation p-value."""
    from peras.core.replicate_pair_certificates import kl_bernoulli_lower
    from scipy.stats import norm
    n = len(y); rng = np.random.default_rng(seed); fold = rng.permutation(n) % 2
    out = dict(cert=0.0, rho=[], perm_p=[], wgcm_p=[], cohort=0.0, true_cohort=None)
    for f in (0, 1):
        tr, te = fold != f, fold == f
        q = _fit_score(cell, Xs, y, tr, te); ct, yt = cell[te], y[te]; h = _loo_centred(ct, q); m = len(yt)
        H = float((h ** 2).mean())
        if H <= 0:
            out["rho"].append(0.0); out["perm_p"].append(1.0); out["wgcm_p"].append(1.0); continue
        C = float((yt * h).mean()); dev = math.sqrt(H * math.log(4 / alpha) / (2 * m))
        rho = C / math.sqrt(H); out["rho"].append(rho)
        Zn = max(C - dev, 0.0) ** 2 / H                                        # lower bound on (1/m) sum (p - pbar)^2
        out["cert"] = max(out["cert"], 2 * kl_bernoulli_lower(Zn, m, alpha / 4))
        if f == 0:                     # cohort target: in-sample NLI of the fold-0 evaluation rows, level alpha
            d1 = math.sqrt(H * math.log(1 / alpha) / (2 * m)); out["cohort"] = 2 * max(C - d1, 0.0) ** 2 / H
            if p_true is not None:
                pb = np.zeros(n)
                for c in np.unique(cell):
                    pb[cell == c] = p_true[cell == c].mean()
                out["true_cohort"] = float(2 * ((p_true[te] - pb[te]) ** 2).mean())
        groups = [np.flatnonzero(ct == c) for c in np.unique(ct)]; ge = 1
        for _ in range(B):
            yp = yt.copy()
            for I in groups:
                yp[I] = yt[I][rng.permutation(len(I))]
            ge += (yp * h).mean() >= C - 1e-15
        out["perm_p"].append(ge / (B + 1))
        r = np.zeros(m)                                                          # WGCM: residual products
        for I in groups:
            r[I] = (yt[I] - yt[I].mean()) * (h[I] - h[I].mean())
        out["wgcm_p"].append(float(1 - norm.cdf(math.sqrt(m) * r.mean() / max(r.std(ddof=1), 1e-12))))
    return dict(nli_proj_cert=out["cert"], nli_proj_cohort=out["cohort"], true_cohort=out["true_cohort"], nli_proj_rho=float(max(out["rho"])),
                nli_proj_perm_p=out["perm_p"][0], nli_proj_cf_p=float(min(1.0, 2 * min(out["perm_p"]))),
                wgcm_p=out["wgcm_p"][0])
