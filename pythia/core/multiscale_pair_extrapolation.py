"""Modulus-free Bayes-floor certificates by certified extrapolation across nested perfect matchings.

Identity (Brier / Gini geometry).  For independent rows with label laws p_i (K classes) and any perfect matching M of
the n rows (pairs disjoint, chosen outcome-blind),
    E Q(M) = Gbar + D(M),   Q(M) = mean_{(i,j) in M} 1{Y_i != Y_j},
    Gbar = (1/n) sum_i g_i,  g_i = 1 - |p_i|^2,   D(M) = mean_{(i,j) in M} |p_i - p_j|^2 / 2  >= 0,
because 1 - <p_i,p_j> = (g_i+g_j)/2 + |p_i-p_j|^2/2 and a perfect matching uses every row once, so the floor term
Gbar is IDENTICAL for every perfect matching.  (Squared loss: same with 1/2 (Y_i-Y_j)^2, Gbar -> mean sigma^2,
D -> mean (mu_i-mu_j)^2/2.)  Then R_01 >= f_K(Gbar) (Cover-Hart inversion + Jensen).

Shape assumption A(gamma) (replaces any Lipschitz constant):  for two matchings a, b with mean squared pair
distances s_a < s_b,   D(a) <= (s_a/s_b)^gamma * D(b).
  gamma = 1   : drift grows at least linearly in squared distance (smooth laws, D ~ c d^2 locally),
  gamma = 1/2 : drift grows at least linearly in distance (laws with jumps / class boundaries),
  gamma -> 0  : no assumption (bound -> 0).
Theorem.  Under A(gamma), with t = (s_a/s_b)^gamma < 1,
    Gbar >= (E Q(a) - t E Q(b)) / (1 - t),
and with Chernoff-KL bounds (pairs inside a matching are independent given X):
    Gbar >= (Q(a)_L - t Q(b)_U) / (1 - t)   with probability >= 1 - alpha_a - alpha_b.
No modulus, no density, no replicates, any metric.  Nested matchings are built by recursive unit matching
(dyadic multiscale), so all levels are perfect matchings of the same rows.
Falsification: A(gamma) implies E Q nondecreasing across levels; an observed significant decrease refutes it.

Prior art (estimation, not certification): Gamma test (Stefansson, Koncar & Jones 1997) extrapolates
1/2 (Y_i-Y_j)^2 vs |x_i-x_j|^2 to zero distance for noise variance; Okuno & Shimodaira (2020) extrapolate k-NN
estimators to k = 0.  New here: one-sided certified extrapolation under an explicit, dial-able shape assumption,
the exact equal-floor property of perfect matchings, the 0-1 (Brier-drift) version with Cover-Hart inversion, and
finite-sample validity.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from .geometry import knn_query

from .replicate_pair_certificates import f_K, kl_bernoulli_lower


def kl_bernoulli_upper(xbar, n, alpha):
    if xbar >= 1:
        return 1.0
    L = math.log(1 / alpha) / n
    kl = lambda x, q: (x * math.log(x / q) if x > 0 else 0) + (1 - x) * math.log((1 - x) / (1 - q))
    lo, hi = xbar, 1 - 1e-15
    if kl(xbar, hi) <= L:
        return 1.0
    for _ in range(100):
        mid = .5 * (lo + hi)
        if kl(xbar, mid) > L:
            hi = mid
        else:
            lo = mid
    return hi


def _greedy_perfect(C, k=15):
    """Greedy perfect matching of the rows of C (points) by increasing Euclidean distance."""
    m = len(C)
    if m == 2:
        return [(0, 1)]
    kk = min(k, m - 1)
    dist, idx = knn_query(C, k=kk + 1)
    # NOTE: with tied points the neighbour list can contain i itself at any rank -> exclude self explicitly
    E = [(dist[i, r], i, idx[i, r]) for i in range(m) for r in range(kk + 1) if idx[i, r] != i]
    E.sort()
    free = np.ones(m, bool); P = []
    for _, i, j in E:
        if free[i] and free[j]:
            free[i] = free[j] = False; P.append((i, j))
    rest = np.flatnonzero(free)
    while len(rest) > 1:
        Dr = ((C[rest][:, None, :] - C[rest][None, :, :]) ** 2).sum(-1); np.fill_diagonal(Dr, np.inf)
        a, b = np.unravel_index(np.argmin(Dr), Dr.shape)
        P.append((rest[a], rest[b])); rest = np.delete(rest, [a, b])
    flat = np.array(P).ravel()
    assert len(set(flat.tolist())) == len(flat) == m, "matching is not perfect (self-pair or reuse)"
    return P


def nested_matchings(X, levels=4, perm_seed=None):
    """Dyadic nested perfect matchings.  n must be divisible by 2**levels.  Returns list of (pairs array, s).
    perm_seed: apply a random row permutation before matching so that tie-breaking and search order cannot depend on
    the row order.  REQUIRED whenever the row order may carry outcome information (e.g. files sorted by label):
    the geometry must be outcome-blind *including* its tie-breaking."""
    if perm_seed is not None:
        X = np.asarray(X, float); pi = np.random.default_rng(perm_seed).permutation(len(X))
        return [(pi[P], s) for P, s in nested_matchings(X[pi], levels)]
    X = np.asarray(X, float); n = len(X); assert n % (2 ** levels) == 0
    units = [np.array([i]) for i in range(n)]; out = []
    for lev in range(levels):
        cent = np.array([X[u].mean(0) for u in units])
        UP = _greedy_perfect(cent)
        pairs, new_units = [], []
        for a, b in UP:
            A, B = units[a], units[b]
            if len(A) == 1:
                pairs.append((A[0], B[0]))
            else:
                Dm = ((X[A][:, None, :] - X[B][None, :, :]) ** 2).sum(-1)
                r, c = linear_sum_assignment(Dm); pairs.extend(zip(A[r], B[c]))
            new_units.append(np.concatenate([A, B]))
        P = np.array(pairs); s = float(((X[P[:, 0]] - X[P[:, 1]]) ** 2).sum(1).mean())
        out.append((P, s)); units = new_units
    return out


def extrapolation_certificate(mats, y, K, alpha=.05, gammas=(1.0, .5), loss="01", y_range=None):
    """Portfolio over level pairs (a<b) and declared gammas; Bonferroni over candidates and the two sides.
    Returns the best lower bound on the floor (0-1: R >= f_K(Gbar_L); squared: mean sigma^2 >= Gbar_L)."""
    y = np.asarray(y)
    Q, nP, S = [], [], []
    for P, s in mats:
        if loss == "01":
            z = (y[P[:, 0]] != y[P[:, 1]]).astype(float)
        else:
            z = .5 * (y[P[:, 0]] - y[P[:, 1]]) ** 2 / (.5 * y_range ** 2)     # scaled to [0,1]
        Q.append(float(z.mean())); nP.append(len(P)); S.append(s)
    cands = [(a, b, g) for a in range(len(mats)) for b in range(a + 1, len(mats)) for g in gammas]
    a_each = alpha / (2 * len(cands))
    best, info, rows = 0.0, None, []
    for a, b, g in cands:
        t = (S[a] / S[b]) ** g
        if not t < 1:
            continue
        GL = (kl_bernoulli_lower(Q[a], nP[a], a_each) - t * kl_bernoulli_upper(Q[b], nP[b], a_each)) / (1 - t)
        GL = max(GL, 0.0)
        val = float(f_K(GL, K)) if loss == "01" else GL * .5 * y_range ** 2
        rows.append(dict(a=a, b=b, gamma=g, t=t, bound=val))
        if val > best:
            best, info = val, dict(a=a, b=b, gamma=g, t=t)
    return dict(bound=best, best=info, Q=Q, s=S, candidates=rows)


def per_gamma_bounds(mats, y, K, alpha=.05, gammas=(1.0, .75, .5, .25), loss="01", y_range=None):
    """Separate portfolio (over level pairs only) for each gamma: the smoothness-order dial."""
    return {g: extrapolation_certificate(mats, y, K, alpha, (g,), loss, y_range)["bound"] for g in gammas}


def lipschitz_pair_bound(X, mats, y, K, L, alpha=.05):
    """Accepted near-replicate route (second-order version, stronger than first-order): declared TV modulus
    TV(p_i,p_j) <= L |x_i-x_j|, so D(M1) <= L^2 s_1 and Gbar >= Q(M1)_L - L^2 s_1."""
    P, s = mats[0]; y = np.asarray(y)
    z = (y[P[:, 0]] != y[P[:, 1]]).astype(float)
    GL = max(kl_bernoulli_lower(float(z.mean()), len(z), alpha) - L ** 2 * s, 0.0)
    return float(f_K(GL, K))


# ============================================================== v2: strengthening (2026-10-16)
# Variogram view.  D(M) is the empirical semivariogram of the label law p(x) at the lag profile of M; Gbar is the
# nugget.  The non-identifiability of "measurement error vs micro-scale variation" inside the nugget (geostatistics)
# is exactly the adversarial failure; replicate measurements at the same location are the classical remedy.
def _pair_terms(P, y, target, y_range=None):
    if target in ("01", "brier"):
        return (y[P[:, 0]] != y[P[:, 1]]).astype(float)
    return .5 * (y[P[:, 0]] - y[P[:, 1]]) ** 2 / (.5 * y_range ** 2)


def bernstein_lower(z, alpha):
    """Variance-adaptive lower bound for the mean of independent [0,1] variables (non-identical allowed):
    Maurer's non-negative lower tail  mu >= zbar - sqrt(2 V L / N)  with V an upper bound on mean E z^2
    (Chernoff-KL upper bound on z^2, which is also [0,1]).  alpha split in two."""
    N = len(z); L = math.log(2 / alpha)
    V = kl_bernoulli_upper(float((z ** 2).mean()), N, alpha / 2)
    return max(float(z.mean()) - math.sqrt(2 * V * L / N), 0.0)


def bernstein_upper(z, alpha):
    N = len(z); L = math.log(2 / alpha)
    V = kl_bernoulli_upper(float((z ** 2).mean()), N, alpha / 2)
    return min(float(z.mean()) + math.sqrt(2 * V * L / N) + L / (3 * N), 1.0)


def _lower(z, alpha, conc):
    if conc == "kl":
        return kl_bernoulli_lower(float(z.mean()), len(z), alpha)
    if conc == "bernstein":
        return bernstein_lower(z, alpha)
    if conc == "betting":
        return betting_lower(z, alpha)
    return max(kl_bernoulli_lower(float(z.mean()), len(z), alpha / 2), betting_lower(z, alpha / 2))


def _upper(z, alpha, conc):
    if conc == "kl":
        return kl_bernoulli_upper(float(z.mean()), len(z), alpha)
    if conc == "bernstein":
        return bernstein_upper(z, alpha)
    if conc == "betting":
        return betting_upper(z, alpha)
    return min(kl_bernoulli_upper(float(z.mean()), len(z), alpha / 2), betting_upper(z, alpha / 2))


def certificate_v2(mats, y, K=None, alpha=.05, gammas=(.5,), target="01", y_range=None, conc="best", pairs=None):
    """target: '01' (R_01 >= f_K(Gbar_L)), 'brier' (Brier/Gini floor Gbar >= Gbar_L, no Jensen), 'mse'
    (mean sigma^2 >= ...).  pairs: optional list of (a, b) level pairs (default all a < b)."""
    y = np.asarray(y)
    Z = [_pair_terms(P, y, target, y_range) for P, _ in mats]; S = [s for _, s in mats]
    pairs = pairs or [(a, b) for a in range(len(mats)) for b in range(a + 1, len(mats))]
    cands = [(a, b, g) for a, b in pairs for g in gammas]; ae = alpha / (2 * len(cands))
    best, info = 0.0, None
    for a, b, g in cands:
        t = (S[a] / S[b]) ** g
        if not t < 1 - 1e-9:          # level b is not coarser than level a: no extrapolation possible
            continue
        GL = max((_lower(Z[a], ae, conc) - t * _upper(Z[b], ae, conc)) / (1 - t), 0.0)
        if target == "01":
            val = float(f_K(GL, K))
        elif target == "brier":
            val = GL
        else:
            val = GL * .5 * y_range ** 2
        if val > best:
            best, info = val, dict(a=a, b=b, gamma=g, t=t)
    return dict(bound=best, best=info, Q=[float(z.mean()) for z in Z], s=S)


def expectation_extrapolation(mats, P, info):
    """Assumption-level check from the TRUE laws: (E Q_a - t E Q_b)/(1-t) - Gbar (> 0 means A(gamma) failed in
    expectation, independent of any concentration or Jensen slack)."""
    G = float((1 - (P ** 2).sum(1)).mean())
    EQ = [float((1 - (P[M[:, 0]] * P[M[:, 1]]).sum(1)).mean()) for M, _ in mats]
    a, b, t = info["a"], info["b"], info["t"]
    return (EQ[a] - t * EQ[b]) / (1 - t) - G


def increments_test(mats, y, gamma, alpha=.05, target="01", y_range=None):
    """Necessary condition of A(gamma) for every level triple a<b<c:  Delta_bc >= kappa Delta_ab,
    kappa = (s_c/s_b)^gamma - 1.  W = (Q_c - Q_b) - kappa (Q_b - Q_a) is linear in pair indicators; the three
    matchings are three colour classes of the pair-dependency graph, so Janson's Hoeffding bound with chi* = 3
    gives a one-sided test of E W >= 0.  Returns the smallest Bonferroni-adjusted p-value over triples."""
    y = np.asarray(y); Z = [_pair_terms(P, y, target, y_range) for P, _ in mats]; S = [s for _, s in mats]
    L = len(mats); triples = [(a, b, c) for a in range(L) for b in range(a + 1, L) for c in range(b + 1, L)]
    pmin, worst = 1.0, None
    for a, b, c in triples:
        if not (S[a] < S[b] < S[c]):
            continue
        kap = (S[c] / S[b]) ** gamma - 1; N = len(Z[a])
        W = (Z[c].mean() - Z[b].mean()) - kap * (Z[b].mean() - Z[a].mean())
        var = (1 + (1 + kap) ** 2 + kap ** 2) / N
        p = math.exp(-2 * W ** 2 / (3 * var)) if W < 0 else 1.0
        if p < pmin:
            pmin, worst = p, (a, b, c, float(W))
    return dict(p=min(1.0, pmin * len(triples)), worst=worst)


def replicate_anchor(Xs, y1, y2, K, gamma, alpha=.05, levels=4, n_main=None):
    """Test-retest substudy: m locations measured twice (independent labels given x).
    (i) Assumption-free floor of the substudy locations: E 1{y1 != y2} = g_i exactly, so Gbar_S >= KL lower bound
        (finite-population transport to n_main rows by Hoeffding-Serfling when the substudy is a random subset).
    (ii) Falsification of A(gamma): T = extrap(copy 1) - Q_rep has E T <= 0 under A(gamma); Janson-Hoeffding with
        three colour classes (matching a, matching b, replicate edges).  Uses levels (0, levels-1)."""
    y1 = np.asarray(y1); y2 = np.asarray(y2); m = len(y1)
    mats = nested_matchings(Xs, levels); a, b = 0, levels - 1
    za = _pair_terms(mats[a][0], y1, "01"); zb = _pair_terms(mats[b][0], y1, "01"); zr = (y1 != y2).astype(float)
    t = (mats[a][1] / mats[b][1]) ** gamma; N = len(za)
    T = (za.mean() - t * zb.mean()) / (1 - t) - zr.mean()
    var = (1 + t ** 2) / ((1 - t) ** 2 * N) + 1 / m
    p_reject = math.exp(-2 * T ** 2 / (3 * var)) if T > 0 else 1.0
    a_ = alpha / 2 if (n_main is not None and n_main > m) else alpha
    GL = kl_bernoulli_lower(float(zr.mean()), m, a_)
    if n_main is not None and n_main > m:
        rng_ = (K - 1) / K; GL = max(GL - rng_ * math.sqrt(math.log(1 / a_) * (1 - (m - 1) / n_main) / (2 * m)), 0.0)
    return dict(T=float(T), p_falsify=p_reject, substudy_Gbar_L=GL, substudy_bound=float(f_K(GL, K)),
                Q_rep=float(zr.mean()), extrap=float((za.mean() - t * zb.mean()) / (1 - t)))


def _bet_stat(z, m, lams, sign):
    # sign=+1: lower bound (H0 mean <= m); sign=-1: upper bound (H0 mean >= m)
    v = sign * (z - m)
    return max(float(np.log1p(l * v).sum()) for l in lams)


def betting_lower(z, alpha, grid=24):
    """Fixed-bet capital processes: for any FIXED lambda in [0, 1/m), E prod(1 + lambda (z_i - m)) =
    prod(1 + lambda (mu_i - m)) <= exp(lambda sum(mu_i - m)) <= 1 when the AVERAGE mean is <= m, so Markov gives a
    valid test for independent, non-identical [0,1] variables; a geometric lambda grid is Bonferroni-charged.
    Adapts to the realised distribution (variance, skew) without estimating a variance."""
    z = np.asarray(z, float); L = math.log(grid / alpha)
    def rejects(m):
        lams = (np.geomspace(1e-3, .999, grid) / max(m, 1e-12))
        return _bet_stat(z, m, lams, 1) >= L
    lo, hi = 0.0, float(z.mean())
    if hi <= 0 or not rejects(1e-9):
        return 0.0
    for _ in range(60):
        mid = .5 * (lo + hi)
        if rejects(mid):
            lo = mid
        else:
            hi = mid
    return lo


def betting_upper(z, alpha, grid=24):
    z = np.asarray(z, float); L = math.log(grid / alpha)
    def rejects(m):
        lams = (np.geomspace(1e-3, .999, grid) / max(1 - m, 1e-12))
        return _bet_stat(z, m, lams, -1) >= L
    lo, hi = float(z.mean()), 1.0
    if not rejects(1 - 1e-9):
        return 1.0
    for _ in range(60):
        mid = .5 * (lo + hi)
        if rejects(mid):
            hi = mid
        else:
            lo = mid
    return hi


def replicate_anchor_v2(X, y, sub, y2, K, gamma, alpha=.05, levels=4):
    """Test-retest substudy anchor (v2).  sub = indices of a RANDOM outcome-blind subset of the n rows that were
    measured a second time (labels y2, independent given x).  Rows outside sub (rest) carry the extrapolation;
    rows in sub carry the zero-lag (exact replicate) measurement, so the two parts are independent.
      E extrap(rest) <= Gbar_rest under A(gamma);   E Q_rep = Gbar_sub;   Gbar_rest - Gbar_sub is a random-split
      deviation of numbers in [0, (K-1)/K].
    T = extrap(rest) - Q_rep has E T <= 0 under A(gamma) (marginally over the split).  Hoeffding with a 2-colouring
    (matching a + replicate edges | matching b) plus the split term gives p_falsify.
    Also returns the assumption-free substudy certificate for the n-row floor (KL + Hoeffding-Serfling transport)."""
    y = np.asarray(y); y2 = np.asarray(y2); n = len(y); m = len(sub)
    rest = np.setdiff1d(np.arange(n), sub); rest = rest[: (len(rest) // 2 ** levels) * 2 ** levels]
    mats = nested_matchings(X[rest], levels); yr = y[rest]
    za = _pair_terms(mats[0][0], yr, "01"); zb = _pair_terms(mats[-1][0], yr, "01"); zr = (y[sub] != y2).astype(float)
    t = (mats[0][1] / mats[-1][1]) ** gamma; N = len(za); r = (K - 1) / K
    assert t < 1, "coarsest level must have larger lag than the finest" 
    extrap = (za.mean() - t * zb.mean()) / (1 - t); T = float(extrap - zr.mean())
    sc2 = (1 + t ** 2) / ((1 - t) ** 2 * N) + 1 / m
    p = math.exp(-2 * T ** 2 / (2 * sc2 + r ** 2 / m)) if T > 0 else 1.0
    a_ = alpha / 2
    GL = kl_bernoulli_lower(float(zr.mean()), m, a_)
    GL = max(GL - r * math.sqrt(math.log(1 / a_) * (1 - (m - 1) / n) / (2 * m)), 0.0)
    return dict(T=T, p_falsify=p, substudy_bound=float(f_K(GL, K)), Q_rep=float(zr.mean()), extrap=float(extrap))


def replicate_anchor_v3(X, y, sub, y2, K, gamma, alpha=.05, beta=.05, levels=4):
    """v3: sharper falsification + an assumption-free TOLERANCE.
    Sub-Gaussian proxies (Hoeffding units, exp(-2 eps^2 / proxy)):
      extrapolation part (two dependent matchings on rest): (sqrt(A_a) + sqrt(A_b))^2 with
        A_a = 1/((1-t)^2 N), A_b = t^2/((1-t)^2 N)    [Holder: proxies of dependent sums add in square root]
      replicate part (independent rows): 1/m
      random-split part Gbar_rest - Gbar_sub: r^2 (1-f)/m * (n/(n-m))^2   (Hoeffding-Serfling)
    Reject A(gamma) when T > eps_beta.  If NOT rejected, deterministically
      GL_rest <= extrap_rest <= Q_rep + eps_beta,
    hence, with Q_rep's own upper deviation and the split term (prob >= 1 - alpha),
      GL_rest <= Gbar_rest + tau,   tau = eps_beta + dev_rep + dev_split      (NO shape assumption),
    i.e. the pipeline's possible violation is capped by a computable tau."""
    y = np.asarray(y); y2 = np.asarray(y2); n = len(y); m = len(sub)
    rest = np.setdiff1d(np.arange(n), sub); rest = rest[: (len(rest) // 2 ** levels) * 2 ** levels]
    mats = nested_matchings(X[rest], levels); yr = y[rest]
    za = _pair_terms(mats[0][0], yr, "01"); zb = _pair_terms(mats[-1][0], yr, "01"); zr = (y[sub] != y2).astype(float)
    t = (mats[0][1] / mats[-1][1]) ** gamma; N = len(za); r = (K - 1) / K; f = m / n
    extrap = float((za.mean() - t * zb.mean()) / (1 - t)); T = extrap - float(zr.mean())
    Aa = 1 / ((1 - t) ** 2 * N); Ab = t ** 2 / ((1 - t) ** 2 * N)
    proxy_split = r ** 2 * (1 - f) / m * (n / (n - m)) ** 2
    proxy = (math.sqrt(Aa) + math.sqrt(Ab)) ** 2 + 1 / m + proxy_split
    p = math.exp(-2 * T ** 2 / proxy) if T > 0 else 1.0
    eps_beta = math.sqrt(proxy * math.log(1 / beta) / 2)
    dev = math.sqrt((1 / m + proxy_split) * math.log(1 / alpha) / 2)
    tau = eps_beta + dev
    a_ = alpha / 2
    GL = kl_bernoulli_lower(float(zr.mean()), m, a_)
    GL = max(GL - r * math.sqrt(math.log(1 / a_) * (1 - (m - 1) / n) / (2 * m)), 0.0)
    return dict(T=T, p_falsify=p, tau=tau, substudy_bound=float(f_K(GL, K)), extrap=extrap, Q_rep=float(zr.mean()),
                rest=rest, mats_rest=mats)


# ============================================================== v3: model-free two-sided sandwich and certified information
def all_pairs_disagreement(y, K):
    """U-statistic over all pairs: Q_all = 1 - sum_k n_k(n_k-1)/(n(n-1)).  E Q_all = Gbar + Dbar_all with
    Dbar_all = mean_{i != j} |p_i - p_j|^2 / 2 = n/(n-1) * I,  I = (1/n) sum_i |p_i - pbar|^2 (Brier information)."""
    y = np.asarray(y); n = len(y); cnt = np.bincount(y, minlength=K).astype(float)
    return float(1 - (cnt * (cnt - 1)).sum() / (n * (n - 1)))


def sandwich(mats, y, K, alpha=.05, gamma=.5, conc="kl"):
    """Two-sided, training-free certificates (rows independent given X, matchings outcome-blind).
    UPPER (assumption-free): E Q(M) = Gbar + D(M) >= Gbar for every perfect matching, so
        Gbar <= min_l Q(M_l)_U,   and R_01 = mean(1 - max p_i) <= Gbar   (1 - max p <= 1 - |p|^2).
    LOWER (A(gamma)): PYTHIA-X.
    INFORMATION I = mean |p_i - pbar|^2 (Brier information = Bayes-rule Brier skill in absolute units):
        I >= (n-1)/n * (Q_all_L - Gbar_U)     assumption-free (Hoeffding's U-statistic argument: Q_all is an
                                               average over permutations of means of n/2 independent pair terms,
                                               so the Chernoff-KL bound with N = floor(n/2) applies),
        I <= (n-1)/n * (Q_all_U - Gbar_L)     under A(gamma).
    Brier skill of the Bayes rule BSS* = I / S with S = 1 - |pbar|^2 (approximately Q_all): bounds reported via the
    same pieces.  alpha split: upper floor alpha/4, lower floor alpha/4, Q_all two sides alpha/4 each."""
    y = np.asarray(y); n = len(y); a4 = alpha / 4; L = len(mats)
    Z = [_pair_terms(P, y, "01") for P, _ in mats]
    GU = min(_upper(z, a4 / L, conc) for z in Z)
    GL = certificate_v2(mats, y, K, a4, (gamma,), "brier", conc=conc)["bound"]
    Qa = all_pairs_disagreement(y, K); N = n // 2
    QaL = kl_bernoulli_lower(Qa, N, a4); QaU = kl_bernoulli_upper(Qa, N, a4)
    c = (n - 1) / n
    return dict(brier_floor=(GL, GU), floor01=(float(f_K(GL, K)), min(GU, (K - 1) / K)),
                info=(max(c * (QaL - GU), 0.0), max(c * (QaU - GL), 0.0)),
                bss=(max(1 - GU / max(QaL - 1 / (n - 1), 1e-12), 0.0), max(1 - GL / QaU, 0.0) if QaU > 0 else 0.0),
                Q_levels=[float(z.mean()) for z in Z], Q_all=Qa)
