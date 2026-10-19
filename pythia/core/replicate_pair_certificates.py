"""Replicate-pair certificates for 0-1 Bayes floors (embedding-free, no smoothness, finite-sample).

Setting.  Rows i = 1..n are grouped into cells c (exact replicates: every row in a cell has the same conditional
label law p_c on K classes, or, with a declared sufficiency slack eps, laws within TV eps of each other).  Cells are
independent; inside a cell the labels are independent given the cell.  The target is the row-average Bayes 0-1
floor R = (1/n) sum_i (1 - max_k p_{c(i)k}).  Singleton cells contribute 0 (safe).

Theorem A (pair-disagreement / Cover-Hart inversion, unbiased; no in-cell optimism).
  For one law p: g = P(Y != Y') = 1 - sum p_k^2 and R_p = 1 - max p satisfy R_p >= f_K(g) with
  f_K(g) = (K-1)/K * (1 - sqrt(1 - K g/(K-1)))  (f_inf(g) = 1 - sqrt(1-g) is valid for every K).
  f_K is increasing and convex, so (1/n) sum_i R_i >= (n2/n) f_K(G), G = sum_c (m_c/n2) g_c over cells with
  m_c >= 2 (Jensen).  ghat_c = fraction of disagreeing pairs in cell c is UNBIASED for g_c for every m_c >= 2,
  so the only slack is Jensen + concentration (the plug-in min(phat, 1-phat) route has O(m^-1/2) optimism).
  Concentration: X_c = w_c ghat_c in [0, w_c] independent; Bernoulli extremality gives the Poisson-Bennett bound
  P(sum X_c >= s) <= exp(-(s log(s/G) - s + G)/w_max); inverted at the observed sum.
  Slack (SECOND ORDER): for two laws, 1 - sum p_i p_j = (g_i+g_j)/2 + |p_i-p_j|_2^2/2 and
  |p_i-p_j|_2^2 <= |p_i-p_j|_1 max|p_i-p_j| <= 2 TV^2.  So if every pair in a cell has TV(p_i,p_j) <= delta, use
  f_K(G_L - delta^2).  (The min-count route pays 2*delta, first order.)
Theorem B (stratified Jensen).  For an outcome-blind partition of cells into strata s (may be learned on an
  independent development split): R >= sum_s (n2_s/n) f_K(G_L,s), each stratum at alpha/S.
Theorem C (polynomial minorant, binary, exact orbits).  u = p(1-p) in [0,1/4], R_p = h(u) = (1-sqrt(1-4u))/2.
  For any polynomial q <= h on [0,1/4] of degree d, the falling-factorial estimator
  [k]_j [m-k]_j / [m]_{2j} is unbiased for u^j when m >= 2j, so sum_c w_c qhat_c is unbiased for a lower bound
  of R with no Jensen loss; q is chosen on development data by LP, concentration by weighted Hoeffding with
  exact per-cell ranges.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import linprog


# ---------------------------------------------------------------- basics
def cells_from_keys(keys):
    _, inv = np.unique(np.asarray(keys), return_inverse=True)
    return inv


def cell_counts(cells, y, K):
    cells = np.asarray(cells); y = np.asarray(y).astype(int)
    C = cells.max() + 1
    cnt = np.zeros((C, K)); np.add.at(cnt, (cells, y), 1)
    return cnt


def f_K(g, K=None):
    g = np.clip(np.asarray(g, float), 0, None)
    if K is None:
        return 1 - np.sqrt(np.clip(1 - g, 0, None))
    r = K / (K - 1)
    return (K - 1) / K * (1 - np.sqrt(np.clip(1 - r * g, 0, None)))


def poisson_bennett_lower(S, wmax, alpha):
    """Smallest G with P(sum X >= S) > alpha under the Poisson-Bennett bound (X_c in [0, w_c], w_c <= wmax)."""
    if S <= 0:
        return 0.0
    L = math.log(1 / alpha)
    fn = lambda G: (S * math.log(S / G) - S + G) / wmax - L
    lo, hi = 1e-15, S
    if fn(lo) <= 0:
        return 0.0
    for _ in range(100):
        mid = .5 * (lo + hi)
        if fn(mid) > 0:
            lo = mid
        else:
            hi = mid
    return lo


def weighted_hoeffding_lower(est, w, ranges, alpha):
    est, w, ranges = map(np.asarray, (est, w, ranges))
    return float((w * est).sum() - math.sqrt((w ** 2 * ranges ** 2).sum() * math.log(1 / alpha) / 2))


# ---------------------------------------------------------------- chunking and within-cell independence
def chunk_cells(cells, M, seed=0):
    """Split every cell into random chunks of at most M rows (outcome-blind).  Rows of a cell are iid given the
    cell, so chunks are again exact replicate cells and are mutually independent; this trades a few pairs for many
    more independent concentration units (needed when cells are large).  M=None keeps cells."""
    cells = np.asarray(cells)
    if M is None:
        return cells
    rng = np.random.default_rng(seed); order = rng.permutation(len(cells))
    out = np.empty(len(cells), np.int64); pos = {}
    for i in order:
        c = int(cells[i]); j = pos.get(c, 0); pos[c] = j + 1; out[i] = c * 10 ** 6 + j // M
    return cells_from_keys(out)


def within_cell_independence_audit(cells, y, K, alpha=.05):
    """Necessary condition of conditional iid labels inside cells: every cell's pair-agreement probability
    s_c = sum_k p_ck^2 >= 1/K.  One-sided Hoeffding test over cells (agreement fractions in [0,1], independent across
    cells) of H0: mean_c s_c >= 1/K, via the Chernoff-KL lower tail.  Rejection => U-statistic routes (pair, minorant) are refused; the min-count
    route (fixed-action domination) does not need within-cell independence."""
    cnt = cell_counts(cells, y, K); m = cnt.sum(1); ok = m >= 2
    agree = (cnt[ok] * (cnt[ok] - 1)).sum(1) / (m[ok] * (m[ok] - 1))
    C = int(ok.sum())
    if C == 0:
        return dict(reject=False, p=1.0, mean_agree=float("nan"), cells=0)
    x, q = float(agree.mean()), 1 / K
    if x >= q:
        p = 1.0
    else:   # Chernoff-KL (Hoeffding 1963) lower tail for independent [0,1] variables with mean >= q
        kl = (x * math.log(x / q) if x > 0 else 0.0) + (1 - x) * math.log((1 - x) / (1 - q))
        p = math.exp(-C * kl)
    return dict(reject=p < alpha, p=p, mean_agree=float(agree.mean()), floor=1 / K, cells=C)


# ---------------------------------------------------------------- Theorem A / B
def pair_disagreement(cnt):
    m = cnt.sum(1)
    agree = (cnt * (cnt - 1)).sum(1)
    ok = m >= 2
    g = np.zeros(len(m)); g[ok] = 1 - agree[ok] / (m[ok] * (m[ok] - 1))
    return m, g, ok


def pair_certificate(cells, y, K, alpha=.05, strata=None, slack=0.0, K_inversion="K"):
    """Lower confidence bound on the row-average 0-1 Bayes floor (Theorems A, B).  slack = declared bound on the
    pairwise TV distance between the label laws of rows sharing a cell (0 for exact replicates)."""
    cnt = cell_counts(cells, y, K); n = cnt.sum()
    m, g, ok = pair_disagreement(cnt)
    if strata is None:
        strata = np.zeros(len(m), int)
    strata = np.asarray(strata)
    S_ids = np.unique(strata[ok]); a = alpha / max(len(S_ids), 1)
    Kinv = K if K_inversion == "K" else None
    total, parts = 0.0, []
    for s in S_ids:
        sel = ok & (strata == s); n2 = m[sel].sum(); w = m[sel] / n2
        Ghat = float((w * g[sel]).sum()); GL = poisson_bennett_lower(Ghat, w.max(), a)
        val = float(f_K(max(GL - slack ** 2, 0.0), Kinv)) * n2 / n
        total += val; parts.append(dict(stratum=int(s), n2=int(n2), Ghat=Ghat, GL=GL, contrib=val))
    return dict(bound=total, n=int(n), n_paired=int(m[ok].sum()), parts=parts)


# ---------------------------------------------------------------- accepted baseline (plug-in / min-count)
def kl_bernoulli_lower(xbar, n, alpha):
    """Smallest mean q with P(mean of n independent [0,1] variables with mean q >= xbar) > alpha (Chernoff-KL)."""
    if xbar <= 0:
        return 0.0
    L = math.log(1 / alpha) / n
    kl = lambda x, q: (x * math.log(x / q) if x > 0 else 0) + ((1 - x) * math.log((1 - x) / (1 - q)) if x < 1 else 0)
    lo, hi = 1e-15, xbar
    if kl(xbar, lo) <= L:
        return 0.0
    for _ in range(100):
        mid = .5 * (lo + hi)
        if kl(xbar, mid) > L:
            lo = mid
        else:
            hi = mid
    return lo


def mincount_certificate(cells, y, K, alpha=.05, slack=0.0):
    """Accepted route: exact in-cell SAA S = sum_c (m_c - max_k n_ck)/n (hard-label averaging, Ishida et al.),
    which is <= mean_i 1{Y_i != a*_c} (fixed-action domination).  Concentration = best of three, alpha/3 each:
    instance-level weighted Hoeffding (as in Ishida et al.), cell-level Poisson-Bennett, and ROW-level Chernoff-KL
    via domination (the earlier Pythia dominated-SAA route; strongest when cells are large)."""
    cnt = cell_counts(cells, y, K); n = cnt.sum(); m = cnt.sum(1); ok = m > 0
    est = 1 - cnt[ok].max(1) / m[ok]; w = m[ok] / n; r = 1 - 1 / K; S = float((w * est).sum())
    h = weighted_hoeffding_lower(est, w, np.full(len(est), r), alpha / 3)
    b = poisson_bennett_lower(S / r, (w).max(), alpha / 3) * r
    d = kl_bernoulli_lower(S, int(n), alpha / 3)
    h_only = weighted_hoeffding_lower(est, w, np.full(len(est), r), alpha)
    return dict(bound=max(max(h, b, d, 0.0) - 2 * slack, 0.0), plug_in=S,
                hardlabel_hoeffding=max(h_only - 2 * slack, 0.0), domination_kl=max(d - 2 * slack, 0.0))


# ---------------------------------------------------------------- Theorem C (any K): linear cell certificates
def _ff(x, j):
    x = np.asarray(x, float); out = np.ones_like(x)
    for t in range(j):
        out = out * (x - t)
    return out


def s_estimates(cnt):
    """Unbiased estimates of s = sum_k p_k^2 (needs m >= 2) and s^2 (needs m >= 4) from count rows."""
    cnt = np.atleast_2d(np.asarray(cnt, float)); m = cnt.sum(1)
    f2 = _ff(cnt, 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        s1 = f2.sum(1) / _ff(m, 2)
        s2 = (_ff(cnt, 4).sum(1) + f2.sum(1) ** 2 - (f2 ** 2).sum(1)) / _ff(m, 4)
    return np.nan_to_num(s1), np.nan_to_num(s2)


def phi(g, K):
    return f_K(g, K)


def fit_g_minorant(cnt_dev, K, d, grid=20001):
    """Polynomial q(g) = sum_j c_j g^j <= f_K(g) on [0, (K-1)/K], maximising the development sum of the unbiased
    cell estimates (cells with m >= 2d).  Shifted down by the exact max violation on a dense grid."""
    cnt_dev = np.asarray(cnt_dev, float); m = cnt_dev.sum(1); cnt_dev = cnt_dev[m >= 2 * d]; m = cnt_dev.sum(1)
    s1, s2 = s_estimates(cnt_dev)
    G = [np.ones_like(s1), 1 - s1, 1 - 2 * s1 + s2][:d + 1]          # unbiased for 1, g, g^2
    obj = -np.array([(m * Gj).sum() for Gj in G])
    gmax = (K - 1) / K
    gg = np.concatenate([np.linspace(0, gmax, grid), gmax - np.geomspace(1e-12, 1e-2, 2000)])
    A = np.column_stack([gg ** j for j in range(d + 1)])
    coef = linprog(obj, A_ub=A, b_ub=phi(gg, K), bounds=[(-20, 20)] * (d + 1), method="highs").x
    gd = np.concatenate([np.linspace(0, gmax, 400_001), gmax - np.geomspace(1e-14, 1e-2, 40000)])
    viol = max(0.0, float((np.column_stack([gd ** j for j in range(d + 1)]) @ coef - phi(gd, K)).max()))
    coef = coef.copy(); coef[0] -= viol + 1e-12
    return coef


def _cell_values(counts_all, option, coefs, K):
    """Estimator values for an array of count rows under an option ('min', 'q1', 'q2')."""
    counts_all = np.atleast_2d(counts_all); m = counts_all.sum(1)
    if option == "min":
        return 1 - counts_all.max(1) / m
    c = coefs[int(option[1])]; s1, s2 = s_estimates(counts_all)
    G = [np.ones_like(s1), 1 - s1, 1 - 2 * s1 + s2]
    return sum(c[j] * G[j] for j in range(len(c)))


def _compositions(m, K):
    from itertools import combinations_with_replacement
    out = []
    for comb in combinations_with_replacement(range(K), m):
        out.append(np.bincount(comb, minlength=K))
    return np.array(out)


_RANGE_CACHE = {}


def _range(m, option, coefs, K):
    key = (m, option, K, tuple(np.round(coefs.get(int(option[1]), [0]), 12)) if option != "min" else ())
    if key not in _RANGE_CACHE:
        if option == "min":
            _RANGE_CACHE[key] = (0.0, 1 - math.ceil(m / K) / m)
        elif math.comb(m + K - 1, K - 1) <= 20000:
            v = _cell_values(_compositions(m, K), option, coefs, K); _RANGE_CACHE[key] = (float(v.min()), float(v.max()))
        else:
            c = coefs[int(option[1])]; a = [c[0] + c[1] + (c[2] if len(c) > 2 else 0), -c[1] - 2 * (c[2] if len(c) > 2 else 0),
                                            (c[2] if len(c) > 2 else 0)]
            lo = a[0] - abs(a[1]) - abs(a[2]); hi = a[0] + abs(a[1]) + abs(a[2]); _RANGE_CACHE[key] = (lo, hi)
    return _RANGE_CACHE[key]


SIZE_CLASSES = ((2, 2), (3, 3), (4, 5), (6, 8), (9, 16), (17, 10 ** 9))


_SC_LO = np.array([a for a, _ in SIZE_CLASSES])


def size_class(m):
    m = np.asarray(m); sc = np.searchsorted(_SC_LO, m, side="right") - 1
    return np.where(m >= 2, sc, -1)


def linear_certificate(cells, y, K, rule, coefs, alpha=.05, slack=0.0):
    """Theorem C: R >= sum_c (m_c/n) E e_c with E e_c <= R_c for every cell (min-count: domination; q_d: unbiased
    for a minorant of f_K(g_c)).  rule[size_class] in {'min','q1','q2'} chosen outcome-blind (development data).
    Concentration: better of weighted Hoeffding (exact ranges) and shifted Poisson-Bennett, alpha/2 each.
    slack = declared pairwise TV delta: 'min' pays 2 delta, 'q1' pays c1 delta^2 (requires c1 >= 0); 'q2' refused."""
    cnt = cell_counts(cells, y, K) if not isinstance(cells, tuple) else cells[0]
    n = cnt.sum(); m = cnt.sum(1).astype(int)
    est, lo, hi, w, pen = [], [], [], [], 0.0
    scs = size_class(m)
    for sc in range(len(SIZE_CLASSES)):
        sel = np.flatnonzero(scs == sc)
        if len(sel) == 0:
            continue
        opt = rule.get(sc, "min")
        if slack > 0 and opt == "q2":
            opt = "q1"
        if opt != "min" and m[sel].min() < 2 * int(opt[1]):
            opt = "min"
        v = _cell_values(cnt[sel], opt, coefs, K)
        um, inv = np.unique(m[sel], return_inverse=True)
        r = np.array([_range(int(x), opt, coefs, K) for x in um])[inv]
        est.append(v); lo.append(r[:, 0]); hi.append(r[:, 1]); w.append(m[sel] / n)
        if slack > 0:
            pen += (m[sel].sum() / n) * (2 * slack if opt == "min" else max(coefs[1][1], 0) * slack ** 2)
    if not est:
        return dict(bound=0.0, estimate=0.0)
    est, lo, hi, w = map(np.concatenate, (est, lo, hi, w))
    hb = weighted_hoeffding_lower(est, w, hi - lo, alpha / 2)
    sh = (w * (est - lo)).sum(); wr = (w * (hi - lo)).max()
    bb = poisson_bennett_lower(float(sh), float(wr), alpha / 2) + float((w * lo).sum()) if wr > 0 else float((w * est).sum())
    return dict(bound=max(max(hb, bb) - pen, 0.0), estimate=float((w * est).sum()))


def choose_rule(cnt_dev, K, coefs, alpha=.05):
    """Pick, per size class, the option maximising the development certificate (greedy, two passes)."""
    rule = {sc: "min" for sc in range(len(SIZE_CLASSES))}
    for _ in range(2):
        for sc in range(len(SIZE_CLASSES)):
            best, bv = rule[sc], -1
            for opt in ("min", "q1", "q2"):
                if int(opt[1]) not in coefs if opt != "min" else False:
                    continue
                trial = dict(rule); trial[sc] = opt
                v = linear_certificate((cnt_dev,), None, K, trial, coefs, alpha)["bound"]
                if v > bv + 1e-12:
                    best, bv = opt, v
            rule[sc] = best
    return rule


# ---------------------------------------------------------------- nuisance-quotient audit
def invariance_audit(canon_cells, residual_dist, y, n_perm=999, max_pairs_per_cell=50, seed=0):
    """Falsification test for a declared nuisance quotient.  Within canonical cells, correlate the label
    disagreement of row pairs with the distance between what the canonicaliser removed (residual_dist(i, j)).
    Under invariance (Y independent of the removed residual given the canonical form) the residuals are
    exchangeable within cells, so permuting them within cells gives an exact one-sided p-value."""
    rng = np.random.default_rng(seed); canon_cells = np.asarray(canon_cells); y = np.asarray(y)
    groups = [np.flatnonzero(canon_cells == c) for c in np.unique(canon_cells)]
    groups = [g for g in groups if len(g) >= 2]
    if not groups:
        return dict(p=1.0, stat=0.0, pairs=0)

    def pairs_of(gr):
        P = [(a, b) for ai, a in enumerate(gr) for b in gr[ai + 1:]]
        if len(P) > max_pairs_per_cell:
            P = [P[t] for t in rng.choice(len(P), max_pairs_per_cell, replace=False)]
        return P
    plist = [pairs_of(g) for g in groups]
    dis = np.array([y[a] != y[b] for P in plist for a, b in P], float)

    def stat(perm_maps):
        d = np.array([residual_dist(perm_maps[gi].get(a, a), perm_maps[gi].get(b, b))
                      for gi, P in enumerate(plist) for a, b in P])
        if d.std() == 0 or dis.std() == 0:
            return 0.0
        return float(np.corrcoef(d, dis)[0, 1])
    ident = [dict() for _ in groups]
    s0 = stat(ident); ge = 1
    for _ in range(n_perm):
        maps = [dict(zip(g, rng.permutation(g))) for g in groups]
        ge += stat(maps) >= s0
    return dict(p=ge / (n_perm + 1), stat=s0, pairs=int(len(dis)))
