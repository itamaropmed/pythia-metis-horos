"""LP bounds with a relaxation dial: partial-identification programs for the best achievable risk (prediction:
0-1 / Brier / log; estimation: MSE) from table + text, and their relaxation path as the assumption slack -> 0.

LP-1  (variogram / design LP).  Unknowns: joint Brier floor G = mean(1-|p|^2) (or mean sigma^2 for MSE), within-cell text
drifts D_l at the nested text scales l, within-cell outcome-law variance V (= DeltaI(T|X)).  Certified data enter as
simultaneous confidence boxes; structural facts are exact linear constraints; the shape assumption is RELAXED by eps:
    lo_l <= G + D_l <= hi_l                  (within-cell text matchings, every row once)
    gxL  <= G + V   <= gxU                   (table floor, cells are exact replicates)
    0 <= D_l <= 2 V                          (a matching's drift is at most twice the within-cell variance: exact)
    D_a <= t_ab^gamma D_b + eps   (a < b)    (A(gamma) relaxed by eps; eps = inf is assumption-free)
min G = certified lower floor L(gamma, eps);  max G = upper floor;  min/max V = text-information interval.
eps -> 0 traces the path from the assumption-free (vacuous) floor to the full-assumption floor; the tipping eps*
(where L hits 0) is the assumption budget of the certificate.

LP-2  (moment LP for the 0-1 floor, Hausdorff-type).  Inside exact replicate cells, unbiased U-statistics estimate
mu_j = E_rows[pbar_c^j] (binary) for j = 1..J.  Over all row-weighted distributions w of the cell law on a grid,
    min / max  sum_g w_g min(p_g, 1-p_g)   s.t.  mu_j in [lo_j - eps, hi_j + eps],  sum w = 1, w >= 0,
is the sharp 0-1 floor given J certified moments (Jensen's f_K(G) is the J = 2 relaxation of it).  eps widens the
moment boxes (e.g. for text-neighbour groups, where eps is calibrated by the drift bound of LP-1).
"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import linprog
from .multiscale_pair_extrapolation import nested_matchings, kl_bernoulli_lower, kl_bernoulli_upper, betting_lower, betting_upper


# ------------------------------------------------------------------ LP-1 inputs
def design_statistics(cells, E, y, K=2, levels=3, alpha=.05, seed=0, target="brier", y_range=None):
    """Within-cell nested text matchings + within-cell all-pairs; simultaneous boxes (Bonferroni over levels+1).
    target 'brier': z = 1{Y_i != Y_j} (mean = G + D);  'mse': z = (Y_i - Y_j)^2/2 (mean = mean sigma^2 + D)."""
    rng = np.random.default_rng(seed); cells = np.asarray(cells); y = np.asarray(y)
    blk = 2 ** levels; groups = []
    for c in np.unique(cells):
        I = np.flatnonzero(cells == c); I = I[rng.permutation(len(I))]; k = (len(I) // blk) * blk
        if k >= blk:
            groups.append(I[:k])
    keep = np.concatenate(groups); n = len(keep); w = np.array([len(I) for I in groups]) / n
    Z = [[] for _ in range(levels)]; S = [[] for _ in range(levels)]
    pair = (lambda a, b: (y[a] != y[b]).astype(float)) if target == "brier" else (lambda a, b: .5 * (y[a] - y[b]) ** 2)
    for I in groups:
        for l, (P, s) in enumerate(nested_matchings(E[I], levels, perm_seed=int(rng.integers(1 << 30)))):
            Z[l].append(pair(I[P[:, 0]], I[P[:, 1]])); S[l].append(s * len(P))
    Z = [np.concatenate(z) for z in Z]; s_l = [sum(S[l]) / len(Z[l]) for l in range(levels)]
    a = alpha / (2 * (levels + 1)); B = 1.0 if target == "brier" else .5 * y_range ** 2
    box = []
    for z in Z:
        zz = np.clip(z / B, 0, 1)
        box.append((B * max(kl_bernoulli_lower(zz.mean(), len(zz), a / 2), betting_lower(zz, a / 2)),
                    B * min(kl_bernoulli_upper(zz.mean(), len(zz), a / 2), betting_upper(zz, a / 2))))
    # table floor box: within-cell all-pairs U-statistics (Hoeffding over independent cells)
    if target == "brier":
        qc = []; pr = 0.0
        for wc, I in zip(w, groups):
            cnt = np.bincount(y[I], minlength=K).astype(float); m = len(I)
            qc.append(1 - (cnt * (cnt - 1)).sum() / (m * (m - 1))); pr += wc ** 2 / (m // 2)
        Q = float(np.dot(w, qc)); mstar = min(len(I) for I in groups); dev = math.sqrt(pr * math.log(2 / a) / 2)
        gx = (max(Q - dev - (1 - 1 / K) / (mstar - 1), 0.0), Q + dev)
    else:
        qc = []; pr = 0.0
        for wc, I in zip(w, groups):
            v = y[I]; m = len(v); qc.append(v.var(ddof=1)); pr += wc ** 2 / (m // 2)
        Q = float(np.dot(w, qc)); mstar = min(len(I) for I in groups); dev = B * math.sqrt(pr * math.log(2 / a) / 2) * 2
        gx = (max(Q - dev, 0.0), Q + dev)
    # global text family (pairs may cross cells; same G, its own drift family), on the same retained rows
    Gl = 4; nG = (len(keep) // 2 ** Gl) * 2 ** Gl; Kp = keep[rng.permutation(len(keep))[:nG]]
    gbox, gs, gQ = [], [], []
    for P, s_ in nested_matchings(E[Kp], Gl, perm_seed=int(rng.integers(1 << 30))):
        z = np.clip(pair(Kp[P[:, 0]], Kp[P[:, 1]]) / B, 0, 1); ag = alpha / (2 * Gl)
        gbox.append((B * max(kl_bernoulli_lower(z.mean(), len(z), ag / 2), betting_lower(z, ag / 2)) * nG / len(keep),
                     B * min(kl_bernoulli_upper(z.mean(), len(z), ag / 2), betting_upper(z, ag / 2)) * nG / len(keep) + (1 - nG / len(keep)) * B))
        gs.append(s_); gQ.append(float(z.mean() * B))
    return dict(box=box, s=s_l, gx=gx, Q=[float(z.mean()) for z in Z], keep=keep, Qcell=Q, gbox=gbox, gs=gs, gQ=gQ)


def variogram_lp(st, gamma=.5, eps=0.0, what="G", sense="min", use_global=True):
    """Solve LP-1.  Variables x = [G, V, D_0..D_{L-1} (within-cell text), F_0..F_{M-1} (global text)]."""
    L = len(st["box"]); M = len(st["gbox"]) if use_global else 0; nv = 2 + L + M; A, b = [], []
    def vec(pairs):
        r = np.zeros(nv)
        for i, v in pairs:
            r[i] += v
        return r
    G, V = 0, 1; D = lambda l: 2 + l; F = lambda l: 2 + L + l
    for l, (lo, hi) in enumerate(st["box"]):
        A.append(vec([(G, 1), (D(l), 1)])); b.append(hi); A.append(vec([(G, -1), (D(l), -1)])); b.append(-lo)
        A.append(vec([(V, -2), (D(l), 1)])); b.append(0.0)                       # D_l <= 2V (exact)
    for l in range(M):
        lo, hi = st["gbox"][l]
        A.append(vec([(G, 1), (F(l), 1)])); b.append(hi); A.append(vec([(G, -1), (F(l), -1)])); b.append(-lo)
    A.append(vec([(G, 1), (V, 1)])); b.append(st["gx"][1]); A.append(vec([(G, -1), (V, -1)])); b.append(-st["gx"][0])
    if eps is not None and np.isfinite(eps):
        for fam, ss, idx in ((L, st["s"], D), (M, st.get("gs", []), F)):
            for a_ in range(fam):
                for b_ in range(a_ + 1, fam):
                    t = (ss[a_] / ss[b_]) ** gamma
                    if t < 1:
                        A.append(vec([(idx(a_), 1), (idx(b_), -t)])); b.append(eps)
    c = np.zeros(nv); c[G if what == "G" else V] = 1 if sense == "min" else -1
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=[(0, None)] * nv, method="highs")
    if res.status != 0:
        return None                                  # infeasible: the data refute A(gamma) at this eps
    return float(res.x[G if what == "G" else V])


def relaxation_path(st, gamma=.5, n_eps=40):
    """L(eps) for eps from 0 up to the largest box value; tipping eps* = smallest eps with L = 0."""
    emax = max(h for _, h in st["box"]); grid = np.concatenate([[0.0], np.geomspace(1e-4, emax, n_eps)])
    path = [(float(e), variogram_lp(st, gamma, e, "G", "min")) for e in grid]
    tip = next((e for e, v in path if v is not None and v <= 1e-9), None)
    return path, tip


# ------------------------------------------------------------------ LP-2: moment LP for the 0-1 floor (binary)
def _ff(x, j):
    out = np.ones_like(np.asarray(x, float))
    for t in range(j):
        out = out * (np.asarray(x, float) - t)
    return out


def moment_boxes(groups, y, J=4, alpha=.05):
    """Row-weighted unbiased estimates of mu_j = E[p^j], j=1..J, from groups of exact replicates (cells), with
    simultaneous Hoeffding boxes (each group's estimate lies in [0,1], groups independent, effective size floor(m/j))."""
    n = sum(len(g) for g in groups); boxes = []; a = alpha / J
    for j in range(1, J + 1):
        est, w2 = 0.0, 0.0
        for g in groups:
            m = len(g); k = float(y[g].sum()); w = m / n
            if m < j:
                continue
            est += w * float(_ff(k, j) / _ff(m, j)); w2 += w ** 2 / (m // j)
        dev = math.sqrt(w2 * math.log(2 / a) / 2)
        boxes.append((max(est - dev, 0.0), min(est + dev, 1.0)))
    return boxes


def moment_lp_01(boxes, eps=0.0, grid=401, sense="min", objective="01"):
    p = np.linspace(0, 1, grid); J = len(boxes)
    f = np.minimum(p, 1 - p) if objective == "01" else 2 * p * (1 - p)
    A, b = [], []
    for j, (lo, hi) in enumerate(boxes, start=1):
        A.append(p ** j); b.append(hi + eps); A.append(-(p ** j)); b.append(-(lo - eps))
    res = linprog(f if sense == "min" else -f, A_ub=np.array(A), b_ub=np.array(b), A_eq=np.ones((1, grid)), b_eq=[1.0],
                  bounds=[(0, None)] * grid, method="highs")
    return None if res.status != 0 else float(f @ res.x)


# ------------------------------------------------------------------ conditional (fibre) information geometry of text
def conditional_text_map(cells, E, y, K, alpha=.05, min_cell=16, seed=0):
    """Per table cell x: the agreement pull-back geometry of the text fibre over x.
    Observable per cell (no assumption): Q_cell(x) (all-pairs disagreement = floor of the cell law + O(1/m)) and
    Q_txt(x) (text-NN matching disagreement = mean g + drift).  Reported:
      dI_point(x) = Q_cell - Q_txt      (text information in the fibre, conservative point estimate),
      dI_lower(x) certified with Holm over cells (family-wise alpha),
      resolution(x) = dI_point / Q_cell (share of the cell's outcome disagreement that text neighbours resolve)."""
    from .anchored_information import _greedy_pairs
    rng = np.random.default_rng(seed); out = []
    ids = [c for c in np.unique(cells) if (cells == c).sum() >= min_cell]
    for c in ids:
        I = np.flatnonzero(cells == c); I = I[rng.permutation(len(I))]; m = len(I)
        cnt = np.bincount(y[I], minlength=K).astype(float); Qc = 1 - (cnt * (cnt - 1)).sum() / (m * (m - 1))
        G = E[I] + 1e-9 * rng.normal(size=E[I].shape)
        z = np.array([float(y[I[a]] != y[I[b]]) for a, b in _greedy_pairs(G)]); N = len(z)
        pr = (math.sqrt(1 / (m // 2)) + math.sqrt(1 / N)) ** 2
        stat = Qc - z.mean(); p = math.exp(-2 * stat ** 2 / pr) if stat > 0 else 1.0
        out.append(dict(cell=int(c), m=m, Q_cell=float(Qc), Q_txt=float(z.mean()), dI_point=float(stat), p=p, pr=pr,
                        resolution=float(stat / Qc) if Qc > 0 else 0.0))
    order = np.argsort([o["p"] for o in out]); k = len(out)
    for rank, i in enumerate(order):                          # Holm step-down
        a_i = alpha / (k - rank); o = out[i]
        o["dI_lower"] = max(((o["m"] - 1) / o["m"]) * (o["dI_point"] - math.sqrt(o["pr"] * math.log(1 / a_i) / 2)), 0.0)
    return out
