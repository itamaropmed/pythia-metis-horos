"""Agreement-Field Information (AFI): a pair-design information theory for records made of a TABLE and a TEXT.

Primitive.  For records r_i = (x_i, t_i) with outcome laws p_i = P(Y | r_i) on K classes, the AGREEMENT KERNEL is
    kappa(i, j) = P(Y_i = Y_j) = <p_i, p_j>                         (cross-collision probability, Renyi order 2)
and the identity  1 - kappa(i, j) = (g_i + g_j)/2 + |p_i - p_j|^2 / 2,  g = 1 - |p|^2  (Gini / Brier floor).

Pair designs.  A design is an outcome-blind rule that selects pairs of rows using RELATIONS between records
(same table cell, text-near at scale s, both, neither).  Balanced designs (every row used equally) satisfy
    E[disagreement | design] = Gbar + D(design),   D = mean over selected pairs of |p_i - p_j|^2 / 2 >= 0.
Information is a CONTRAST between designs: how much more often outcomes agree when records are related.

Quantities (all in Brier units; I(.) = variance of the outcome law explained by the representation):
    I(X)          table information              Q(random) - Q(same cell)                 two-sided, assumption-free
    Delta I(T|X)  text information beyond table  Q(same cell) - Q(same cell & text-near)  lower bound, assumption-free
    I(T)          text information               Q(random) - Q(text-near)                 lower bound, assumption-free
    Gbar_X        table floor                    Q(same cell)                              two-sided, assumption-free
    Gbar_XT       joint floor                    Q(same cell & text-near)                  upper bound, assumption-free
    spectrum      Delta I(T|X; s)                per text scale s                          where in the text the signal lives
    chain rule    I(X,T) >= I(X) + Delta I(T|X) contrast telescopes exactly at design level
    design co-information  I(X) + I(T) - [I(X) + Delta I(T|X)]  (> 0 echo/redundancy, < 0 synergy; exploratory)
Assumptions for the certified statements: rows independent given their records (for I(T): iid rows), the table is
discrete (cells = exact replicates of X), designs outcome-blind (random tie-breaking, shuffled row order).
"""
from __future__ import annotations
import math
import numpy as np
from .multiscale_pair_extrapolation import nested_matchings, kl_bernoulli_lower, kl_bernoulli_upper


def _allpairs_disagree(y, K):
    m = len(y); c = np.bincount(y, minlength=K).astype(float)
    return 1 - (c * (c - 1)).sum() / (m * (m - 1))


def afi_certificate(cells, E, y, K, alpha=.05, levels=3, seed=0, text_levels_global=4):
    """Simultaneous (Bonferroni over the 6 reported statements) certificates.  cells: table cell id per row;
    E: outcome-blind text geometry (rows x dims); y: labels 0..K-1."""
    rng = np.random.default_rng(seed); cells = np.asarray(cells); y = np.asarray(y); a = alpha / 6
    blk = 2 ** levels
    # ---- retained rows: per cell, a random multiple of 2^levels (outcome-blind)
    keep, cell_rows = [], []
    for c in np.unique(cells):
        I = np.flatnonzero(cells == c); I = I[rng.permutation(len(I))]; k = (len(I) // blk) * blk
        if k >= blk:
            keep.append(I[:k]); cell_rows.append(I[:k])
    keep = np.concatenate(keep); n = len(keep)
    w = np.array([len(I) for I in cell_rows]) / n; ms = np.array([len(I) for I in cell_rows])
    # ---- designs on the retained rows
    Q_rand = _allpairs_disagree(y[keep], K)
    Q_cell = float(sum(wc * _allpairs_disagree(y[I], K) for wc, I in zip(w, cell_rows)))
    lvlZ = [[] for _ in range(levels)]
    for I in cell_rows:
        for l, (P, s) in enumerate(nested_matchings(E[I], levels, perm_seed=int(rng.integers(1 << 30)))):
            lvlZ[l].append((y[I][P[:, 0]] != y[I][P[:, 1]]).astype(float))
    lvlZ = [np.concatenate(z) for z in lvlZ]; Q_ct = [float(z.mean()) for z in lvlZ]; N = len(lvlZ[0])
    # ---- concentration (Hoeffding units; dependent statistics combine via sqrt-sum)
    pr_rand = 1 / (n // 2); pr_cell = float((w ** 2 / (ms // 2)).sum()); pr_ct = 1 / N
    L = math.log(1 / a); Ll = math.log(levels / a)
    dev = lambda pr, LL=L: math.sqrt(pr * LL / 2)
    mstar = int(ms.min()); rK = 1 - 1 / K
    # I(X): E(Q_rand - Q_cell) = I(X) + I_XT/(n-1) - sum_c w_c V_c/(m_c-1)
    d_IX = dev((math.sqrt(pr_rand) + math.sqrt(pr_cell)) ** 2)
    IX = (max(Q_rand - Q_cell - d_IX - rK / (n - 1), 0.0), Q_rand - Q_cell + d_IX + rK / (mstar - 1))
    GX = (max(Q_cell - dev(pr_cell) - rK / (mstar - 1), 0.0), Q_cell + dev(pr_cell))
    # Delta I(T|X): best level, Bonferroni over levels
    d_dI = dev((math.sqrt(pr_cell) + math.sqrt(pr_ct)) ** 2, Ll)
    spectrum = [Q_cell - q for q in Q_ct]
    dI_L = max(((mstar - 1) / mstar) * (max(spectrum) - d_dI), 0.0)
    GXT_U = min(min(Q_ct) + dev(pr_ct, Ll), 1.0)
    p_ci = min(1.0, levels * math.exp(-2 * max(max(spectrum), 0) ** 2 / ((math.sqrt(pr_cell) + math.sqrt(pr_ct)) ** 2)))
    # I(T) (text only, iid rows): global nested text matchings on a random multiple of 2^text_levels_global rows
    bg = 2 ** text_levels_global; G = rng.permutation(len(y))[: (len(y) // bg) * bg]
    QT = [float((y[G][P[:, 0]] != y[G][P[:, 1]]).mean()) for P, s in nested_matchings(E[G], text_levels_global, perm_seed=int(rng.integers(1 << 30)))]
    Q_randG = _allpairs_disagree(y[G], K); nG = len(G)
    d_IT = dev((math.sqrt(1 / (nG // 2)) + math.sqrt(1 / (nG // 2))) ** 2, math.log(text_levels_global / a))
    IT_L = max(Q_randG - min(QT) - d_IT - rK / (nG - 1), 0.0)
    return dict(I_X=IX, G_X=GX, dI_lower=dI_L, G_XT_upper=GXT_U, I_T_lower=IT_L, p_text_adds_nothing=p_ci,
                spectrum=spectrum, Q=dict(rand=Q_rand, cell=Q_cell, cell_text=Q_ct, text=QT),
                design_coinfo_point=(Q_rand - Q_cell) + (Q_randG - min(QT)) - ((Q_rand - Q_cell) + max(spectrum)),
                n_used=int(n), cells_used=len(cell_rows), m_star=mstar, keep=keep, keepG=G)
