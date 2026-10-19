import math
from itertools import product

import numpy as np

from pythia.core.replicate_pair_certificates import (
    f_K, s_estimates, fit_g_minorant, cell_counts, pair_certificate, mincount_certificate, linear_certificate,
    choose_rule, chunk_cells, within_cell_independence_audit)


def test_cover_hart_inversion_is_a_minorant():
    rng = np.random.default_rng(0)
    for K in (2, 3, 5, 11):
        P = rng.dirichlet(np.full(K, .3), 5000)
        g = 1 - (P ** 2).sum(1)
        assert np.all(1 - P.max(1) >= f_K(g, K) - 1e-12)
        assert np.all(1 - P.max(1) >= f_K(g, None) - 1e-12)


def test_s_estimates_unbiased_exact():
    p = np.array([.5, .3, .2]); m = 5
    e1 = e2 = 0.0
    for seq in product(range(3), repeat=m):
        pr = np.prod(p[list(seq)]); cnt = np.bincount(seq, minlength=3)
        s1, s2 = s_estimates(cnt); e1 += pr * s1[0]; e2 += pr * s2[0]
    s = (p ** 2).sum()
    assert abs(e1 - s) < 1e-12 and abs(e2 - s ** 2) < 1e-12


def test_fitted_minorant_below_f_everywhere():
    rng = np.random.default_rng(1)
    for K in (2, 4):
        cells = np.repeat(np.arange(800), 4); P = rng.dirichlet(np.ones(K), 800)
        y = np.array([rng.choice(K, p=P[c]) for c in cells]); cnt = cell_counts(cells, y, K)
        for d in (1, 2):
            c = fit_g_minorant(cnt, K, d); g = np.linspace(0, (K - 1) / K, 200001)
            assert np.max(np.column_stack([g ** j for j in range(d + 1)]) @ c - f_K(g, K)) <= 1e-10


def test_certificates_valid_in_simulation():
    rng = np.random.default_rng(2); K = 3; viol = 0
    for rep in range(15):
        C = 1200; P = rng.dirichlet(np.full(K, .5), C); ms = rng.choice([1, 2, 3, 5], C)
        cells = np.repeat(np.arange(C), ms); y = np.array([rng.choice(K, p=P[c]) for c in cells])
        gold = (1 - P.max(1))[cells].mean()
        cnt = cell_counts(cells, y, K); coefs = {d: fit_g_minorant(cnt, K, d) for d in (1, 2)}
        rule = choose_rule(cnt, K, coefs, .025)
        for b in (pair_certificate(cells, y, K, .05)["bound"], mincount_certificate(cells, y, K, .05)["bound"],
                  linear_certificate(cells, y, K, rule, coefs, .05)["bound"]):
            viol += b > gold
    assert viol <= 1


def test_chunking_keeps_membership():
    cells = np.repeat(np.arange(10), 37); ch = chunk_cells(cells, 8, seed=0)
    for c in np.unique(ch):
        assert len(np.unique(cells[ch == c])) == 1 and (ch == c).sum() <= 8


def test_independence_audit():
    rng = np.random.default_rng(3); K = 10
    cells = np.repeat(np.arange(1500), 2)
    y_iid = rng.integers(K, size=cells.size)
    y_wor = np.concatenate([rng.choice(K, 2, replace=False) for _ in range(1500)])   # distinct labels per cell
    assert not within_cell_independence_audit(cells, y_iid, K)["reject"]
    assert within_cell_independence_audit(cells, y_wor, K)["reject"]
