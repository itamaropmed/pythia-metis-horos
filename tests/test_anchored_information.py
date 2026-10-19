import numpy as np
from pythia.core.anchored_information import anchored_information, _greedy_pairs


def test_no_false_information_when_text_is_irrelevant():
    pos = 0
    for rep in range(10):
        r = np.random.default_rng(rep); n = 2000; cells = r.integers(0, 8, n)
        p = .2 + .08 * cells; y = (r.random(n) < p).astype(int); E = r.normal(size=(n, 5))
        pos += anchored_information(cells, E, y, 2, .05, seed=rep)["dI_lower"] > 0
    assert pos == 0


def test_detects_strong_text_information():
    r = np.random.default_rng(1); n = 6000; cells = r.integers(0, 4, n); E = r.random((n, 2))
    p = np.where(E[:, 0] > .5, .9, .1); y = (r.random(n) < p).astype(int)
    assert anchored_information(cells, E, y, 2, .05)["dI_lower"] > .1


def test_pairs_with_ties_are_disjoint():
    E = np.zeros((50, 2))
    P = _greedy_pairs(E); flat = [v for pr in P for v in pr]
    assert len(set(flat)) == len(flat) == 50
