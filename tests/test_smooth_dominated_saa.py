import math
import numpy as np
from pythia.core.smooth_dominated_saa import (kd_blocks, matching_blocks, smooth_dominated_lower_bound,
                                             smooth_penalty)


def test_blocks_partition_rows():
    X = np.random.default_rng(0).uniform(size=(64, 3))
    for B in (kd_blocks(X, 8), matching_blocks(X, 8)):
        assert B.shape == (8, 8) and len(np.unique(B)) == 64


def test_second_order_penalty_never_exceeds_first():
    X = np.random.default_rng(1).uniform(size=(128, 2)); B = kd_blocks(X, 16)
    kap = np.full(128, 1.6)
    assert smooth_penalty(X, B, 3., kap, 1., "best") <= smooth_penalty(X, B, 3., kap, 1., "first") + 1e-12


def test_valid_on_smooth_design_and_refuses_nothing_silently():
    r = np.random.default_rng(2); miss = 0
    for rep in range(25):
        X = r.uniform(size=(400, 1)); y = 2 * X[:, 0] + .5 * r.standard_normal(400)
        gold = .5 * math.sqrt(2 / math.pi)
        kap = np.full(400, 2 / math.sqrt(2 * math.pi) / .5)
        b, info = smooth_dominated_lower_bound(X, y, loss="mae", L=2., kappa_rows=kap, block_sizes=(4, 16, 64),
                                               caps=(1., 2.), alpha=.1, geometries=("kd",))
        miss += b > gold
        assert info is None or info["concentration"] in ("domination_bentkus", "block_dkelly")
    assert miss <= 5
