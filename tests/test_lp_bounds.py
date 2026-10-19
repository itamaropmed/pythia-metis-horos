import numpy as np
from pythia.core.lp_bounds import design_statistics, variogram_lp, relaxation_path, moment_boxes, moment_lp_01


def _data(rep):
    r = np.random.default_rng(rep); n = 4000; cell = r.integers(0, 4, n); E = r.random((n, 2))
    p = 1 / (1 + np.exp(-(3 * (E[:, 0] + E[:, 1] - 1) + .3 * (cell - 1.5)))); y = (r.random(n) < p).astype(int)
    return cell, E, y, p


def test_lp_floor_valid_and_monotone_in_relaxation():
    cell, E, y, p = _data(0); st = design_statistics(cell, E, y, 2, 3, .05, 0)
    G = np.mean(2 * p[st["keep"]] * (1 - p[st["keep"]]))
    assert variogram_lp(st, .5, 0.0) <= G <= variogram_lp(st, .5, np.inf, "G", "max")
    path, tip = relaxation_path(st, .5, 10); vals = [v for _, v in path if v is not None]
    assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:]))


def test_moment_lp_brackets_table_floor():
    r = np.random.default_rng(1); groups = [np.arange(i * 200, (i + 1) * 200) for i in range(20)]
    pc = r.uniform(.05, .95, 20); y = np.concatenate([(r.random(200) < q).astype(int) for q in pc])
    lo, hi = moment_lp_01(moment_boxes(groups, y, 3)), moment_lp_01(moment_boxes(groups, y, 3), sense="max")
    assert lo <= np.minimum(pc, 1 - pc).mean() <= hi
