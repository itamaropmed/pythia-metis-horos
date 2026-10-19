import numpy as np
from pythia.core.dynamic_geometry import diffusion_coords
from horos.models.pythia_adapters import dynamic_certificate


def _data(rep, s):
    r = np.random.default_rng(rep); n = 3000; cell = r.integers(0, 6, n); E = r.normal(size=(n, 6))
    p = 1 / (1 + np.exp(-(.4 * (cell - 2.5) + s * (2 * (E[:, 0] > 0) - 1)))); y = (r.random(n) < p).astype(int)
    pbar = np.array([p[cell == c].mean() for c in cell]); dI = np.mean(2 * (p - pbar) ** 2)
    return cell, E, y, dI


def test_no_false_information_any_learner():
    for rep in range(4):
        cell, E, y, dI = _data(rep, 0.0)
        for L in ("static", "posterior", "nca"):
            assert dynamic_certificate(cell, E, y, 2, .05, B=3, learner=L, seed=rep)["dI_lower"] == 0.0


def test_valid_and_positive_with_signal():
    cell, E, y, dI = _data(1, 1.2)
    for L in ("static", "posterior"):
        v = dynamic_certificate(cell, E, y, 2, .05, B=3, learner=L, fracs=(.2, .4, .4))["dI_lower"]
        assert 0 < v <= dI


def test_diffusion_coords_shapes():
    E = np.random.default_rng(0).normal(size=(300, 5)); D = diffusion_coords(E, k=10, n_comp=8, ts=(1, 4))
    assert set(D) == {1, 4} and D[1].shape == (300, 8)
