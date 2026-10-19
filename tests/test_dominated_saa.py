import numpy as np
import pytest

from pythia.core.dominated_saa import (bentkus_lower_from_statistic, dominated_saa_lower_bound,
                                      truncated_cell_sum)


def _brute(y, loss, kappa, q=None, w=None):
    grid = np.linspace(y.min() - 50, y.max() + 50, 40001)
    grid = np.unique(np.concatenate([grid, y, y - kappa, y + kappa]))
    d = y[:, None] - grid[None, :]
    if loss == "mae":
        return float(np.minimum(np.abs(d), kappa).sum(0).min())
    if loss == "pinball":
        return float(np.minimum(np.where(d >= 0, q * d, (q - 1) * d), kappa).sum(0).min())
    return float((np.abs(d) > w).sum(0).min())


@pytest.mark.parametrize("loss", ["mae", "pinball", "window_miss"])
def test_truncated_minimum_is_exact_lower_than_any_grid(loss):
    r = np.random.default_rng(3)
    for _ in range(40):
        y = r.standard_t(2, size=r.integers(1, 30)) * 2
        kw = dict(kappa=float(r.uniform(.2, 4)))
        q = float(r.uniform(.1, .9)); w = float(r.uniform(.1, 2))
        exact = truncated_cell_sum(y, loss=loss, quantile=q, window=w, **kw)
        assert exact <= _brute(y, loss, kw["kappa"], q, w) + 1e-9
        assert exact >= 0


def test_bentkus_inversion_monotone_and_below_statistic():
    vals = [bentkus_lower_from_statistic(s, 1000, 2.0, .05) for s in (.1, .5, 1.0)]
    assert vals[0] < vals[1] < vals[2]
    assert all(v <= s for v, s in zip(vals, (.1, .5, 1.0)))
    assert bentkus_lower_from_statistic(0.0, 10, 1.0, .05) == 0.0


def test_refuses_without_contracts():
    y = np.arange(20.); lab = np.zeros(20); cl = np.arange(20)
    res = dominated_saa_lower_bound(lab, y, cl, loss="mae", caps=(1.,), representation_provenance="x",
                                    independent_clusters_attested=False, outcome_blind_labels_attested=True)
    assert res.status == "refused" and res.bound is None


def test_coverage_with_dependent_clusters_and_singletons():
    r = np.random.default_rng(11)
    misses = 0
    for rep in range(60):
        G = 80
        sizes = r.integers(1, 5, G)
        cl = np.repeat(np.arange(G), sizes)
        lab = r.integers(0, 6, len(cl))
        u = np.repeat(r.normal(size=G), sizes)
        y = 2 * lab + .5 * u + .5 * r.normal(size=len(cl))
        gold = np.sqrt(.5) * np.sqrt(2 / np.pi)          # MAE of N(0, .5)
        res = dominated_saa_lower_bound(lab, y, cl, loss="mae", caps=(.5, 1., 2.), alpha=.1,
                                        max_cases_per_cluster=3, representation_provenance="t",
                                        independent_clusters_attested=True, outcome_blind_labels_attested=True)
        misses += res.bound > gold
        assert res.diagnostics["model_training_performed"] is False
    assert misses <= 6


def test_retention_grid_is_bonferroni_charged_and_reported():
    r = np.random.default_rng(5)
    cl = np.repeat(np.arange(300), 3); lab = r.integers(0, 4, len(cl)); y = lab + r.normal(size=len(cl))
    kw = dict(loss="mae", caps=(1., 2.), representation_provenance="t", independent_clusters_attested=True,
              outcome_blind_labels_attested=True)
    res = dominated_saa_lower_bound(lab, y, cl, retention_grid=(1, 2, None), **kw)
    assert len(res.candidates) == 2 * 2 * 3 and abs(res.alpha_per_candidate - .05 / 12) < 1e-15
    assert res.diagnostics["selected_retention"] in (1, 2, None)
    with pytest.raises(ValueError):
        dominated_saa_lower_bound(lab, y, cl, retention_grid=(1,), max_cases_per_cluster=2, **kw)
