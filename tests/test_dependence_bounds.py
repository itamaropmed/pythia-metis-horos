import numpy as np
import pytest

from pythia.core.dependence_bounds import (
    _sharp_rectangle_lower_bound,
    bounded_cluster_pairwise_bound,
    bounded_cluster_sensitivity_curve,
    bounded_sample_pairwise_bound,
    catoni_cluster_pairwise_bound,
    empirical_bernstein_patient_pairwise_bound,
    median_of_means_cluster_pairwise_bound,
    sharp_pairwise_population_bound,
)


def _population_moments(r_star, variances, covariance):
    variances = np.asarray(variances, float)
    covariance = np.asarray(covariance, float)
    risks = r_star + variances
    disagreement = variances[:, None] + variances[None, :] - 2 * covariance
    np.fill_diagonal(disagreement, 0.0)
    return risks, disagreement


def test_population_bound_is_exact_when_active_correlation_reaches_cap():
    r_star = 2.0
    variances = np.array([1.0, 4.0])
    lam = 0.5
    covariance = np.diag(variances)
    covariance[0, 1] = covariance[1, 0] = lam * np.sqrt(np.prod(variances))
    risks, disagreement = _population_moments(r_star, variances, covariance)
    result = sharp_pairwise_population_bound(risks, disagreement, lam)
    assert result["bound"] == pytest.approx(r_star)


def test_sharp_rectangle_analytic_minimum_matches_dense_search():
    li, ui, lj, uj, d, lam = .8, 1.4, 1.0, 1.7, .9, .55
    analytic = _sharp_rectangle_lower_bound(li, ui, lj, uj, d, lam)
    c = 1.0 - lam * lam
    lo = max(li - uj, -d / np.sqrt(c))
    hi = min(ui - lj, d / np.sqrt(c))
    deltas = np.linspace(lo, hi, 100_001)
    sums = np.maximum(2 * li - deltas, 2 * lj + deltas)
    feasible = sums <= np.minimum(2 * ui - deltas, 2 * uj + deltas)
    excess = (d + lam * np.sqrt(np.maximum(d * d - c * deltas ** 2, 0.0))) / c
    dense = max(float(np.min(.5 * (sums[feasible] - excess[feasible]))), 0.0)
    assert analytic == pytest.approx(dense, abs=2e-6)


def test_population_bound_is_monotone_in_allowed_dependence():
    variances = np.array([1.0, 2.0, 3.0])
    covariance = np.diag(variances)
    risks, disagreement = _population_moments(0.7, variances, covariance)
    values = [sharp_pairwise_population_bound(risks, disagreement, lam)["bound"] for lam in (0, .2, .5, .8)]
    assert np.all(np.diff(values) <= 1e-12)
    assert values[0] == pytest.approx(0.7)


def test_infeasible_population_moments_refuse():
    result = sharp_pairwise_population_bound(
        np.array([1.0, 10.0]), np.array([[0.0, 0.1], [0.1, 0.0]]), 0.0
    )
    assert result["status"] == "refused"


def test_bounded_sample_rung_refuses_broken_error_cap():
    y = np.array([0.0, 1.0, 2.0])
    p = np.vstack([y, y + 2.0])
    result = bounded_sample_pairwise_bound(y, p, 0.2, 1.0)
    assert result["status"] == "refused"


def test_bounded_sample_rung_obeys_basic_range():
    rng = np.random.default_rng(5)
    n = 5000
    signal = rng.uniform(-1, 1, n)
    noise = rng.uniform(-0.5, 0.5, n)
    y = signal + noise
    p = np.vstack([
        signal + rng.uniform(-0.2, 0.2, n),
        signal + rng.uniform(-0.3, 0.3, n),
    ])
    result = bounded_sample_pairwise_bound(y, p, 0.0, 0.8)
    assert result["status"] == "complete"
    assert 0 <= result["bound"] <= np.mean(noise ** 2)


def test_empirical_bernstein_is_available_alongside_hoeffding():
    rng = np.random.default_rng(11)
    y = rng.uniform(-1, 1, 1000)
    p = np.vstack([y + rng.uniform(-.1, .1, 1000), y + rng.uniform(-.2, .2, 1000)])
    eb = bounded_sample_pairwise_bound(y, p, 0.0, 0.3)
    hf = bounded_sample_pairwise_bound(y, p, 0.0, 0.3, concentration="hoeffding")
    assert eb["concentration"] == "empirical_bernstein"
    assert hf["concentration"] == "hoeffding"


def test_cluster_bound_reports_effective_sample_size_and_is_valid():
    rng = np.random.default_rng(19)
    groups = np.repeat(np.arange(2000), rng.integers(1, 4, 2000))
    n = groups.size
    signal = rng.uniform(-1, 1, n)
    noise = rng.uniform(-.5, .5, n)
    y = signal + noise
    p = np.vstack([signal + rng.uniform(-.1, .1, n), signal + rng.uniform(-.2, .2, n)])
    result = bounded_cluster_pairwise_bound(
        y, p, groups, 0.0, 0.7, prediction_difference_bound=0.4,
    )
    assert result["status"] == "complete"
    assert result["n_clusters"] == 2000
    assert 0 < result["effective_clusters"] <= 2000
    assert 0 <= result["bound"] <= np.mean(noise ** 2)


def test_patient_empirical_bernstein_is_valid_and_variance_adaptive():
    rng = np.random.default_rng(191)
    groups = np.repeat(np.arange(5000), rng.integers(1, 4, 5000))
    signal = rng.uniform(-1, 1, groups.size)
    noise = rng.uniform(-.4, .4, groups.size)
    y = signal + noise
    p = np.vstack([
        signal + rng.uniform(-.1, .1, groups.size),
        signal + rng.uniform(-.2, .2, groups.size),
    ])
    result = empirical_bernstein_patient_pairwise_bound(
        y, p, groups, 0.0, absolute_error_bound=.7,
        prediction_difference_bound=.4,
    )
    patient_noise = np.asarray([
        np.mean(noise[groups == group] ** 2) for group in np.unique(groups)
    ])
    assert result["status"] == "complete"
    assert result["n_patients"] == 5000
    assert result["concentration"] == "patient-level empirical bernstein"
    assert 0 <= result["bound"] <= np.mean(patient_noise)
    active = next(row for row in result["pairs"] if tuple(row["pair"]) == tuple(result["active_pair"]))
    assert active["bound"] >= active["conservative_bound"]
    assert result["moment_geometry"].startswith("sharp pairwise")


def test_patient_empirical_bernstein_refuses_broken_ranges():
    y = np.arange(10, dtype=float)
    p = np.vstack([y, y + 2.0])
    result = empirical_bernstein_patient_pairwise_bound(
        y, p, np.arange(10), .2, absolute_error_bound=1.0,
        prediction_difference_bound=3.0,
    )
    assert result["status"] == "refused"


def test_patient_empirical_bernstein_tightens_patient_hoeffding():
    rng = np.random.default_rng(192)
    groups = np.arange(10000)
    y = rng.uniform(-.5, .5, groups.size)
    p = np.vstack([rng.uniform(-.1, .1, groups.size), rng.uniform(-.2, .2, groups.size)])
    kwargs = dict(y=y, predictions=p, groups=groups, lambda_max=0.0,
                  absolute_error_bound=.7, prediction_difference_bound=.4)
    eb = empirical_bernstein_patient_pairwise_bound(**kwargs)
    hf = empirical_bernstein_patient_pairwise_bound(**kwargs, concentration="hoeffding")
    assert eb["bound"] >= hf["bound"]


def test_cluster_sensitivity_curve_is_simultaneous_and_monotone():
    rng = np.random.default_rng(23)
    groups = np.arange(5000)
    signal = rng.uniform(-1, 1, groups.size)
    y = signal + rng.uniform(-.5, .5, groups.size)
    p = np.vstack([signal + rng.uniform(-.1, .1, groups.size), signal + rng.uniform(-.2, .2, groups.size)])
    result = bounded_cluster_sensitivity_curve(
        y, p, groups, np.array([0.0, .2, .5, .8]), .7,
        prediction_difference_bound=.4,
    )
    bounds = np.asarray([row["bound"] for row in result["curve"]])
    assert result["simultaneous_over_lambda_grid"] is True
    assert np.all(np.diff(bounds) <= 1e-12)


def test_median_of_means_cluster_bound_handles_heavy_tails():
    rng = np.random.default_rng(29)
    groups = np.repeat(np.arange(4000), 2)
    signal = rng.normal(size=groups.size)
    noise = rng.standard_t(df=5, size=groups.size) * .3
    common = rng.normal(scale=.05, size=groups.size)
    errors = np.vstack([common + rng.normal(scale=s, size=groups.size) for s in (.1, .2)])
    y = signal + noise
    p = signal[None, :] + errors
    # Deliberately loose oracle caps; the test checks theorem plumbing rather
    # than estimating these nuisance constants from the same sample.
    result = median_of_means_cluster_pairwise_bound(
        y, p, groups, 0.3, risk_variance_bound=5.0,
        disagreement_variance_bound=1.0, seed=3,
    )
    assert result["status"] == "complete"
    assert result["estimand"].startswith("equal-patient")
    assert 0 <= result["bound"] <= np.mean(noise ** 2)


def test_catoni_cluster_bound_handles_heavy_tails_more_tightly_than_mom():
    rng = np.random.default_rng(31)
    groups = np.arange(8000)
    signal = rng.normal(size=groups.size)
    noise = .3 * rng.standard_t(5, size=groups.size)
    common = rng.normal(scale=.1, size=groups.size)
    errors = np.vstack([common + rng.normal(scale=s, size=groups.size) for s in (.1, .2)])
    y = signal + noise
    p = signal[None, :] + errors
    kwargs = dict(
        y=y, predictions=p, groups=groups, lambda_max=.6,
        risk_variance_bound=1.0, disagreement_variance_bound=.1,
    )
    catoni = catoni_cluster_pairwise_bound(**kwargs)
    mom = median_of_means_cluster_pairwise_bound(**kwargs, seed=2)
    assert catoni["status"] == "complete"
    assert 0 <= catoni["bound"] <= np.mean(noise ** 2)
    assert catoni["bound"] >= mom["bound"]
