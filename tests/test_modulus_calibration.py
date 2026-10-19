import numpy as np
import pytest

from pythia import (
    calibrate_repeated_environment_discrepancy,
    calibrate_repeated_environment_discrepancy_v1,
    calibrated_barycentric_lower_bound,
    repeated_environment_barycentric_costs,
    support_aware_barycentric_cost_ranges,
)


def _example(seed=0, environments=400):
    rng = np.random.default_rng(seed)
    u = rng.random((environments, 1))
    location = np.array([-1.0, -.2, .4, 1.1, 2.0, 2.3, 3.1, 3.8])
    scale = np.array([.6, 1.0, .8, 1.2, .7, 1.1, .9, 1.3])
    y = location + scale * (u - .5)
    blocks = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    features = np.column_stack([location, scale])
    return y, blocks, features


def test_repeated_environment_costs_respect_ranges():
    y, blocks, _ = _example()
    absolute, squared, arange, srange = repeated_environment_barycentric_costs(
        y, blocks, clip=(-5, 5))
    assert np.all((0 <= absolute) & (absolute <= arange))
    assert np.all((0 <= squared) & (squared <= srange))
    assert arange == 5
    assert srange == 25


def test_calibration_upper_bounds_known_common_uniform_coupling():
    y, blocks, features = _example(environments=4000)
    # The grid average is a deterministic high-accuracy population reference.
    u = (np.arange(200_000)[:, None] + .5) / 200_000
    location = features[:, 0]
    scale = features[:, 1]
    population = location + scale * (u - .5)
    true_abs, true_sq, _, _ = repeated_environment_barycentric_costs(
        population, blocks, clip=(-5, 5))
    result = calibrate_repeated_environment_discrepancy(
        y, blocks, features, clip=(-5, 5), alpha_family=.05,
        marginal_contract="target_unit_conditional",
        coupling_provenance="common-uniform target coupling v1",
        metric_provenance="location-scale Euclidean v1",
        portfolio_frozen_before_replicates=True)
    assert result.status == "conditional"
    assert result.absolute_discrepancy_cap >= np.mean(true_abs)
    assert result.squared_discrepancy_cap >= np.mean(true_sq)
    assert result.diagnostics["exact_target_replicates_available"] is True
    assert result.effective_absolute_modulus_cap > 0


def test_calibration_refuses_missing_contract_or_posthoc_portfolio():
    y, blocks, features = _example()
    common = dict(
        replicates=y, blocks=blocks, features=features, clip=(-5, 5),
        coupling_provenance="coupling", metric_provenance="metric")
    refused = calibrate_repeated_environment_discrepancy(
        **common, marginal_contract=None,
        portfolio_frozen_before_replicates=True)
    assert refused.status == "refused"
    refused = calibrate_repeated_environment_discrepancy(
        **common, marginal_contract="target_unit_conditional",
        portfolio_frozen_before_replicates=False)
    assert refused.status == "refused"


def test_shift_requires_provenance_and_integer_blocks():
    y, blocks, features = _example()
    with pytest.raises(ValueError, match="shift provenance"):
        calibrate_repeated_environment_discrepancy(
            y, blocks, features, clip=(-5, 5), shift=np.zeros(y.shape[1]),
            marginal_contract="target_unit_conditional",
            coupling_provenance="coupling", metric_provenance="metric",
            portfolio_frozen_before_replicates=True)
    with pytest.raises(ValueError, match="integer blocks"):
        repeated_environment_barycentric_costs(
            y, blocks.astype(float) + .1, clip=(-5, 5))


def test_frozen_v1_router_uses_loss_specific_methods():
    y, blocks, features = _example(environments=1000)
    result = calibrate_repeated_environment_discrepancy_v1(
        y, blocks, features, clip=(-5, 5),
        marginal_contract="target_unit_conditional",
        coupling_provenance="coupling", metric_provenance="metric",
        portfolio_frozen_before_replicates=True)
    assert [row["method"] for row in result.candidates["absolute"]] == ["dKelly"]
    assert [row["method"] for row in result.candidates["squared"]] == ["Bentkus"]
    assert result.alpha_per_endpoint == pytest.approx(.025)


def test_support_aware_ranges_are_exact_and_tighter():
    supports = np.array([[0., 1.], [0., 1.], [2., 3.], [2., 3.]])
    blocks = np.array([[0, 1], [2, 3]])
    absolute, squared = support_aware_barycentric_cost_ranges(
        supports, blocks, clip=(0, 3))
    assert absolute == pytest.approx(.5)
    assert squared == pytest.approx(.25)
    assert absolute < 1.5
    assert squared < 2.25


def test_support_aware_calibration_checks_hard_supports():
    y, blocks, features = _example(environments=1000)
    supports = np.column_stack([
        features[:, 0] - features[:, 1]/2,
        features[:, 0] + features[:, 1]/2])
    result = calibrate_repeated_environment_discrepancy_v1(
        y, blocks, features, clip=(-5, 5), unit_supports=supports,
        support_provenance="declared post-shift supports v1",
        marginal_contract="target_unit_conditional",
        coupling_provenance="coupling", metric_provenance="metric",
        portfolio_frozen_before_replicates=True)
    assert result.diagnostics["support_aware_ranges_used"] is True
    broken = supports.copy()
    broken[0, 1] -= .5
    with pytest.raises(ValueError, match="violate"):
        calibrate_repeated_environment_discrepancy_v1(
            y, blocks, features, clip=(-5, 5), unit_supports=broken,
            support_provenance="bad support",
            marginal_contract="target_unit_conditional",
            coupling_provenance="coupling", metric_provenance="metric",
            portfolio_frozen_before_replicates=True)


def test_calibrated_lower_bound_composes_events_and_checks_block_identity():
    y, blocks, features = _example(environments=1000)
    calibration = calibrate_repeated_environment_discrepancy_v1(
        y, blocks, features, clip=(-5, 5), alpha_family=.025,
        marginal_contract="target_unit_conditional",
        coupling_provenance="coupling", metric_provenance="metric",
        portfolio_frozen_before_replicates=True)
    block_hash = calibration.diagnostics["block_design_hash"]
    result = calibrated_barycentric_lower_bound(
        lower_score_mean=2, retained_fraction=.8, loss="Q90",
        calibration=calibration, score_failure_probability=.025,
        score_event_id="score-e1", calibration_event_id="cal-e1",
        score_block_design_hash=block_hash, score_provenance="score-v1")
    assert result.status == "conditional"
    assert result.alpha_family == pytest.approx(.05)
    assert result.penalty_cap == pytest.approx(
        .5*calibration.absolute_discrepancy_cap)
    assert result.to_dict()["bound"] == result.bound
    refused = calibrated_barycentric_lower_bound(
        lower_score_mean=2, retained_fraction=.8, loss="MAE",
        calibration=calibration, score_failure_probability=.025,
        score_event_id="score-e1", calibration_event_id="cal-e1",
        score_block_design_hash="wrong", score_provenance="score-v1")
    assert refused.status == "refused"
