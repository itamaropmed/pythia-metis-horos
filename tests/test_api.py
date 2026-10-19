import numpy as np

from pythia import (
    ClaimTier, DependenceGuardSpec, FoldMatrix, GuardSpec, PairMatrix,
    RangeDependenceGuardSpec, ReplicateGuardSpec, Representation, TaskCard, certify,
)


def _folds() -> FoldMatrix:
    y = np.array([0.0, 1.0, 0.0, 1.0])
    p = np.array([[0.1, 0.8, 0.2, 0.9], [0.2, 0.7, 0.1, 0.8]])
    return FoldMatrix(y, p, ("a", "b"), ("x", "y"), np.array([0, 0, 1, 1]))


def test_unsupported_loss_returns_structured_refusal():
    result = certify(TaskCard("y", "auc"), _folds(), Representation("demo"))
    assert result.status == "refused"
    assert result.claim_tier is ClaimTier.REFUSED
    assert result.bound is None and result.refusal["reason"]


def test_bregman_claim_is_conditional():
    result = certify(TaskCard("y", "mse"), _folds(), Representation("demo"))
    assert result.status == "complete"
    assert result.claim_tier is ClaimTier.CONDITIONAL
    assert 0 <= result.bound <= result.estimator <= result.achieved_loss


def test_mae_without_guard_refuses():
    result = certify(TaskCard("y", "mae"), _folds(), Representation("demo"))
    assert result.claim_tier is ClaimTier.REFUSED
    assert "guard" in result.refusal["reason"].lower()


def test_mae_certificate_always_exposes_unselected_shared_bias_curve():
    rng = np.random.default_rng(12)
    y = rng.normal(size=2000)
    p = np.vstack([rng.normal(scale=.2, size=2000), rng.normal(scale=.3, size=2000)])
    folds = FoldMatrix(y, p, ("a", "b"), ("x", "y"), np.arange(2000) % 2)
    guard = GuardSpec(
        version="test", kind="test", bound_grid=np.ones((3, 3)),
        estimator_grid=np.ones((3, 3)), rho_domain=(0.0, 2.0),
        tau_domain=(0.0, 2.0), calibration_population="test",
        source_hash="test",
    )
    result = certify(TaskCard("y", "mae"), folds, Representation("demo"), guard)
    curve = result.diagnostics["shared_bias_sensitivity"]
    assert curve["curve"]["calibrated_2d"][0]["common_bias_mad_cap"] == 0.0
    assert "no cap selected" in result.diagnostics["shared_bias_cap_source"]
    assert any("shared" in warning.lower() for warning in result.warnings)


def _prospective_folds() -> FoldMatrix:
    rng = np.random.default_rng(19)
    y = rng.normal(size=800)
    p = np.vstack([y + rng.normal(scale=.5, size=y.size),
                   y + rng.normal(scale=.7, size=y.size)])
    return FoldMatrix(
        y, p, ("a", "b"), ("x", "y"), np.zeros(y.size, dtype=int),
        groups=np.arange(y.size),
        meta={"evaluation_role": "prospective",
              "predictions_frozen_before_evaluation": True},
    )


def _dependence_guard(**changes) -> DependenceGuardSpec:
    values = {
        "version": "test-v1", "lambda_max": .25,
        "risk_variance_bound": 1.0, "disagreement_variance_bound": 1.0,
        "calibration_population": "predeclared synthetic family",
        "source_hash": "sha256:test",
    }
    values.update(changes)
    return DependenceGuardSpec(**values)


def test_dependence_guard_accepts_only_frozen_independent_evaluation():
    result = certify(
        TaskCard("y", "mse"), _prospective_folds(), Representation("demo"),
        _dependence_guard(),
    )
    assert result.status == "complete"
    assert result.claim_tier is ClaimTier.CONDITIONAL
    assert 0 <= result.bound <= result.estimator <= result.achieved_loss
    assert result.provenance["dependence_guard_version"] == "test-v1"


def test_dependence_guard_refuses_oof_or_unfrozen_predictions():
    folds = _prospective_folds()
    folds = FoldMatrix(
        folds.y, folds.P_oof, folds.names, folds.families, folds.fold_id,
        groups=folds.groups,
        meta={"evaluation_role": "oof", "predictions_frozen_before_evaluation": False},
    )
    result = certify(TaskCard("y", "mse"), folds, Representation("demo"), _dependence_guard())
    assert result.claim_tier is ClaimTier.REFUSED
    assert "overlapping OOF" in result.refusal["reason"]


def test_dependence_guard_refuses_wrong_loss_and_invalid_guard():
    wrong_loss = certify(
        TaskCard("y", "mae"), _prospective_folds(), Representation("demo"),
        _dependence_guard(),
    )
    invalid = certify(
        TaskCard("y", "mse"), _prospective_folds(), Representation("demo"),
        _dependence_guard(lambda_max=1.0),
    )
    assert wrong_loss.claim_tier is ClaimTier.REFUSED
    assert "squared loss" in wrong_loss.refusal["reason"]
    assert invalid.claim_tier is ClaimTier.REFUSED
    assert "lambda_max" in invalid.refusal["reason"]


def test_range_dependence_guard_uses_variance_adaptive_bounded_rung():
    guard = RangeDependenceGuardSpec(
        version="range-v1", lambda_max=.25, absolute_error_bound=3.0,
        prediction_difference_bound=3.0,
        calibration_population="predeclared endpoint policy",
        source_hash="sha256:range-test",
    )
    result = certify(
        TaskCard("y", "mse"), _prospective_folds(), Representation("demo"), guard,
    )
    assert result.status == "complete"
    assert result.claim_tier is ClaimTier.CONDITIONAL
    assert "empirical_bernstein_bounded_range" in result.rungs
    assert 0 <= result.bound <= result.estimator <= result.achieved_loss


def test_range_dependence_guard_refuses_observed_cap_violation():
    guard = RangeDependenceGuardSpec(
        version="range-v1", lambda_max=.25, absolute_error_bound=.01,
        prediction_difference_bound=3.0,
        calibration_population="predeclared endpoint policy",
        source_hash="sha256:range-test",
    )
    result = certify(
        TaskCard("y", "mse"), _prospective_folds(), Representation("demo"), guard,
    )
    assert result.claim_tier is ClaimTier.REFUSED
    assert "exceeds" in result.refusal["reason"]


def _pair_matrix(**meta_changes) -> PairMatrix:
    rng = np.random.default_rng(31)
    n = 2000
    x = rng.uniform(size=n)
    d = rng.uniform(0, .01, n)
    meta = {
        "pairing_outcome_blind": True, "iid_randomized_blocks": True,
        "conditionally_independent_entities": True,
        "independent_blocks": True,
    }
    meta.update(meta_changes)
    return PairMatrix(
        2 * x + rng.uniform(-.5, .5, n),
        2 * (x + d) + rng.uniform(-.5, .5, n), d,
        np.arange(n), np.arange(n, 2 * n), np.arange(n), meta,
    )


def _replicate_guard() -> ReplicateGuardSpec:
    return ReplicateGuardSpec(
        version="rep-v1", lipschitz_bound=2.0, distance_bound=.01,
        outcome_difference_bound=1.02, calibration_population="synthetic metric",
        source_hash="sha256:rep-test",
    )


def _audited_representation() -> Representation:
    return Representation(
        "metric-v1", source_hash="sha256:metric-test",
        provenance={"available_at_prediction_time": True, "outcome_blind": True},
    )


def test_public_api_accepts_audited_replicate_mae_without_predictions():
    result = certify(
        TaskCard("y", "mae"), _pair_matrix(), _audited_representation(),
        _replicate_guard(),
    )
    assert result.status == "complete"
    assert result.claim_tier is ClaimTier.CONDITIONAL
    assert 0 < result.bound <= result.estimator
    assert result.achieved_loss is None
    assert "lower-bound center" in result.diagnostics["estimate_role"]


def test_public_api_exposes_selection_safe_clipped_replicate_rung():
    guard = ReplicateGuardSpec(
        version="rep-clips-v1", lipschitz_bound=2.0, distance_bound=.01,
        outcome_difference_bound=1.02, calibration_population="synthetic metric",
        source_hash="sha256:predeclared-clips",
        concentration="conditional_empirical_bernstein",
        outcome_term_clip_grid=(.25, .4, .51),
    )
    result = certify(
        TaskCard("y", "mae"), _pair_matrix(), _audited_representation(), guard,
    )
    assert result.status == "complete"
    assert result.diagnostics["clip_selection_simultaneous"] is True
    assert result.diagnostics["selected_clip"] in guard.outcome_term_clip_grid
    assert "clipped_envelope" in next(iter(result.rungs))


def test_public_api_accepts_audited_w30_miss_without_raw_outcome_cap():
    guard = ReplicateGuardSpec(
        version="w30-v1", lipschitz_bound=0.0, distance_bound=.01,
        outcome_difference_bound=None, calibration_population="synthetic metric",
        source_hash="sha256:w30-guard",
        concentration="conditional_empirical_bernstein",
    )
    result = certify(
        TaskCard("y", "w30_miss", operational_unit="miss fraction"),
        _pair_matrix(), _audited_representation(), guard,
    )
    assert result.status == "complete"
    assert 0 <= result.bound <= .5
    assert result.diagnostics["attainable_within_window_success_upper_percent"] == 100 * (1 - result.bound)


def test_public_api_accepts_audited_pinball_quantile_task():
    guard = ReplicateGuardSpec(
        version="pinball-v1", lipschitz_bound=2.0, distance_bound=.01,
        outcome_difference_bound=1.02, calibration_population="synthetic metric",
        source_hash="sha256:pinball-guard",
        concentration="conditional_empirical_bernstein",
    )
    result = certify(
        TaskCard("y", "pinball", quantile=.9),
        _pair_matrix(), _audited_representation(), guard,
    )
    assert result.status == "complete"
    assert result.diagnostics["quantile"] == .9


def test_public_api_exposes_conditional_bentkus_for_window_and_quantile():
    for task, cap in (
        (TaskCard("y", "pinball", quantile=.1), 1.02),
        (TaskCard("y", "window_miss", window=.2), None),
    ):
        guard = ReplicateGuardSpec(
            version="bentkus-v1", lipschitz_bound=2., distance_bound=.01,
            outcome_difference_bound=cap,
            calibration_population="synthetic metric",
            source_hash="sha256:predeclared-bentkus",
            concentration="conditional_bentkus",
        )
        result = certify(task, _pair_matrix(), _audited_representation(), guard)
        assert result.status == "complete"
        assert result.claim_tier is ClaimTier.CONDITIONAL
        assert result.diagnostics["selected_concentration"] == "conditional_bentkus"
        assert 0 <= result.bound <= result.estimator


def test_public_api_refuses_unaudited_replicate_design():
    result = certify(
        TaskCard("y", "mae"), _pair_matrix(pairing_outcome_blind=False),
        _audited_representation(), _replicate_guard(),
    )
    assert result.claim_tier is ClaimTier.REFUSED
    assert "outcome-blind" in result.refusal["reason"]


def test_public_api_refuses_missing_representation_timing_or_hash():
    for representation in (
        Representation("metric-v1"),
        Representation(
            "metric-v1", source_hash="sha256:metric-test",
            provenance={"outcome_blind": True},
        ),
    ):
        result = certify(TaskCard("y", "mae"), _pair_matrix(), representation, _replicate_guard())
        assert result.claim_tier is ClaimTier.REFUSED
        assert "representation audit" in result.refusal["reason"]


def test_public_api_refuses_explicit_target_or_post_cutoff_leakage():
    for provenance in (
        {"uses_target": True},
        {"available_at_prediction_time": False},
    ):
        result = certify(
            TaskCard("y", "mae"), _pair_matrix(),
            Representation("oracle", provenance=provenance), _replicate_guard(),
        )
        assert result.claim_tier is ClaimTier.REFUSED
        assert "target-derived" in result.refusal["reason"]


def test_public_api_conditional_replicate_requires_independent_entities():
    guard = ReplicateGuardSpec(
        version="rep-v1", lipschitz_bound=2.0, distance_bound=.01,
        outcome_difference_bound=1.02, calibration_population="synthetic metric",
        source_hash="sha256:rep-test", concentration="conditional_hoeffding",
    )
    refused = certify(
        TaskCard("y", "mae"),
        _pair_matrix(conditionally_independent_entities=False),
        _audited_representation(), guard,
    )
    accepted = certify(
        TaskCard("y", "mae"), _pair_matrix(), _audited_representation(), guard,
    )
    assert refused.claim_tier is ClaimTier.REFUSED
    assert "conditionally independent" in refused.refusal["reason"]
    assert accepted.status == "complete"
    assert "fixed-design" in accepted.intervals["type"]


def test_public_api_conditional_block_replicate_accepts_cluster_contract():
    guard = ReplicateGuardSpec(
        version="rep-v1", lipschitz_bound=2.0, distance_bound=.01,
        outcome_difference_bound=1.02, calibration_population="synthetic metric",
        source_hash="sha256:rep-test",
        concentration="conditional_block_empirical_bernstein",
    )
    result = certify(
        TaskCard("y", "mae"),
        _pair_matrix(conditionally_independent_entities=False),
        _audited_representation(), guard,
    )
    assert result.status == "complete"
    assert result.diagnostics["n_concentration_units"] == 2000


def test_public_api_auto_replicate_selects_safest_available_dependence_rung():
    guard = ReplicateGuardSpec(
        version="rep-v1", lipschitz_bound=2.0, distance_bound=.01,
        outcome_difference_bound=1.02, calibration_population="synthetic metric",
        source_hash="sha256:rep-test", concentration="auto",
    )
    pair_result = certify(
        TaskCard("y", "mae"), _pair_matrix(), _audited_representation(), guard,
    )
    block_result = certify(
        TaskCard("y", "mae"),
        _pair_matrix(conditionally_independent_entities=False),
        _audited_representation(), guard,
    )
    refused = certify(
        TaskCard("y", "mae"),
        _pair_matrix(
            conditionally_independent_entities=False, independent_blocks=False,
            iid_randomized_blocks=False,
        ),
        _audited_representation(), guard,
    )
    assert pair_result.diagnostics["selected_concentration"] == "conditional_empirical_bernstein"
    assert block_result.diagnostics["selected_concentration"] == "conditional_block_empirical_bernstein"
    assert refused.claim_tier is ClaimTier.REFUSED
    assert "neither" in refused.refusal["reason"]
