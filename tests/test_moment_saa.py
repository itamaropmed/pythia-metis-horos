import numpy as np
import pytest

from pythia import (
    cantelli_saa_lower_bound,
    catoni_second_moment_saa_lower_bound,
    location_saa_score_variance_cap,
)


def test_cantelli_formula_and_zero_variance_limit():
    result = cantelli_saa_lower_bound(
        [2., 3., 4.], loss="MAE", score_variance_bounds=0., alpha=.05,
        cap_source="analytic deterministic law", cap_frozen_before_evaluation=True,
        outcome_scale="minutes")
    assert result.status == "conditional"
    assert result.lower == 3.
    assert result.radius == 0.
    assert result.diagnostics["original_scale"] is True
    assert result.diagnostics["clipping_used"] is False


def test_cantelli_is_monotone_in_variance_and_confidence():
    common = dict(scores=np.full(40, 3.), loss="MAE", cap_source="declared",
                  cap_frozen_before_evaluation=True)
    small = cantelli_saa_lower_bound(score_variance_bounds=.1, alpha=.05, **common)
    large = cantelli_saa_lower_bound(score_variance_bounds=1., alpha=.05, **common)
    more_confident = cantelli_saa_lower_bound(score_variance_bounds=.1, alpha=.01, **common)
    assert small.lower > large.lower
    assert more_confident.lower < small.lower


def test_same_evaluation_cap_and_bad_inputs_refuse():
    reused = cantelli_saa_lower_bound(
        [1., 2.], loss="MAE", score_variance_bounds=1., alpha=.05,
        cap_source="sample variance", cap_frozen_before_evaluation=False)
    assert reused.status == "refused"
    assert "not frozen" in reused.refusal["reason"]
    bad = cantelli_saa_lower_bound(
        [1., -1.], loss="MAE", score_variance_bounds=1., alpha=.05,
        cap_source="analytic", cap_frozen_before_evaluation=True)
    assert bad.status == "refused"


def test_location_moment_caps_cover_mae_pinball_and_mse():
    mae = location_saa_score_variance_cap(
        loss="MAE", block_size=4, first_absolute_moment=2., second_moment=5.)
    assert mae["variance_bound"] == pytest.approx(4.25)
    q90 = location_saa_score_variance_cap(
        loss="Q90", block_size=4, first_absolute_moment=2., second_moment=5.)
    assert q90["variance_bound"] == pytest.approx(.9 ** 2 * 4.25)
    mse = location_saa_score_variance_cap(
        loss="MSE", block_size=4, first_absolute_moment=2.,
        second_moment=5., fourth_moment=40.)
    assert mse["variance_bound"] == pytest.approx(28.75)
    with pytest.raises(ValueError, match="fourth moment"):
        location_saa_score_variance_cap(
            loss="MSE", block_size=4, first_absolute_moment=2., second_moment=5.)


def test_cantelli_covers_heterogeneous_nonidentical_means():
    # Xi are independent Gamma(shape_i, scale_i), so exact means and variances
    # are known and differ across coordinates.  This tests the theorem's actual
    # average-expectation estimand rather than an iid/common-mean shortcut.
    rng = np.random.default_rng(20261012)
    shapes = np.linspace(.8, 4., 60)
    scales = np.linspace(.4, 1.2, 60)
    variances = shapes * scales ** 2
    truth = float(np.mean(shapes * scales))
    violations = 0
    repetitions = 5000
    for _ in range(repetitions):
        values = rng.gamma(shapes, scales)
        result = cantelli_saa_lower_bound(
            values, loss="generic nonnegative score",
            score_variance_bounds=variances, alpha=.05,
            cap_source="analytic Gamma variances",
            cap_frozen_before_evaluation=True)
        violations += result.lower > truth + 1e-12
    # Cantelli guarantees <= 5%; leave statistical slack in this finite check.
    assert violations / repetitions < .06


def test_catoni_second_moment_endpoint_handles_zero_and_refuses_reuse():
    zero = catoni_second_moment_saa_lower_bound(
        [0., 0.], loss="MAE", score_second_moment_bounds=0., alpha=.05,
        cap_source="deterministic", cap_frozen_before_evaluation=True)
    assert zero.lower == 0.
    assert zero.diagnostics["common_mean_required"] is False
    contradiction = catoni_second_moment_saa_lower_bound(
        [0., 1.], loss="MAE", score_second_moment_bounds=0., alpha=.05,
        cap_source="wrong", cap_frozen_before_evaluation=True)
    assert contradiction.status == "refused"
    reused = catoni_second_moment_saa_lower_bound(
        [1., 2.], loss="MAE", score_second_moment_bounds=5., alpha=.05,
        cap_source="evaluation", cap_frozen_before_evaluation=False)
    assert reused.status == "refused"


def test_catoni_covers_heterogeneous_nonidentical_means():
    rng = np.random.default_rng(20261013)
    shapes = np.linspace(.8, 4., 60)
    scales = np.linspace(.4, 1.2, 60)
    means = shapes * scales
    second_moments = shapes * (shapes + 1.) * scales ** 2
    truth = float(np.mean(means))
    violations = 0
    repetitions = 5000
    for _ in range(repetitions):
        values = rng.gamma(shapes, scales)
        result = catoni_second_moment_saa_lower_bound(
            values, loss="generic nonnegative score",
            score_second_moment_bounds=second_moments, alpha=.05,
            cap_source="analytic Gamma second moments",
            cap_frozen_before_evaluation=True)
        violations += result.lower > truth + 1e-12
    assert violations / repetitions < .06
