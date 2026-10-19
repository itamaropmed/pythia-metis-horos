from horos.benchmark.multiloss_comparison import run_multiloss_comparison


def test_numeric_and_note_multiloss_holdout_reproduces_wins_and_losses():
    result = run_multiloss_comparison(
        seed=20261006, repetitions=20, n_entities=512,
    )
    assert result["pythia_trained_predictive_model"] is False
    assert result["external_estimators_cross_fitted"] is True
    assert result["confidence_per_draw_simultaneous_across_losses"] == .95
    assert result["summary"]["n_independent_draws"] == 200
    assert result["summary"]["combined"]["valid"] == 1000
    assert result["summary"]["routed_bentkus"]["valid"] == 1000
    assert set(result["loss_registry"]) == {
        "q10_pinball", "q90_pinball", "w15_miss", "w30_miss", "w60_miss",
    }
    assert result["summary"]["n_zero_true_floor_results"] == 200
    assert result["by_loss"]["w60_miss"]["combined"]["median_usable_tightness"] is None
    assert result["by_loss"]["w60_miss"]["combined"]["valid"] == 200
    assert result["summary"]["routed_joint_draw_valid"] == 200
    assert result["summary"]["routed_bentkus"]["median_usable_tightness"] > result[
        "summary"
    ]["combined"]["median_usable_tightness"]
    assert result["by_loss"]["w30_miss"]["routed_bentkus"][
        "median_usable_tightness"
    ] > result["by_loss"]["w30_miss"]["conditional_bentkus"][
        "median_usable_tightness"
    ]
    # Preserve negative controls: Pythia is not yet the best point estimator.
    errors = result["summary"]["median_absolute_point_error"]
    assert errors["combined_center"] > errors["knn"]
    assert errors["combined_center"] > errors["kernel"]
