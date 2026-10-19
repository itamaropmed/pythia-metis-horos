from horos.benchmark.exact_group_comparison import run_exact_group_comparison


def test_exact_group_known_risk_smoke_preserves_validity_and_no_training():
    result = run_exact_group_comparison(
        seed=20261008, repetitions=1, n_entities=64,
    )
    assert result["pythia_trained_predictive_model"] is False
    assert result["selection_between_witnesses_adjusted"] is False
    assert result["summary"]["total"] == 50
    assert result["summary"]["exact"]["valid"] == 50
    assert result["summary"]["exact_vs_radial"]["strict_losses"] == 0
