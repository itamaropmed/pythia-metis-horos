from horos.benchmark.information_simulation import run_information_simulation


def test_known_risk_information_simulation_is_training_free_and_valid_at_true_cap():
    result = run_information_simulation(seed=9, n=12_000)
    assert result["pythia_trained_models"] is False
    assert result["summary"]["dependence_empirical_valid"] == result["n_representation_cases"]
    assert result["information_chain"] == ["intact_note", "erased_note", "tabular"]
