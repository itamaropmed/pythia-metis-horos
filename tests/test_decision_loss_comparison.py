from horos.benchmark.decision_loss_comparison import run_decision_loss_comparison


def test_decision_loss_comparison_is_deterministic_and_model_free():
    first = run_decision_loss_comparison(seed=9, repetitions=1, n_entities=100)
    second = run_decision_loss_comparison(seed=9, repetitions=1, n_entities=100)
    assert first == second
    assert first["summary"]["hoeffding"]["total"] == 20
    assert first["pythia_trained_models"] is False
