from horos.benchmark.replicate_adversarial import run_replicate_adversarial


def test_adversarial_replicate_suite_is_deterministic_and_model_free():
    first = run_replicate_adversarial(seed=3, repetitions=1)
    second = run_replicate_adversarial(seed=3, repetitions=1)
    assert first == second
    assert first["n_runs"] == 60
    assert first["pythia_trained_models"] is False
    assert len(first["by_design"]) == 10
