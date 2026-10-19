from horos.benchmark.replicate_comparison import run_replicate_comparison


def test_replicate_comparison_is_reproducible_and_never_trains():
    first = run_replicate_comparison(seed=91, repetitions=1)
    second = run_replicate_comparison(seed=91, repetitions=1)
    assert first == second
    assert first["n_runs"] == 36
    assert first["pythia_trained_models"] is False
    assert set(first["by_loss"]) == {"mae", "mse"}
    assert set(first["by_noise_family"]) == {"uniform", "spike_mixture"}
