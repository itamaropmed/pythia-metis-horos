from horos.benchmark.clipped_replicate_comparison import run_clipped_replicate_comparison


def test_clipped_comparison_is_reproducible_and_model_free():
    first = run_clipped_replicate_comparison(seed=9, repetitions=1, n_entities=100)
    second = run_clipped_replicate_comparison(seed=9, repetitions=1, n_entities=100)
    assert first == second
    assert first["summary"]["plain"]["total"] == 20
    assert first["pythia_trained_models"] is False
