from horos.benchmark.heavy_tail_comparison import run_heavy_tail_comparison


def test_heavy_tail_comparison_is_training_free_and_catoni_covers():
    result = run_heavy_tail_comparison(seed=51, repetitions=5)
    assert result["pythia_trained_models"] is False
    assert result["summary"]["catoni"]["valid"] == result["n_runs"]
