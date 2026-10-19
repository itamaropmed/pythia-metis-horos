from horos.benchmark.dependence_comparison import run_dependence_comparison


def test_comparison_is_training_free_and_finite_sample_rungs_cover():
    result = run_dependence_comparison(seed=44, repetitions=8)
    assert result["pythia_trained_models"] is False
    total = result["n_runs"]
    assert result["summary"]["empirical_bernstein_bound"]["valid"] == total
    assert result["summary"]["hoeffding_bound"]["valid"] == total
