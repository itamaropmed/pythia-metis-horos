from horos.benchmark.ood_mae_comparison import run_ood_mae_comparison


def test_ood_benchmark_smoke_is_deterministic_and_no_pythia_model_fit():
    first = run_ood_mae_comparison(seed=3, repetitions=1, n_entities=32)
    second = run_ood_mae_comparison(seed=3, repetitions=1, n_entities=32)
    assert first == second
    assert first["summary"]["total_independent_draws"] == 10
    assert first["summary"]["median_analytic_bayes_mae"] > 0
    assert first["summary"]["grouped_95"]["median_bound"] >= 0
    assert len(first["by_design"]) == 10
    assert first["pythia_trained_predictive_model"] is False
