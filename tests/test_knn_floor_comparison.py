from horos.benchmark.knn_floor_comparison import run_knn_floor_comparison


def test_knn_floor_comparison_is_reproducible_and_keeps_training_out_of_pythia():
    first = run_knn_floor_comparison(seed=9, repetitions=1, n_entities=100, k=5)
    second = run_knn_floor_comparison(seed=9, repetitions=1, n_entities=100, k=5)
    assert first == second
    assert first["summary"]["runs"] == 20
    assert first["pythia_trained_models"] is False
    assert first["horos_knn_uses_outcomes"] is True
