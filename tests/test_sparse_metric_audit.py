from horos.benchmark.sparse_metric_audit import run_sparse_metric_audit


def test_sparse_metric_audit_is_deterministic_and_model_free():
    first = run_sparse_metric_audit(seed=7, repetitions=1, entity_counts=(100,))
    second = run_sparse_metric_audit(seed=7, repetitions=1, entity_counts=(100,))
    assert first == second
    assert first["n_generated_datasets"] == 1
    assert first["pythia_trained_models"] is False
    assert set(first["results"]) == {"mae", "mse"}
