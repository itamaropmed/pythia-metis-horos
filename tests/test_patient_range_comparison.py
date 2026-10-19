from horos.benchmark.patient_range_comparison import run_patient_range_comparison


def test_patient_range_comparison_smoke_is_reproducible_and_valid():
    first = run_patient_range_comparison(seed=77, repetitions=2)
    second = run_patient_range_comparison(seed=77, repetitions=2)
    assert first == second
    assert first["n_runs"] == 18
    for method in (
        "patient_eb_sharp", "patient_eb_conservative", "patient_hoeffding",
        "catoni_oracle",
    ):
        assert first["summary"][method]["valid"] == 18
    assert first["summary"]["empirical_bernstein_gain"]["strict_wins"] > 0
