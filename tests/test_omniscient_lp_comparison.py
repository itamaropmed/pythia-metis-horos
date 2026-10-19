from horos.benchmark.omniscient_lp_comparison import run_omniscient_lp_comparison


def test_lp_comparison_is_deterministic_and_keeps_claim_tiers_separate():
    first = run_omniscient_lp_comparison(seed=4, repetitions=1, n_entities=20)
    second = run_omniscient_lp_comparison(seed=4, repetitions=1, n_entities=20)
    assert first == second
    assert first["summary"]["total"] == 10
    assert first["lp_difference_from_paper"]
