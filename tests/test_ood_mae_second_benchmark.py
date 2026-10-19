"""Deterministic known-risk regression, not an independent coverage proof."""
import pytest

from horos.benchmark.ood_mae_comparison import run_second_ood_mae_comparison


def test_second_ten_designs_reproduce_bound_coverage_and_metric_failure():
    result = run_second_ood_mae_comparison(
        seed=20260930, repetitions=20, n_entities=512,
    )
    summary = result["summary"]
    assert len(result["design_names"]) == 10
    assert summary["combined_95"]["valid"] == summary["combined_95"]["total"] == 200
    assert summary["combined_95"]["wilson_lower_95"] > .95
    assert summary["combined_95"]["median_usable_tightness"] == pytest.approx(
        .48620303343615323, abs=1e-10,
    )
    assert result["by_design"]["sparse12_uniform"]["combined_95"][
        "median_usable_tightness"
    ] == 0
    comparisons = summary["paired_comparisons"]
    assert comparisons["combined_bound_strict_wins_over_paired"] == 175
    assert comparisons["paired_bound_strict_wins_over_combined"] == 5
    assert summary["local_jackknife"]["median_absolute_error"] > summary[
        "knn_oof"
    ]["median_absolute_error"]
