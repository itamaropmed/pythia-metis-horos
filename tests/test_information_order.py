import pytest

from pythia.information_order import (
    RiskInterval,
    close_information_order,
    decompose_achieved_gain,
    information_value_interval,
)


def test_closure_propagates_bounds_in_the_valid_directions():
    result = close_information_order(
        {
            "rich": RiskInterval(2.0, 4.0, 8.0, "calibrated", simultaneous=True),
            "middle": RiskInterval(1.0, 5.0, 7.0, "calibrated", simultaneous=True),
            "poor": RiskInterval(0.5, 9.0, 10.0, "calibrated", simultaneous=True),
        },
        [("rich", "middle"), ("middle", "poor")],
    )
    assert result.status == "complete"
    assert result.intervals["poor"].lower == pytest.approx(2.0)
    assert result.intervals["rich"].upper == pytest.approx(7.0)
    assert result.diagnostics["simultaneous_coverage_inherited"] is True


def test_estimates_are_projected_to_information_order():
    result = close_information_order(
        {
            "rich": RiskInterval(0.0, 6.0, 10.0, "exploratory"),
            "poor": RiskInterval(0.0, 3.0, 10.0, "exploratory"),
        },
        [("rich", "poor")],
    )
    assert result.status == "complete"
    assert result.intervals["rich"].estimate == pytest.approx(4.5, abs=1e-6)
    assert result.intervals["poor"].estimate == pytest.approx(4.5, abs=1e-6)
    assert set(result.diagnostics["estimate_projected"]) == {"rich", "poor"}


def test_conflicting_intervals_refuse():
    result = close_information_order(
        {
            "rich": RiskInterval(8.0, 9.0, 10.0, "conditional", simultaneous=True),
            "poor": RiskInterval(1.0, 2.0, 4.0, "conditional", simultaneous=True),
        },
        [("rich", "poor")],
    )
    assert result.status == "refused"
    assert result.claim_tier == "refused"
    assert result.refusal and "conflict" in result.refusal["reason"]


def test_refused_input_cannot_be_upgraded_by_order_closure():
    result = close_information_order(
        {
            "rich": RiskInterval(0.0, 1.0, 2.0, "refused"),
            "poor": RiskInterval(1.0, 2.0, 3.0, "conditional"),
        },
        [("rich", "poor")],
    )
    assert result.status == "refused"
    assert result.diagnostics["refused_inputs"] == ("rich",)


def test_information_value_interval_uses_closed_risk_intervals():
    result = close_information_order(
        {
            "with_notes": RiskInterval(2.0, 3.0, 4.0, "conditional", simultaneous=True),
            "tabular": RiskInterval(5.0, 6.0, 8.0, "conditional", simultaneous=True),
        },
        [("with_notes", "tabular")],
    )
    value = information_value_interval(result, richer="with_notes", poorer="tabular")
    assert value["lower"] == pytest.approx(1.0)
    assert value["estimate"] == pytest.approx(3.0)
    assert value["upper"] == pytest.approx(6.0)
    assert value["simultaneous"] is True


def test_cycles_and_unknown_nodes_are_rejected():
    intervals = {
        "a": RiskInterval(0.0, 1.0, 2.0, "proved"),
        "b": RiskInterval(0.0, 1.0, 2.0, "proved"),
    }
    with pytest.raises(ValueError, match="acyclic"):
        close_information_order(intervals, [("a", "b"), ("b", "a")])
    with pytest.raises(ValueError, match="unknown"):
        close_information_order(intervals, [("a", "missing")])


def test_achieved_gain_decomposition_detects_learner_fragility():
    result = close_information_order(
        {
            "intact": RiskInterval(2.0, 3.0, 4.0, "calibrated", simultaneous=True),
            "tabular": RiskInterval(5.0, 6.0, 8.0, "calibrated", simultaneous=True),
        },
        [("intact", "tabular")],
    )
    decomposition = decompose_achieved_gain(
        result, richer="intact", poorer="tabular",
        achieved_richer=10.0, achieved_poorer=8.0,
    )
    assert decomposition["achieved_gain"] == pytest.approx(-2.0)
    assert decomposition["decision_information"]["estimate"] == pytest.approx(3.0)
    assert decomposition["learner_gap_change"]["estimate"] == pytest.approx(-5.0)
    assert "increased" in decomposition["learner_gap_change"]["interpretation"]
