import numpy as np
from dataclasses import replace

from metis.case_risk import anonymize_id, fit_case_risk_map


def _data(seed=7, n=5000):
    rng = np.random.default_rng(seed)
    hard = rng.integers(0, 2, n)
    center = 90 + 100 * hard + rng.normal(0, 10, n)
    y = center + rng.normal(0, 8 + 32 * hard, n)
    basic = center + rng.normal(0, 13 + 35 * hard, n)
    enriched = center + rng.normal(0, 9 + 20 * hard, n)
    return y, basic, enriched, np.where(hard == 1, "complex", "routine")


def test_unseen_case_scoring_is_outcome_free_and_model_specific():
    y, basic, enriched, strata = _data()
    risk_map = fit_case_risk_map(
        y[:4000], {"basic_xgb": basic[:4000], "xgb_plus_11": enriched[:4000]},
        strata=strata[:4000], entity_ids=np.arange(4000), min_entities=80,
    )
    report = risk_map.score(
        "case-4001", {"basic_xgb": basic[4001], "xgb_plus_11": enriched[4001]},
        id_salt="test-only",
    )
    assert report.anonymous_case_id != "case-4001"
    assert len(report.model_risks) == 2
    assert all(r.risk_value >= 0 and 0 <= r.probability_over_tolerance <= 1
               for r in report.model_risks)
    assert all(r.risk_category in {"low", "moderate", "high", "very_high"}
               for r in report.model_risks)
    assert all("not provenance-verified" in r.uncertainty_status
               for r in report.model_risks)


def test_risk_categories_order_heldout_error_in_synthetic_poc():
    y, basic, enriched, strata = _data(seed=19)
    risk_map = fit_case_risk_map(
        y[:4000], {"basic_xgb": basic[:4000], "xgb_plus_11": enriched[:4000]},
        strata=strata[:4000], entity_ids=np.arange(4000), min_entities=80,
    )
    scores = {name: [] for name in risk_map.model_names}
    errors = {name: [] for name in risk_map.model_names}
    for i in range(4000, 5000):
        values = {"basic_xgb": basic[i], "xgb_plus_11": enriched[i]}
        result = risk_map.score(str(i), values, id_salt="test-only")
        for item in result.model_risks:
            scores[item.model].append(item.risk_value)
            errors[item.model].append(abs(y[i] - values[item.model]))
    for name in risk_map.model_names:
        correlation = np.corrcoef(scores[name], errors[name])[0, 1]
        assert correlation > 0.20


def test_anonymization_requires_private_salt():
    assert anonymize_id("abc", "one") == anonymize_id("abc", "one")
    assert anonymize_id("abc", "one") != anonymize_id("abc", "two")


def test_risk_map_refuses_observed_error_above_declared_cap():
    y = np.zeros(300)
    predictions = {"basic": np.full(300, 100.), "notes": np.full(300, 80.)}
    try:
        fit_case_risk_map(
            y, predictions, entity_ids=np.arange(300), tolerance=30.,
            error_cap=50., min_entities=10,
        )
    except ValueError as exc:
        assert "observed absolute error exceeds error_cap" in str(exc)
    else:
        raise AssertionError("must refuse observed errors beyond the declared support cap")


def test_error_cap_cannot_hide_over_tolerance_cases():
    y = np.zeros(300)
    predictions = {"basic": np.full(300, 100.), "notes": np.full(300, 80.)}
    try:
        fit_case_risk_map(
            y, predictions, entity_ids=np.arange(300), tolerance=30.,
            error_cap=10., min_entities=10,
        )
    except ValueError as exc:
        assert "error_cap must be at least the over-tolerance threshold" in str(exc)
    else:
        raise AssertionError("error_cap below tolerance must be rejected")


def test_uncapped_mae_and_miss_rate_are_reported_inside_cap():
    y = np.zeros(300)
    predictions = {"basic": np.full(300, 100.), "notes": np.full(300, 80.)}
    risk_map = fit_case_risk_map(
        y, predictions, entity_ids=np.arange(300), tolerance=30.,
        error_cap=120., min_entities=10,
    )
    for name, expected in (("basic", 100.), ("notes", 80.)):
        cell = risk_map.cells[name]["global:all"]
        assert cell.expected_absolute_error == expected
        assert cell.error_over_tolerance_probability == 1.0


def test_case_risk_card_cannot_accept_injected_pythia_claims():
    y, basic, enriched, strata = _data(seed=27, n=1000)
    risk_map = fit_case_risk_map(
        y[:800], {"basic": basic[:800], "notes": enriched[:800]},
        strata=strata[:800], entity_ids=np.arange(800), min_entities=20,
    )
    try:
        risk_map.score(
            "case-801", {"basic": basic[800], "notes": enriched[800]},
            id_salt="test-only", cohort_lower_bound=999.,
            cohort_claim_tier="proved",
            information_set_floors={"basic": (999., "proved")},
        )
    except TypeError as exc:
        assert "unexpected keyword" in str(exc)
    else:
        raise AssertionError("Metis score must not accept caller-injected Pythia claims")


def test_external_point_forecast_is_labeled_noncertificate_and_capped():
    y, basic, enriched, strata = _data(seed=28, n=1000)
    risk_map = fit_case_risk_map(
        y[:800], {"basic": basic[:800], "notes": enriched[:800]},
        strata=strata[:800], entity_ids=np.arange(800), min_entities=20,
    )
    result = risk_map.score(
        "case-801", {"basic": basic[800], "notes": enriched[800]},
        id_salt="test-only", point_risk_overrides={"basic": 0., "notes": 0.},
        probability_overrides={"basic": 0., "notes": 0.},
        risk_method="caller_test",
    )
    assert all(r.uncertainty_status.startswith("caller_supplied_model_risk_forecast")
               for r in result.model_risks)
    assert all(r.mae_upper is None and r.probability_upper is None for r in result.model_risks)
    cell_result = risk_map.score(
        "case-801", {"basic": basic[800], "notes": enriched[800]}, id_salt="test-only",
    )
    assert all(r.mae_upper is not None and r.probability_upper is not None
               for r in cell_result.model_risks)
    assert not hasattr(result, "cohort_claim_tier")
    assert not hasattr(result.model_risks[0], "lower_bound_claim_tier")
    try:
        risk_map.score(
            "case-801", {"basic": basic[800], "notes": enriched[800]},
            id_salt="test-only", point_risk_overrides={"basic": 901., "notes": 0.},
        )
    except ValueError as exc:
        assert "declared error support" in str(exc)
    else:
        raise AssertionError("external risk forecasts above the support cap must be refused")


def test_frozen_risk_map_recursively_rejects_nested_mutation_and_serializes():
    y, basic, enriched, strata = _data(seed=20261024, n=700)
    risk_map = fit_case_risk_map(
        y[:600], {"basic": basic[:600], "notes": enriched[:600]},
        strata=strata[:600], entity_ids=np.arange(600), min_entities=20,
    )
    case = {"basic": float(basic[600]), "notes": float(enriched[600])}
    before = risk_map.score("case-601", case, id_salt="test")
    cell = risk_map.cells["basic"]["global:all"]
    try:
        risk_map.cells["basic"]["global:all"] = replace(cell, expected_absolute_error=0.)
    except TypeError:
        pass
    else:
        raise AssertionError("nested risk cells must be immutable")
    try:
        risk_map.prediction_edges["basic"] = ()
    except TypeError:
        pass
    else:
        raise AssertionError("prediction edges must be immutable")
    try:
        dict.__setitem__(risk_map.cells["basic"], "global:all", cell)
    except TypeError:
        pass
    else:
        raise AssertionError("mapping-proxy base-class bypass must fail")
    after = risk_map.score("case-601", case, id_salt="test")
    assert before == after
    serialized = risk_map.to_dict()
    assert serialized["cells"]["basic"]["global:all"]["expected_absolute_error"] == cell.expected_absolute_error
    assert '"cells"' in risk_map.to_json()


def test_risk_map_refuses_implicit_row_independence():
    y, basic, enriched, strata = _data(seed=91, n=500)
    try:
        fit_case_risk_map(
            y, {"basic": basic, "notes": enriched}, strata=strata,
            min_entities=10,
        )
    except ValueError as exc:
        assert "entity_ids are required" in str(exc)
    else:
        raise AssertionError("clustered risk map must not invent independent row IDs")


def test_risk_map_refuses_exact_or_affine_target_copy_even_with_forged_oof_metadata():
    rng = np.random.default_rng(20261024)
    y = rng.normal(100., 20., 600)
    predictions = {"copy": y.copy(), "offset": y + 5.}
    forged_provenance = {
        "prediction_source": "verified_oof", "same_frozen_model": "true",
        "independent_entities_attested": "true", "model_sha256": "placeholder",
    }
    try:
        fit_case_risk_map(
            y, predictions, entity_ids=np.arange(len(y)), error_cap=100.,
            min_entities=50, provenance=forged_provenance,
        )
    except ValueError as exc:
        assert "affine copy of calibration targets" in str(exc)
    else:
        raise AssertionError("target-copy predictions must be refused despite forged OOF metadata")


def test_risk_map_refuses_missing_entity_ids():
    y, basic, enriched, strata = _data(seed=92, n=500)
    for ids in (np.array(["p1"] * 499 + [None], dtype=object),
                np.array(["p1"] * 499 + [""], dtype=object)):
        try:
            fit_case_risk_map(
                y, {"basic": basic, "notes": enriched}, strata=strata,
                entity_ids=ids, min_entities=10,
            )
        except ValueError as exc:
            assert "entity_ids" in str(exc)
        else:
            raise AssertionError("missing entity identity must not enter clustered inference")
