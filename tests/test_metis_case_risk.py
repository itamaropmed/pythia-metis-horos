import numpy as np

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
        strata=strata[:4000], min_entities=80,
    )
    report = risk_map.score(
        "case-4001", {"basic_xgb": basic[4001], "xgb_plus_11": enriched[4001]},
        stratum=strata[4001], id_salt="test-only",
    )
    assert report.anonymous_case_id != "case-4001"
    assert len(report.model_risks) == 2
    assert all(r.risk_value >= 0 and 0 <= r.probability_over_tolerance <= 1
               for r in report.model_risks)
    assert all(r.risk_category in {"low", "moderate", "high", "very_high"}
               for r in report.model_risks)


def test_risk_categories_order_heldout_error_in_synthetic_poc():
    y, basic, enriched, strata = _data(seed=19)
    risk_map = fit_case_risk_map(
        y[:4000], {"basic_xgb": basic[:4000], "xgb_plus_11": enriched[:4000]},
        strata=strata[:4000], min_entities=80,
    )
    scores = {name: [] for name in risk_map.model_names}
    errors = {name: [] for name in risk_map.model_names}
    for i in range(4000, 5000):
        values = {"basic_xgb": basic[i], "xgb_plus_11": enriched[i]}
        result = risk_map.score(str(i), values, stratum=strata[i], id_salt="test-only")
        for item in result.model_risks:
            scores[item.model].append(item.risk_value)
            errors[item.model].append(abs(y[i] - values[item.model]))
    for name in risk_map.model_names:
        correlation = np.corrcoef(scores[name], errors[name])[0, 1]
        assert correlation > 0.20


def test_anonymization_requires_private_salt():
    assert anonymize_id("abc", "one") == anonymize_id("abc", "one")
    assert anonymize_id("abc", "one") != anonymize_id("abc", "two")
