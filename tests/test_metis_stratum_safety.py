import numpy as np
import pytest

from metis.case_risk import fit_case_risk_map


def test_caller_cannot_select_an_easy_or_hard_stratum_for_new_case():
    n = 600
    y = np.r_[np.full(300, 101.), np.full(300, 150.)]
    predictions = {"a": np.full(n, 100.), "b": np.full(n, 100.)}
    strata = np.array(["easy"] * 300 + ["hard"] * 300)
    risk_map = fit_case_risk_map(
        y, predictions, strata=strata, entity_ids=np.arange(n),
        prediction_bins=1, disagreement_bins=1, min_entities=50,
        error_cap=100., tolerance=30.,
    )

    aggregate = risk_map.score(
        "new-case", {"a": 100., "b": 100.}, id_salt="test",
    )
    assert [row.risk_value for row in aggregate.model_risks] == [25.5, 25.5]
    for attacker_choice in ("easy", "hard"):
        with pytest.raises(ValueError, match="caller-selected strata are not accepted"):
            risk_map.score(
                "new-case", {"a": 100., "b": 100.},
                stratum=attacker_choice, id_salt="test",
            )
