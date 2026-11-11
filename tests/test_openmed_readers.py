"""Open-source medical readers: entity-bag privacy floor and the regex PHI backstop (no model weights needed)."""
import numpy as np
import pytest
from horos.represent.openmed_readers import entity_bag_features, regex_phi_sweep


def test_entity_bag_respects_min_groups():
    ents = [[("DISEASE", "diabetes")]] * 60 + [[("DISEASE", "rare thing")]] * 3
    X, cols = entity_bag_features(ents, np.arange(63), min_groups=50)
    assert "term:diabetes" in cols and "term:rare thing" not in cols
    assert X.shape == (63, len(cols))


def test_regex_phi_sweep():
    assert regex_phi_sweep(["call 507-555-0199", "general anesthesia"]) == ["call 507-555-0199"]


def test_fast_rule_reader_negation():
    pytest.importorskip("medspacy")
    from horos.represent.openmed_readers import FastRuleReader
    r = FastRuleReader()
    X = r.features(["The patient has diabetes.", "No complications.", "Denies hypertension."])
    c = dict(zip(r.columns, X.T))
    assert c["DIABETES:asserted"][0] == 1 and c["COMPLICATION:negated"][1] == 1 and c["HYPERTENSION:negated"][2] == 1
