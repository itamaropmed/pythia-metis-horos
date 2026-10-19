import numpy as np
import pytest

from pythia.core.local_model_risk import build_local_model_risk_registry, build_support_partition
from pythia.core.local_transport import (BettingRiskMonitor, odds_transport, tipping_lambda,
                                         transported_relation)


def _data(seed=0, n_pat=600):
    rng = np.random.default_rng(seed)
    ent = np.repeat(np.arange(n_pat), 2)
    cell = np.repeat(rng.choice(["a", "b", "c"], n_pat, p=[.5, .45, .05]), 2)
    p = np.where(cell == "a", .2, np.where(cell == "b", .6, .4))
    loss = np.repeat(rng.random(n_pat), 2) < p
    return {"m": loss.astype(float)}, {"cell": cell, "global": np.repeat("all", 2 * n_pat)}, ent


def _reg(**kw):
    losses, hier, ent = _data()
    return build_local_model_risk_registry(losses, hier, ent, loss_cap=1., alpha=.05, min_entities=100,
                                           outcome_blind_hierarchy=True, independent_entities=True,
                                           loss_cap_attested=True, **kw)


def test_eligible_ledger_charges_fewer_tests_and_narrows():
    full, elig = _reg(), _reg(alpha_ledger="eligible")
    assert elig.n_charged_tests < full.n_charged_tests
    a, b = full.nodes["m|cell|a"], elig.nodes["m|cell|a"]
    assert b.upper - b.lower < a.upper - a.lower
    assert not elig.nodes["m|cell|c"].eligible and elig.nodes["m|cell|c"].upper == 1.0


def test_contrast_relation_and_sign():
    reg = _reg(alpha_ledger="eligible", relation_test="contrast")
    na, nb = reg.nodes["m|cell|a"], reg.nodes["m|cell|b"]
    assert na.contrast_upper < 0 < nb.contrast_lower
    assert reg.route("m", {"cell": "a", "global": "all"}).local_relation == "certified_lower_than_global"
    assert reg.route("m", {"cell": "b", "global": "all"}).local_relation == "certified_higher_than_global"


def test_support_partition_residual_rule():
    rng = np.random.default_rng(1)
    ent = np.arange(1000)
    fine = rng.choice(["x", "y", "z"], 1000, p=[.6, .35, .05])
    coarse = np.where(fine == "x", "X", "YZ")
    part = build_support_partition({"fine": fine, "coarse": coarse, "global": np.repeat("all", 1000)}, ent, 100)
    lab = part.labels({"fine": fine, "coarse": coarse, "global": np.repeat("all", 1000)})
    assert set(lab[fine == "x"]) == {"fine|x"} and set(lab[fine == "y"]) == {"fine|y"}
    assert set(lab[fine == "z"]) == {"global|all"}           # 'z' residual (~50 entities) backs off past YZ
    assert part.region_of({"fine": "z", "coarse": "YZ", "global": "all"}) == "global|all"


def test_odds_transport_is_monotone_and_sharp_at_one():
    assert odds_transport(.3, .4, 1.0) == pytest.approx((.3, .4))
    lo, hi = odds_transport(.3, .4, 2.0)
    assert lo < .3 and hi > .4 and hi == pytest.approx(4 / 7)
    assert transported_relation((.5, .6), (.3, .35), 1.0) == "certified_higher_than_global"
    lam = tipping_lambda((.5, .6), (.3, .35))
    assert lam > 1 and transported_relation((.5, .6), (.3, .35), lam * 1.01) == "local_shift_unresolved"


def test_monitor_false_alarm_rate_and_power():
    rng = np.random.default_rng(3); alarms = 0
    for _ in range(300):
        m = BettingRiskMonitor(.4, .05)
        for x in rng.random(400) < .4:
            m.update(x)
        alarms += m.alarm_at is not None
    assert alarms / 300 <= .08
    m = BettingRiskMonitor(.4, .05)
    for x in rng.random(400) < .6:
        m.update(x)
    assert m.alarm_at is not None
