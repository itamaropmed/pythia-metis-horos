import numpy as np

from pythia.core.local_model_risk import build_local_model_risk_registry


def test_local_registry_routes_outcome_blindly_and_contains_observed_mean():
    rng = np.random.default_rng(19)
    n = 1200
    groups = np.repeat(np.arange(600), 2)
    family = np.where(np.arange(n) % 4 < 2, "routine", "complex")
    prediction_bin = np.where(np.arange(n) % 6 < 3, "short", "long")
    losses = {
        "basic": np.clip(rng.gamma(2, 10, n) + 15 * (family == "complex"), 0, 120),
        "plus11": np.clip(rng.gamma(2, 9, n) + 12 * (family == "complex"), 0, 120),
    }
    registry = build_local_model_risk_registry(
        losses, {"family": family, "prediction": prediction_bin,
                 "global": np.repeat("all", n)}, groups,
        loss_cap=120, alpha=.05, min_entities=100,
        outcome_blind_hierarchy=True, independent_entities=True,
        loss_cap_attested=True,
    )
    result = registry.route("basic", {"family": "complex", "prediction": "long", "global": "all"})
    assert result.status == "ok"
    assert result.level == "family"
    assert result.lower <= result.estimate <= result.upper
    assert result.claim_tier == "conditional"
    assert result.global_lower is not None
    assert result.local_relation in {
        "certified_higher_than_global", "certified_lower_than_global",
        "local_shift_unresolved",
    }


def test_missing_assumption_refuses_all_routes():
    registry = build_local_model_risk_registry(
        {"m": np.arange(10.)}, {"global": ["all"] * 10}, np.arange(10),
        loss_cap=10, min_entities=2, outcome_blind_hierarchy=True,
        independent_entities=False, loss_cap_attested=True,
    )
    result = registry.route("m", {"global": "all"})
    assert result.status == "refused"
    assert "independent entities" in result.reason
    assert result.local_relation == "refused"


def test_simultaneous_intervals_can_certify_local_risk_shift():
    rng = np.random.default_rng(122)
    n = 2000
    groups = np.arange(n)
    cell = np.where(np.arange(n) < n // 2, "low", "high")
    loss = np.where(cell == "low", rng.uniform(2, 8, n), rng.uniform(82, 88, n))
    registry = build_local_model_risk_registry(
        {"m": loss}, {"cell": cell, "global": ["all"] * n}, groups,
        loss_cap=100, alpha=.05, min_entities=100,
        outcome_blind_hierarchy=True, independent_entities=True,
        loss_cap_attested=True,
    )
    low = registry.route("m", {"cell": "low", "global": "all"})
    high = registry.route("m", {"cell": "high", "global": "all"})
    assert low.local_relation == "certified_lower_than_global"
    assert high.local_relation == "certified_higher_than_global"


def test_observed_cap_violation_fails_closed():
    try:
        build_local_model_risk_registry(
            {"m": [1., 11., 2.]}, {"global": ["all"] * 3}, np.arange(3),
            loss_cap=10, min_entities=2, outcome_blind_hierarchy=True,
            independent_entities=True, loss_cap_attested=True,
        )
    except ValueError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("cap violation must fail closed")


def test_tiny_node_is_trivial_and_router_backs_off():
    registry = build_local_model_risk_registry(
        {"m": [1., 2., 3., 4.]},
        {"fine": ["rare", "common", "common", "common"],
         "global": ["all"] * 4},
        np.arange(4), loss_cap=10, min_entities=2,
        outcome_blind_hierarchy=True, independent_entities=True,
        loss_cap_attested=True,
    )
    rare = registry.nodes["m|fine|rare"]
    assert (rare.lower, rare.upper) == (0., 10.)
    routed = registry.route("m", {"fine": "rare", "global": "all"})
    assert routed.level == "global"
