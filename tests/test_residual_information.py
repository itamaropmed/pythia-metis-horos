import numpy as np

from pythia.core.residual_information import (
    hierarchical_residual_information,
    hierarchical_residual_ladder,
    multiscale_residual_information,
    support_residual_information,
)


def test_multiscale_selection_is_alpha_accounted_and_finds_signal():
    rng = np.random.default_rng(8123)
    n = 1800
    cells = rng.integers(0, 6, n)
    signal = rng.integers(0, 2, n)
    p = np.where(signal == 1, 0.9, 0.1)
    y = (rng.random(n) < p).astype(int)
    good = signal[:, None].astype(float)
    noisy = rng.normal(size=(n, 4))
    result = multiscale_residual_information(
        cells, [noisy, good], y, 2, seed=8123,
        names=["noise", "concept"],
        provenance={"frozen_before_outcomes": True, "outcome_blind": True},
    )
    assert result["status"] == "complete"
    assert result["selection_simultaneous"] is True
    assert result["selected_scale"] == "concept"
    assert result["dI_lower"] > 0
    assert result["alpha_split"] == [0.025, 0.025]


def test_resolution_weighting_is_outcome_blind_and_labels_sparse_support():
    rng = np.random.default_rng(8124)
    n = 900
    cells = rng.integers(0, 9, n)
    y = rng.integers(0, 2, n)
    result = multiscale_residual_information(
        cells, [rng.normal(size=(n, 2)), rng.normal(size=(n, 8))], y, 2,
        alpha_weighting="resolution", min_pairs=10,
        names=["low_dim", "high_dim"],
        provenance={"frozen_before_outcomes": True, "outcome_blind": True},
    )
    assert result["status"] == "complete"
    assert abs(sum(result["alpha_split"]) - 0.05) < 1e-12
    assert all(scale["resolution_status"] in {
        "compatible_with_zero", "insufficient_resolution", "positive_certificate"
    } for scale in result["scales"])


def test_hierarchical_route_requires_and_applies_declared_penalty():
    rng = np.random.default_rng(8125)
    n = 1600
    child = rng.integers(0, 8, n)
    parent = child // 2
    signal = rng.integers(0, 2, n)
    y = (rng.random(n) < np.where(signal, .9, .1)).astype(int)
    geometry = signal[:, None].astype(float)
    result = hierarchical_residual_information(
        child, parent, geometry, y, 2, variation_penalty=.01,
        provenance={"variation_penalty_source": "external synthetic contract"},
    )
    assert result["status"] == "complete"
    assert result["dI_lower"] >= 0
    refused = hierarchical_residual_information(
        child, parent, geometry, y, 2, variation_penalty=None,
        provenance={"variation_penalty_source": "missing"},
    )
    assert refused["claim_tier"] == "refused"


def test_multiscale_refuses_missing_provenance_or_bad_geometry():
    y = np.array([0, 1, 0, 1])
    cells = np.array([0, 0, 1, 1])
    geometry = np.zeros((4, 1))
    refused = multiscale_residual_information(cells, [geometry], y, 2)
    assert refused["claim_tier"] == "refused"
    bad = multiscale_residual_information(
        cells, [np.zeros((3, 1))], y, 2,
        provenance={"frozen_before_outcomes": True},
    )
    assert bad["claim_tier"] == "refused"


def test_support_route_detects_registered_rare_signal_and_not_null():
    def draw(seed, signal):
        rng = np.random.default_rng(seed)
        n = 5000
        cells = rng.integers(0, 8, n)
        rare = rng.random(n) < .08
        logit = .25 * (cells - 3.5) + signal * rare
        p = 1 / (1 + np.exp(-logit))
        y = (rng.random(n) < p).astype(int)
        return cells, rare.astype(str), y

    cells, support, y = draw(551, 2.8)
    positive = support_residual_information(
        cells, support, y, 2, provenance={"registry": "frozen-synthetic"})
    assert positive["status"] == "complete"
    assert positive["dI_lower"] > 0
    assert positive["resolution_status"] == "positive_certificate"

    cells, support, y = draw(552, 0.0)
    null = support_residual_information(
        cells, support, y, 2, provenance={"registry": "frozen-synthetic"})
    assert null["status"] == "complete"
    assert null["dI_lower"] == 0


def test_support_route_reports_insufficient_resolution_for_tiny_support():
    rng = np.random.default_rng(18)
    cells = rng.integers(0, 4, 200)
    support = np.array(["common"] * 197 + ["rare"] * 3)
    y = rng.integers(0, 2, 200)
    out = support_residual_information(
        cells, support, y, 2, min_support=8,
        provenance={"registry": "frozen-synthetic"})
    assert out["dI_lower"] == 0
    assert out["resolution_status"] == "insufficient_resolution"


def test_hierarchical_ladder_is_selection_safe():
    rng = np.random.default_rng(777)
    n = 2400
    child = rng.integers(0, 24, n)
    concept = rng.integers(0, 2, n)
    y = (rng.random(n) < np.where(concept, .88, .12)).astype(int)
    result = hierarchical_residual_ladder(
        child, [child // 2, child // 6], concept[:, None], y, 2,
        variation_penalties=[.01, .02], names=["near", "coarse"],
        provenance={"hierarchy": "frozen synthetic adjacency"},
    )
    assert result["status"] == "complete"
    assert result["selection_simultaneous"] is True
    assert abs(sum(result["alpha_split"]) - .05) < 1e-12
    assert result["selected_hierarchy"] in {"near", "coarse"}
    assert result["dI_lower"] > 0
