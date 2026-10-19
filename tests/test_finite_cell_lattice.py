import numpy as np
import pytest

from pythia import finite_cell_information_lattice


def call(representations, y, edges, **kwargs):
    return finite_cell_information_lattice(
        representations, y, np.arange(len(y)), richer_to_poorer=edges,
        loss="mae", outcome_range_bound=12., alpha=.05,
        representation_provenance={name: f"{name}-parser-v1"
                                   for name in representations},
        support_provenance="analytic-global-support-v1",
        nesting_provenance="frozen-hierarchy-v1",
        iid_rows_attested=True, outcome_blind_attested=True,
        nesting_attested=True, **kwargs)


def test_family_procedure_neighborhood_lattice_is_simultaneous_and_ordered():
    rng = np.random.default_rng(11)
    procedure = np.repeat(np.arange(20), 30)
    family = procedure // 4
    neighborhood = np.array([f"{cell}:{i % 2}" for i, cell in enumerate(procedure)])
    y = (procedure - 9.5) / 2 + rng.uniform(-.5, .5, len(procedure))
    result = call(
        {"family": family, "procedure": procedure, "neighborhood": neighborhood},
        y, (("neighborhood", "procedure"), ("procedure", "family")))
    assert result.status == "complete"
    assert result.spent_alpha == pytest.approx(.05)
    assert result.diagnostics["per_node_alpha"] == pytest.approx(.05 / 3)
    assert result.diagnostics["model_training_performed"] is False
    intervals = result.order_result.intervals
    assert intervals["neighborhood"].estimate <= intervals["procedure"].estimate
    assert intervals["procedure"].estimate <= intervals["family"].estimate
    assert all(value.simultaneous for value in intervals.values())


def test_bijective_relabeling_of_rich_node_is_exactly_invariant():
    rng = np.random.default_rng(21)
    fine = np.repeat(np.arange(12), 25)
    coarse = fine // 3
    y = fine / 4 + rng.uniform(-.5, .5, len(fine))
    first = call({"coarse": coarse, "fine": fine}, y, (("fine", "coarse"),))
    encoded = np.array([f"cipher-{37 * int(value) + 5}" for value in fine])
    second = call({"coarse": coarse, "fine": encoded}, y,
                  (("fine", "coarse"),))
    assert first.node_results["fine"].bound == pytest.approx(
        second.node_results["fine"].bound)
    assert first.order_result.intervals == second.order_result.intervals


def test_observed_non_nesting_fails_closed_even_with_attestation():
    fine = np.array(["a", "a", "b", "b"])
    coarse = np.array(["x", "y", "x", "x"])
    result = call({"coarse": coarse, "fine": fine}, np.arange(4.),
                  (("fine", "coarse"),))
    assert result.status == "refused"
    assert "contradict" in result.refusal["reason"]
    assert result.diagnostics["offending_rich_label"] == "a"


def test_missing_population_nesting_attestation_refuses():
    labels = np.repeat(np.arange(2), 4)
    result = finite_cell_information_lattice(
        {"node": labels}, np.linspace(0, 1, 8), np.arange(8),
        richer_to_poorer=(), loss="mae", outcome_range_bound=1., alpha=.05,
        representation_provenance={"node": "node-v1"},
        support_provenance="support-v1", nesting_provenance="draft-map",
        iid_rows_attested=True, outcome_blind_attested=True,
        nesting_attested=False)
    assert result.status == "refused"
    assert "not attested" in result.refusal["reason"]


def test_cell_local_ranges_flow_through_lattice_without_global_cap():
    rng = np.random.default_rng(31)
    fine = np.repeat(np.arange(8), 30).astype(str)
    coarse = (fine.astype(int) // 2).astype(str)
    centers = fine.astype(float)
    y = centers + rng.uniform(-.1, .1, len(fine))
    fine_caps = {str(i): .2 for i in range(8)}
    coarse_caps = {str(i): 1.2 for i in range(4)}
    result = finite_cell_information_lattice(
        {"coarse": coarse, "fine": fine}, y, np.arange(len(y)),
        richer_to_poorer=(("fine", "coarse"),), loss="mae",
        outcome_range_bound=None, alpha=.05,
        representation_provenance={"coarse": "family-v1", "fine": "cell-v1"},
        support_provenance="analytic-support-v1",
        nesting_provenance="integer-family-map-v1",
        iid_rows_attested=True, outcome_blind_attested=True,
        nesting_attested=True,
        cell_outcome_range_bounds={"coarse": coarse_caps, "fine": fine_caps},
        cell_support_provenance={
            "coarse": "analytic-family-support-v1",
            "fine": "analytic-cell-support-v1",
        })
    assert result.status == "complete"
    assert result.node_results["fine"].concentration_route == "cell_specific_range"
    assert result.node_results["fine"].bound > 0
    assert result.order_result.intervals["coarse"].lower >= result.node_results["fine"].bound


def test_target_blindness_is_required_at_every_node():
    y = np.linspace(0, 1, 20)
    result = finite_cell_information_lattice(
        {"leak": y.astype(str)}, y, np.arange(len(y)),
        richer_to_poorer=(), loss="mae", outcome_range_bound=1., alpha=.05,
        representation_provenance={"leak": "literal-target"},
        support_provenance="unit-support", nesting_provenance="single-node",
        iid_rows_attested=True, outcome_blind_attested=False,
        nesting_attested=True)
    assert result.status == "refused"
    assert "outcome-blind" in result.refusal["reason"]


def test_attested_independent_nuisance_equality_repairs_fragmentation():
    rng = np.random.default_rng(44)
    canonical = np.repeat(np.arange(12), 30)
    nuisance = np.array([f"{cell}|random-id-{i}" for i, cell in enumerate(canonical)])
    y = canonical / 5 + rng.uniform(-.5, .5, len(canonical))
    result = finite_cell_information_lattice(
        {"canonical": canonical, "naive_text": nuisance}, y, np.arange(len(y)),
        richer_to_poorer=(("naive_text", "canonical"),),
        equal_risk_pairs=(("naive_text", "canonical"),),
        loss="mae", outcome_range_bound=4., alpha=.05,
        representation_provenance={
            "canonical": "canonical-parser-v1", "naive_text": "raw-token-parser-v1"},
        support_provenance="analytic-support-v1",
        nesting_provenance="strip-random-token-v1",
        equivalence_provenance="independent-seeded-token-generator-v1",
        iid_rows_attested=True, outcome_blind_attested=True,
        nesting_attested=True, equivalence_attested=True,
        cell_outcome_range_bounds={
            "canonical": {str(value): 1. for value in np.unique(canonical)},
            "naive_text": {str(value): 1. for value in np.unique(nuisance)},
        },
        cell_support_provenance={
            "canonical": "analytic-noise-support-v1",
            "naive_text": "independent-token-noise-support-v1",
        })
    assert result.status == "complete"
    assert result.node_results["naive_text"].bound == 0
    canonical_bound = result.node_results["canonical"].bound
    assert canonical_bound > 0
    assert result.order_result.intervals["naive_text"].lower == pytest.approx(
        canonical_bound)
    assert "naive_text" in result.diagnostics["equality_tightened_nodes"]
    assert result.diagnostics["equality_estimator_source"]["naive_text"] == "canonical"
    assert result.order_result.intervals["naive_text"].estimate == pytest.approx(
        result.node_results["canonical"].empirical_saa_score)


def test_unattested_equivalence_refuses_instead_of_repairing_bound():
    labels = np.repeat(np.arange(3), 4)
    result = finite_cell_information_lattice(
        {"left": labels, "right": labels.astype(str)}, np.linspace(0, 1, 12),
        np.arange(12), richer_to_poorer=(("right", "left"),),
        equal_risk_pairs=(("right", "left"),), loss="mae",
        outcome_range_bound=1., alpha=.05,
        representation_provenance={"left": "left-v1", "right": "right-v1"},
        support_provenance="unit-support", nesting_provenance="map-v1",
        iid_rows_attested=True, outcome_blind_attested=True,
        nesting_attested=True, equivalence_attested=False)
    assert result.status == "refused"
    assert "attestation" in result.refusal["reason"]
