import math

import numpy as np
import pytest

from pythia import (
    EvidenceContext,
    bounded_loss_gap_from_cmi,
    compile_certificate_graph,
    finite_cmi_upper_bound,
    finite_decision_gap_upper_bound,
    finite_decision_gap_resolution,
    gap_evidence_from_finite_cmi,
)
from pythia.certificate_graph import LowerEvidence


def cmi(y, z, g, alpha=.05):
    return finite_cmi_upper_bound(
        y, z, g, y_alphabet=(0, 1), z_alphabet=(0, 1),
        g_alphabet=(0,), alpha=alpha, alphabet_frozen_before_evaluation=True,
        provenance={"registry": "unit-test"})


def test_finite_cmi_handles_independence_and_target_disclosure():
    independent = cmi(
        np.tile([0, 0, 1, 1], 5000), np.tile([0, 1, 0, 1], 5000),
        np.zeros(20_000, int))
    disclosed = cmi(
        np.tile([0, 1], 10_000), np.tile([0, 1], 10_000),
        np.zeros(20_000, int))
    assert independent.status == "conditional"
    assert independent.plugin_cmi == pytest.approx(0., abs=1e-12)
    assert 0 <= independent.upper_cmi < math.log(2.)
    assert disclosed.plugin_cmi == pytest.approx(math.log(2.))
    assert disclosed.upper_cmi == pytest.approx(math.log(2.))


def test_more_data_tightens_same_empirical_independence_law():
    small = cmi(np.tile([0, 0, 1, 1], 50), np.tile([0, 1, 0, 1], 50),
                np.zeros(200, int))
    large = cmi(np.tile([0, 0, 1, 1], 50_000), np.tile([0, 1, 0, 1], 50_000),
                np.zeros(200_000, int))
    assert large.upper_cmi < small.upper_cmi


def test_registry_failures_refuse_instead_of_discovering_categories():
    unknown = finite_cmi_upper_bound(
        [0, 2], [0, 1], [0, 0], y_alphabet=(0, 1), z_alphabet=(0, 1),
        g_alphabet=(0,), alphabet_frozen_before_evaluation=True,
        provenance={"registry": "x"})
    assert unknown.status == "refused"
    assert "outside" in unknown.refusal["reason"]
    posthoc = finite_cmi_upper_bound(
        [0, 1], [0, 1], [0, 0], y_alphabet=(0, 1), z_alphabet=(0, 1),
        g_alphabet=(0,), alphabet_frozen_before_evaluation=False,
        provenance={"registry": "x"})
    assert posthoc.status == "refused"


def test_cmi_gap_compiles_and_spends_its_event_alpha():
    result = cmi(np.tile([0, 0, 1, 1], 10_000),
                 np.tile([0, 1, 0, 1], 10_000), np.zeros(40_000, int), alpha=.02)
    gap_result = bounded_loss_gap_from_cmi(
        result, loss="MAE", loss_range=100., maximum_bayes_risk=50.)
    context = EvidenceContext(
        target="duration", population="finite audit", loss="MAE",
        outcome_scale="clipped minutes", decision_time="preoperative",
        outcome_clip=(0., 100.))
    edge = gap_evidence_from_finite_cmi(
        gap_result, context=context, poorer="tabular", richer="concept",
        evidence_id="finite-cmi", source="finite multinomial CMI UCB",
        provenance={"joint": "sha256:x"}, event_id="cmi-event")
    base = LowerEvidence(
        evidence_id="base", problem_id=context.problem_id, node="tabular",
        lower=20., claim_tier="conditional", source="base",
        provenance={"hash": "base"}, event_id="base-event",
        failure_probability=.02)
    compiled = compile_certificate_graph(
        target_node="concept", lower_evidence=(base,), gap_evidence=(edge,),
        family_alpha=.05)
    assert compiled.status == "complete"
    assert compiled.family_failure_probability == pytest.approx(.04)
    assert compiled.bound == pytest.approx(max(0., 20. - gap_result["gap_cap"]))


def test_finite_cmi_upper_endpoint_covers_random_small_alphabet_laws():
    rng = np.random.default_rng(20261015)
    violations = 0
    for _ in range(300):
        p = rng.dirichlet(np.full(8, .7)).reshape(2, 2, 2)
        flat = rng.choice(8, size=5000, p=p.ravel())
        y, z, g = flat // 4, (flat // 2) % 2, flat % 2
        result = finite_cmi_upper_bound(
            y, z, g, y_alphabet=(0, 1), z_alphabet=(0, 1),
            g_alphabet=(0, 1), alpha=.05,
            alphabet_frozen_before_evaluation=True,
            provenance={"registry": "random-law-check"})
        pyg, pzg, pg = p.sum(1), p.sum(0), p.sum((0, 1))
        def h(x):
            q = x[x > 0]
            return -float(np.sum(q * np.log(q)))
        truth = h(pyg) + h(pzg) - h(pg) - h(p)
        violations += result.upper_cmi + 1e-12 < truth
    assert violations / 300 < .07


def test_direct_decision_gap_is_sharper_than_cmi_pinsker_for_independence():
    y = np.tile([0, 0, 1, 1], 5000)
    z = np.tile([0, 1, 0, 1], 5000)
    g = np.zeros(20_000, int)
    information = cmi(y, z, g)
    pinsker = bounded_loss_gap_from_cmi(
        information, loss="zero_one", loss_range=1., maximum_bayes_risk=.5)
    direct = finite_decision_gap_upper_bound(
        y, z, g, y_alphabet=(0, 1), z_alphabet=(0, 1), g_alphabet=(0,),
        loss="zero_one", loss_matrix=[[0., 1.], [1., 0.]], alpha=.05,
        alphabet_frozen_before_evaluation=True,
        provenance={"registry": "direct-check"}, maximum_bayes_risk=.5)
    assert direct.plugin_gap == pytest.approx(0.)
    assert direct.gap_cap < pinsker["gap_cap"]


def test_direct_decision_gap_covers_random_population_gaps():
    rng = np.random.default_rng(20261016)
    loss = np.ones((2, 2)) - np.eye(2)
    violations = 0
    for _ in range(300):
        p = rng.dirichlet(np.full(8, .7)).reshape(2, 2, 2)
        flat = rng.choice(8, size=5000, p=p.ravel())
        y, z, g = flat // 4, (flat // 2) % 2, flat % 2
        result = finite_decision_gap_upper_bound(
            y, z, g, y_alphabet=(0, 1), z_alphabet=(0, 1),
            g_alphabet=(0, 1), loss="zero_one", loss_matrix=loss,
            alpha=.05, alphabet_frozen_before_evaluation=True,
            provenance={"registry": "random-direct-check"},
            maximum_bayes_risk=.5)
        coarse_cost = np.einsum("yzg,ya->ga", p, loss)
        fine_cost = np.einsum("yzg,ya->gza", p, loss)
        truth = (float(np.min(coarse_cost, axis=1).sum())
                 - float(np.min(fine_cost, axis=2).sum()))
        violations += result.gap_cap + 1e-12 < truth
    assert violations / 300 < .07


@pytest.mark.parametrize("matrix", [
    np.abs(np.arange(3)[:, None] - np.arange(3)[None, :]),
    (np.arange(3)[:, None] - np.arange(3)[None, :]) ** 2,
    np.maximum(.9 * (np.arange(3)[:, None] - np.arange(3)[None, :]),
               -.1 * (np.arange(3)[:, None] - np.arange(3)[None, :])),
])
def test_direct_gap_supports_mae_mse_and_pinball(matrix):
    y = np.tile([0, 0, 1, 1, 2, 2], 5000)
    z = np.tile([0, 1, 0, 1, 0, 1], 5000)
    g = np.zeros(len(y), int)
    result = finite_decision_gap_upper_bound(
        y, z, g, y_alphabet=(0, 1, 2), z_alphabet=(0, 1),
        g_alphabet=(0,), loss="registered", loss_matrix=matrix,
        alpha=.05, alphabet_frozen_before_evaluation=True,
        provenance={"registry": "multi-loss"})
    assert result.status == "conditional"
    assert result.gap_cap >= 0
    assert result.plugin_gap == pytest.approx(0., abs=1e-12)


def test_outcome_blind_resolution_preflight_matches_theorem_penalty():
    plan = finite_decision_gap_resolution(
        n=11_355, y_alphabet_size=2, z_alphabet_size=2,
        g_alphabet_size=8, loss_range=1., alpha=.05)
    assert plan["joint_alphabet_size"] == 32
    assert plan["gap_uncertainty"] == pytest.approx(plan["l1_radius"])
    assert plan["gap_uncertainty"] == pytest.approx(.0666, abs=1e-4)
    saturated = finite_decision_gap_resolution(
        n=11_355, y_alphabet_size=575, z_alphabet_size=5,
        g_alphabet_size=149, loss_range=814., alpha=.05)
    assert saturated["saturated"] is True
    assert saturated["gap_uncertainty"] == 1628.
