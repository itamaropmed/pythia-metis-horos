import pytest

from pythia.certificate_graph import (
    GapEvidence,
    LowerEvidence,
    compile_certificate_graph,
)


def lower(evidence_id, node, value, *, event="base", alpha=.01, tier="conditional"):
    return LowerEvidence(
        evidence_id=evidence_id, problem_id="mayo-mae-v1", node=node,
        lower=value, claim_tier=tier,
        source="unit test base", provenance={"hash": evidence_id},
        event_id=event if alpha else None, failure_probability=alpha,
        assumptions=("same registered population and loss",))


def gap(evidence_id, poor, rich, value, *, event=None, alpha=0.,
        kind="declared_sensitivity", tier="conditional"):
    return GapEvidence(
        evidence_id=evidence_id, problem_id="mayo-mae-v1",
        poorer=poor, richer=rich, upper_gap=value,
        claim_tier=tier, kind=kind, source="unit test gap",
        provenance={"hash": evidence_id}, event_id=event,
        failure_probability=alpha,
        assumptions=("registered gap cap dominates the decision-risk reduction",))


def test_compiles_best_path_and_charges_shared_event_once():
    result = compile_certificate_graph(
        target_node="raw", family_alpha=.05,
        lower_evidence=(lower("b-tab", "tabular", 30.),),
        gap_evidence=(
            gap("g-concept", "tabular", "concepts", 3., event="gap-family", alpha=.02,
                kind="statistical_upper"),
            gap("g-raw", "concepts", "raw", 5., event="gap-family", alpha=.02,
                kind="statistical_upper"),
            gap("g-direct", "tabular", "raw", 10.),
        ))
    assert result.status == "complete"
    assert result.bound == 22.
    assert result.selected_path == ("tabular", "concepts", "raw")
    assert result.family_failure_probability == pytest.approx(.03)
    assert result.alpha_ledger == {"base": .01, "gap-family": .02}
    assert len(result.candidates) == 2


def test_multiple_bases_are_selected_only_under_joint_ledger():
    result = compile_certificate_graph(
        target_node="raw", family_alpha=.05,
        lower_evidence=(
            lower("b-tab", "tabular", 30., event="joint-bases", alpha=.02),
            lower("b-concept", "concepts", 28., event="joint-bases", alpha=.02),
        ),
        gap_evidence=(
            gap("g-concept", "tabular", "concepts", 3.),
            gap("g-raw", "concepts", "raw", 4.),
        ))
    assert result.bound == 24.
    assert result.selected_evidence == ("b-concept", "g-raw")
    assert result.family_failure_probability == pytest.approx(.02)


def test_structural_zero_transfers_without_alpha():
    result = compile_certificate_graph(
        target_node="parsed-text", family_alpha=.05,
        lower_evidence=(lower("b", "concepts", 12.),),
        gap_evidence=(gap("zero", "concepts", "parsed-text", 0.,
                          kind="structural_zero"),))
    assert result.bound == 12.
    assert result.claim_tier == "conditional"


def test_alpha_overspend_refuses():
    result = compile_certificate_graph(
        target_node="raw", family_alpha=.05,
        lower_evidence=(lower("b", "tabular", 10., alpha=.03),),
        gap_evidence=(gap("g", "tabular", "raw", 1., event="gap", alpha=.03,
                          kind="statistical_upper"),))
    assert result.status == "refused"
    assert "alpha ledger" in result.refusal["reason"]


def test_missing_path_and_disallowed_tier_refuse():
    missing = compile_certificate_graph(
        target_node="raw", lower_evidence=(lower("b", "tabular", 10.),),
        gap_evidence=())
    assert missing.status == "refused"
    weak = compile_certificate_graph(
        target_node="raw",
        lower_evidence=(lower("b", "raw", 10., tier="calibrated"),),
        gap_evidence=())
    assert weak.status == "refused"


def test_invalid_provenance_cycles_and_event_accounting_fail_closed():
    with pytest.raises(ValueError, match="acyclic"):
        compile_certificate_graph(
            target_node="b", lower_evidence=(lower("base", "a", 2.),),
            gap_evidence=(gap("ab", "a", "b", 1.), gap("ba", "b", "a", 1.)))
    with pytest.raises(ValueError, match="conflicting alpha"):
        compile_certificate_graph(
            target_node="raw",
            lower_evidence=(lower("b", "raw", 2., event="same", alpha=.01),
                            lower("c", "raw", 3., event="same", alpha=.02)),
            gap_evidence=())
    with pytest.raises(ValueError, match="provenance"):
        compile_certificate_graph(
            target_node="raw",
            lower_evidence=(LowerEvidence("b", "mayo-mae-v1", "raw", 2., "conditional", "x", {}),),
            gap_evidence=())


def test_incompatible_decision_problems_refuse():
    first = lower("a", "raw", 2.)
    second = LowerEvidence(
        evidence_id="b", problem_id="different-loss", node="raw", lower=3.,
        claim_tier="conditional", source="other", provenance={"hash": "b"},
        event_id="other", failure_probability=.01)
    result = compile_certificate_graph(
        target_node="raw", lower_evidence=(first, second), gap_evidence=())
    assert result.status == "refused"
    assert "incompatible decision problems" in result.refusal["reason"]


def test_topologically_irrelevant_evidence_spends_no_alpha_or_tier():
    relevant = lower("good", "raw", 4., event="good-event", alpha=.01)
    irrelevant = lower(
        "orphan", "orphan-node", 100., event="orphan-event", alpha=.9,
        tier="calibrated")
    result = compile_certificate_graph(
        target_node="raw", lower_evidence=(relevant, irrelevant),
        gap_evidence=(), family_alpha=.05)
    assert result.status == "complete"
    assert result.bound == 4.
    assert result.claim_tier == "conditional"
    assert result.family_failure_probability == pytest.approx(.01)
    assert result.alpha_ledger == {"good-event": .01}
    assert result.diagnostics["ignored_evidence"] == ("orphan",)


def test_adding_preaccounted_candidate_cannot_reduce_compiled_bound():
    base = lower("b", "tabular", 20., event="joint", alpha=.02)
    direct = gap("direct", "tabular", "raw", 8.)
    first = compile_certificate_graph(
        target_node="raw", lower_evidence=(base,), gap_evidence=(direct,))
    second = compile_certificate_graph(
        target_node="raw", lower_evidence=(base,),
        gap_evidence=(direct, gap("via", "tabular", "concept", 2.),
                      gap("finish", "concept", "raw", 3.)))
    assert first.bound == 12.
    assert second.bound == 15.
