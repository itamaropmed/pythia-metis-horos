import pytest

from pythia import (
    EvidenceContext,
    compile_certificate_graph,
    gap_evidence_from_cap,
    gap_evidence_from_profile,
    lower_evidence_from_result,
    structural_zero_gap,
    truncated_saa_portfolio,
)


CTX = EvidenceContext(
    target="duration", population="frozen cohort", loss="MAE",
    outcome_scale="minutes", decision_time="two days before surgery",
    outcome_clip=(0., 300.))


def base_result(**updates):
    row = {
        "status": "conditional", "claim_tier": "conditional",
        "loss": "MAE", "lower": 30., "alpha_family": .02,
        "outcome_clip": [0., 300.], "assumptions": ["stable exact groups"],
    }
    row.update(updates)
    return row


def test_adapters_compile_an_end_to_end_ladder():
    base = lower_evidence_from_result(
        base_result(), context=CTX, node="tabular", evidence_id="base",
        source="exact group SAA", provenance={"cohort_hash": "abc"},
        event_id="base-family")
    profile = {
        "status": "sensitivity_only", "claim_tier": "conditional",
        "loss": "MAE", "gap_caps": [2., 5.],
        "assumptions": ["declared concept information cap"],
    }
    concept = gap_evidence_from_profile(
        profile, index=0, context=CTX, poorer="tabular", richer="concepts",
        evidence_id="concept-gap", source="registered sensitivity",
        provenance={"registry": "v1"})
    raw = gap_evidence_from_cap(
        context=CTX, poorer="concepts", richer="raw-notes", upper_gap=4.,
        evidence_id="raw-gap", kind="statistical_upper",
        source="independent acquisition audit", provenance={"hash": "def"},
        event_id="gap-family", failure_probability=.02,
        assumptions=("gap calibration is stable",))
    result = compile_certificate_graph(
        target_node="raw-notes", lower_evidence=(base,),
        gap_evidence=(concept, raw), family_alpha=.05)
    assert result.status == "complete"
    assert result.bound == 24.
    assert result.family_failure_probability == pytest.approx(.04)


def test_context_is_content_addressed_and_mismatches_refuse_or_raise():
    same = EvidenceContext(**CTX.__dict__)
    other = EvidenceContext(**{**CTX.__dict__, "loss": "MSE"})
    assert same.problem_id == CTX.problem_id
    assert other.problem_id != CTX.problem_id
    with pytest.raises(ValueError, match="loss conflicts"):
        lower_evidence_from_result(
            base_result(), context=other, node="x", evidence_id="x",
            source="x", provenance={"hash": "x"}, event_id="x")


def test_adapter_refuses_ambiguous_bounds_and_unaccounted_randomness():
    with pytest.raises(ValueError, match="exactly one"):
        lower_evidence_from_result(
            base_result(bound=31.), context=CTX, node="x", evidence_id="x",
            source="x", provenance={"hash": "x"}, event_id="x")
    with pytest.raises(ValueError, match="event id"):
        lower_evidence_from_result(
            base_result(), context=CTX, node="x", evidence_id="x",
            source="x", provenance={"hash": "x"})


def test_localized_profile_and_structural_zero_adapt_without_claim_upgrade():
    profile = {
        "status": "sensitivity_only", "claim_tier": "conditional",
        "loss": "MAE", "epsilon": [0., .25, 1.],
        "maximum_clipped_bayes_risk": 100.,
        "assumptions": ["sufficient outside registered exceptional set"],
    }
    edge = gap_evidence_from_profile(
        profile, index=1, context=CTX, poorer="groups", richer="notes",
        evidence_id="localized", source="sensitivity",
        provenance={"version": "v1"})
    assert edge.upper_gap == 25.
    zero = structural_zero_gap(
        context=CTX, poorer="notes", richer="parser-equivalent-notes",
        evidence_id="zero", source="deterministic parser equivalence",
        provenance={"parser_hash": "p"},
        assumptions=("the richer object is a deterministic recoding with identical sigma-field",))
    assert zero.upper_gap == 0
    assert zero.kind == "structural_zero"


def test_structural_zero_needs_a_real_justification():
    with pytest.raises(ValueError, match="justification"):
        structural_zero_gap(
            context=CTX, poorer="a", richer="b", evidence_id="z",
            source="claim", provenance={"hash": "z"}, assumptions=())


def test_original_scale_truncated_saa_compiles_as_base_evidence():
    context = EvidenceContext(
        target="duration", population="frozen prospective cohort", loss="MAE",
        outcome_scale="original minutes", decision_time="preoperative cutoff")
    route = truncated_saa_portfolio(
        {"k2": [3.] * 200}, loss="MAE", score_caps=(2., 4.),
        route_provenance={"blocks": "sha256:blocks"},
        portfolio_frozen_before_evaluation=True, outcome_scale="original minutes")
    evidence = lower_evidence_from_result(
        route.to_dict(), context=context, node="registered-representation",
        evidence_id="original-scale-saa", source="score-truncated dKelly SAA",
        provenance={"scores": "sha256:scores"}, event_id="saa-family")
    compiled = compile_certificate_graph(
        target_node="registered-representation", lower_evidence=(evidence,),
        gap_evidence=(), family_alpha=.05)
    assert compiled.status == "complete"
    assert compiled.bound == route.bound
    assert compiled.family_failure_probability == pytest.approx(.05)
