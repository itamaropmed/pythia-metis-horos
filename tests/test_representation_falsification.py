import pytest

from pythia import RiskInterval, audit_representation_intervention


def interval(lo, estimate, hi, tier="conditional"):
    return RiskInterval(lo, estimate, hi, tier, simultaneous=True)


def audit(base, changed, contract, **kwargs):
    return audit_representation_intervention(
        base, changed, contract=contract,
        intervention_provenance="frozen-synthetic-transform-v1",
        contract_attested=True, **kwargs)


def test_lossless_and_independent_nuisance_require_equal_population_risk():
    base = interval(8, 10, 12)
    assert audit(base, interval(9, 10.5, 11), "bijective_reencoding").verdict == "compatible"
    bad = audit(base, interval(1, 2, 3), "independent_nuisance")
    assert bad.verdict == "contradicted"


def test_garbling_and_enrichment_use_blackwell_direction():
    base = interval(8, 10, 12)
    assert audit(base, interval(13, 14, 15), "garbling").verdict == "confirmed"
    assert audit(base, interval(1, 4, 7), "enrichment").verdict == "confirmed"
    assert audit(base, interval(2, 4, 7), "garbling").verdict == "contradicted"
    assert audit(base, interval(13, 14, 15), "enrichment").verdict == "contradicted"
    assert audit(base, interval(7, 10, 13), "garbling").verdict == "unresolved"


def test_exact_and_partial_target_reveal_have_known_relations():
    base = interval(8, 10, 12)
    exact = audit(base, interval(0, 0, 0), "exact_target_reveal")
    assert exact.verdict == "confirmed"
    partial = audit(
        base, interval(4, 5, 6), "independent_partial_target_reveal",
        reveal_fraction=.5)
    assert partial.expected_lower == pytest.approx(4)
    assert partial.expected_upper == pytest.approx(6)
    assert partial.verdict == "compatible"
    wrong = audit(
        base, interval(7, 8, 9), "independent_partial_target_reveal",
        reveal_fraction=.5)
    assert wrong.verdict == "contradicted"


def test_mae_quantization_is_an_upper_risk_contract():
    base = interval(8, 10, 12)
    confirmed = audit(
        base, interval(.5, 1, 1.2), "mae_quantization",
        quantization_radius=2.5)
    assert confirmed.verdict == "confirmed"
    contradicted = audit(
        base, interval(3, 4, 5), "mae_quantization",
        quantization_radius=2.5)
    assert contradicted.verdict == "contradicted"


def test_missing_attestation_and_refused_intervals_fail_closed():
    base = interval(8, 10, 12)
    missing = audit_representation_intervention(
        base, interval(8, 10, 12), contract="independent_nuisance",
        intervention_provenance="transform-v1", contract_attested=False)
    assert missing.status == "refused"
    refused = RiskInterval(0, 0, 0, "refused")
    result = audit_representation_intervention(
        base, refused, contract="garbling", intervention_provenance="x",
        contract_attested=True)
    assert result.status == "refused"


def test_invalid_contract_parameters_are_rejected():
    base = interval(1, 2, 3)
    with pytest.raises(ValueError, match="unknown"):
        audit(base, base, "magic")
    with pytest.raises(ValueError, match="fraction"):
        audit(base, base, "independent_partial_target_reveal", reveal_fraction=1.1)
    with pytest.raises(ValueError, match="quantization"):
        audit(base, base, "mae_quantization")
