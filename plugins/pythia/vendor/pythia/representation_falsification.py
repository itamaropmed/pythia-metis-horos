"""Metamorphic audits for Bayes-risk intervals under representation changes."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .information_order import RiskInterval


_CONTRACTS = {
    "bijective_reencoding",
    "independent_nuisance",
    "garbling",
    "enrichment",
    "exact_target_reveal",
    "independent_partial_target_reveal",
    "mae_quantization",
}


@dataclass(frozen=True)
class RepresentationInterventionAudit:
    status: str
    verdict: str
    contract: str
    expected_relation: str
    expected_lower: float | None
    expected_upper: float | None
    observed_lower: float
    observed_estimate: float
    observed_upper: float
    estimate_change: float
    intervention_provenance: str
    assumptions: tuple[str, ...]
    reason: str | None = None


def _result(*, baseline, intervention, contract, provenance, verdict,
            relation, expected_lower=None, expected_upper=None,
            assumptions=(), status="complete", reason=None):
    return RepresentationInterventionAudit(
        status=status, verdict=verdict, contract=contract,
        expected_relation=relation, expected_lower=expected_lower,
        expected_upper=expected_upper, observed_lower=float(intervention.lower),
        observed_estimate=float(intervention.estimate),
        observed_upper=float(intervention.upper),
        estimate_change=float(intervention.estimate - baseline.estimate),
        intervention_provenance=str(provenance), assumptions=tuple(assumptions),
        reason=reason,
    )


def audit_representation_intervention(
        baseline: RiskInterval, intervention: RiskInterval, *, contract: str,
        intervention_provenance: str, contract_attested: bool,
        reveal_fraction: float | None = None,
        quantization_radius: float | None = None,
        tolerance: float = 1e-9) -> RepresentationInterventionAudit:
    """Check reported intervals against a registered population risk relation.

    This is a falsification audit, not a way to prove that an intervention
    satisfies its contract. Conditional independence, bijectivity, timing and
    target derivation must be established externally and recorded in
    ``intervention_provenance``.
    """
    baseline.validate()
    intervention.validate()
    selected = str(contract)
    if selected not in _CONTRACTS:
        raise ValueError(f"unknown intervention contract: {selected!r}")
    if not intervention_provenance:
        raise ValueError("intervention provenance is required")
    if baseline.claim_tier == "refused" or intervention.claim_tier == "refused":
        return _result(
            baseline=baseline, intervention=intervention, contract=selected,
            provenance=intervention_provenance, verdict="refused",
            relation="unavailable for refused risk intervals", status="refused",
            reason="a refused interval cannot support a metamorphic comparison")
    if not contract_attested:
        return _result(
            baseline=baseline, intervention=intervention, contract=selected,
            provenance=intervention_provenance, verdict="refused",
            relation="contract requires external attestation", status="refused",
            reason="the transformation contract was not independently attested")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")

    common = (
        "baseline and intervention target the same population, loss and decision timing",
        "reported intervals are simultaneous when a coverage claim is attached to the comparison",
        "the transformation was frozen without certification outcomes",
    )
    if selected in {"bijective_reencoding", "independent_nuisance"}:
        lo, hi = baseline.lower, baseline.upper
        contradicted = intervention.upper < lo - tolerance or intervention.lower > hi + tolerance
        return _result(
            baseline=baseline, intervention=intervention, contract=selected,
            provenance=intervention_provenance,
            verdict="contradicted" if contradicted else "compatible",
            relation="R_new = R_baseline", expected_lower=lo, expected_upper=hi,
            assumptions=common + ((
                "the encoding is measurably invertible" if selected == "bijective_reencoding"
                else "the added channel is conditionally independent of the target given baseline information"
            ),))
    if selected == "garbling":
        contradicted = intervention.upper < baseline.lower - tolerance
        confirmed = intervention.lower >= baseline.upper - tolerance
        verdict = "contradicted" if contradicted else ("confirmed" if confirmed else "unresolved")
        return _result(
            baseline=baseline, intervention=intervention, contract=selected,
            provenance=intervention_provenance, verdict=verdict,
            relation="R_new >= R_baseline", expected_lower=baseline.lower,
            assumptions=common + ("the new representation is a target-independent garbling of baseline",))
    if selected == "enrichment":
        contradicted = intervention.lower > baseline.upper + tolerance
        confirmed = intervention.upper <= baseline.lower + tolerance
        verdict = "contradicted" if contradicted else ("confirmed" if confirmed else "unresolved")
        return _result(
            baseline=baseline, intervention=intervention, contract=selected,
            provenance=intervention_provenance, verdict=verdict,
            relation="R_new <= R_baseline", expected_upper=baseline.upper,
            assumptions=common + ("the baseline representation is recoverable from the enrichment",))
    if selected == "exact_target_reveal":
        contradicted = intervention.lower > tolerance
        confirmed = intervention.upper <= tolerance
        return _result(
            baseline=baseline, intervention=intervention, contract=selected,
            provenance=intervention_provenance,
            verdict="contradicted" if contradicted else ("confirmed" if confirmed else "compatible"),
            relation="R_new = 0", expected_lower=0., expected_upper=0.,
            assumptions=common + ("the exact target and reveal marker are measurable from the representation",))
    if selected == "independent_partial_target_reveal":
        if reveal_fraction is None or not math.isfinite(reveal_fraction) or not 0 <= reveal_fraction <= 1:
            raise ValueError("partial reveal requires a fraction in [0,1]")
        scale = 1.0 - float(reveal_fraction)
        lo, hi = scale * baseline.lower, scale * baseline.upper
        contradicted = intervention.upper < lo - tolerance or intervention.lower > hi + tolerance
        return _result(
            baseline=baseline, intervention=intervention, contract=selected,
            provenance=intervention_provenance,
            verdict="contradicted" if contradicted else "compatible",
            relation="R_new = (1-p) R_baseline", expected_lower=lo, expected_upper=hi,
            assumptions=common + (
                "the reveal indicator is observed",
                "reveal assignment is independent of target and baseline information",
                "loss is zero when the exact target is revealed",
            ))
    if quantization_radius is None or not math.isfinite(quantization_radius) or quantization_radius < 0:
        raise ValueError("MAE quantization requires a finite nonnegative radius")
    radius = float(quantization_radius)
    contradicted = intervention.lower > radius + tolerance
    confirmed = intervention.upper <= radius + tolerance
    return _result(
        baseline=baseline, intervention=intervention, contract=selected,
        provenance=intervention_provenance,
        verdict="contradicted" if contradicted else ("confirmed" if confirmed else "compatible"),
        relation="R_new <= quantization_radius", expected_lower=0., expected_upper=radius,
        assumptions=common + (
            "the decoded value differs from the target by at most the registered radius",
            "the endpoint is MAE under the same units as the quantization radius",
        ))
