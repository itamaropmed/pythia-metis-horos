"""Typed adapters from Pythia route outputs to the certificate graph."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Mapping

from .certificate_graph import GapEvidence, LowerEvidence


@dataclass(frozen=True)
class EvidenceContext:
    target: str
    population: str
    loss: str
    outcome_scale: str
    decision_time: str
    outcome_clip: tuple[float, float] | None = None

    def validate(self) -> None:
        if not all((self.target, self.population, self.loss,
                    self.outcome_scale, self.decision_time)):
            raise ValueError("complete decision-problem context required")
        if self.outcome_clip is not None:
            lo, hi = map(float, self.outcome_clip)
            if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
                raise ValueError("finite increasing outcome clip required")

    @property
    def problem_id(self) -> str:
        self.validate()
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def _check_result_context(result: Mapping[str, object], context: EvidenceContext) -> None:
    context.validate()
    if "loss" in result and str(result["loss"]) != context.loss:
        raise ValueError("route loss conflicts with the evidence context")
    if "outcome_clip" in result and context.outcome_clip is not None:
        route_clip = tuple(map(float, result["outcome_clip"]))
        if route_clip != tuple(map(float, context.outcome_clip)):
            raise ValueError("route outcome clip conflicts with the evidence context")


def _infer_alpha(result: Mapping[str, object], supplied: float | None) -> float:
    if supplied is not None:
        alpha = float(supplied)
    else:
        keys = [key for key in ("alpha_family", "alpha_endpoint", "alpha")
                if key in result]
        if not keys:
            raise ValueError("route failure probability must be supplied or recorded")
        alpha = float(result[keys[0]])
    if not math.isfinite(alpha) or not 0 <= alpha < 1:
        raise ValueError("valid route failure probability required")
    return alpha


def lower_evidence_from_result(
        result: Mapping[str, object], *, context: EvidenceContext, node: str,
        evidence_id: str, source: str, provenance: Mapping[str, object],
        bound_key: str | None = None, event_id: str | None = None,
        failure_probability: float | None = None) -> LowerEvidence:
    """Adapt an already-valid lower endpoint without changing its claim."""
    status = result.get("status")
    if status is None and result.get("claim_tier") in {"proved", "conditional"}:
        status = "conditional"  # compatibility with early typed Pythia routes
    if status not in {"conditional", "complete"}:
        raise ValueError("only completed/conditional route outputs can become base evidence")
    _check_result_context(result, context)
    if bound_key is None:
        found = [key for key in ("bound", "lower", "tuple_bound") if key in result]
        if len(found) != 1:
            raise ValueError("identify exactly one lower-bound field")
        bound_key = found[0]
    if bound_key not in result:
        raise ValueError("lower-bound field missing from route output")
    lower = float(result[bound_key])
    if not math.isfinite(lower) or lower < 0:
        raise ValueError("finite nonnegative lower endpoint required")
    tier = str(result.get("claim_tier", "conditional"))
    alpha = _infer_alpha(result, failure_probability)
    if alpha > 0 and not event_id:
        raise ValueError("stochastic lower evidence requires an event id")
    return LowerEvidence(
        evidence_id=evidence_id, problem_id=context.problem_id, node=node,
        lower=lower, claim_tier=tier, source=source, provenance=dict(provenance),
        event_id=event_id if alpha else None, failure_probability=alpha,
        assumptions=tuple(map(str, result.get("assumptions", ()))),
    )


def gap_evidence_from_cap(
        *, context: EvidenceContext, poorer: str, richer: str,
        upper_gap: float, evidence_id: str, kind: str, source: str,
        provenance: Mapping[str, object], claim_tier: str = "conditional",
        event_id: str | None = None, failure_probability: float = 0.,
        assumptions: tuple[str, ...] = ()) -> GapEvidence:
    """Adapt a separately justified risk-gap cap."""
    context.validate()
    upper = float(upper_gap)
    alpha = float(failure_probability)
    if not math.isfinite(upper) or upper < 0:
        raise ValueError("finite nonnegative risk-gap cap required")
    if not math.isfinite(alpha) or not 0 <= alpha < 1:
        raise ValueError("valid cap failure probability required")
    if alpha > 0 and not event_id:
        raise ValueError("stochastic gap evidence requires an event id")
    return GapEvidence(
        evidence_id=evidence_id, problem_id=context.problem_id,
        poorer=poorer, richer=richer, upper_gap=upper,
        claim_tier=claim_tier, kind=kind, source=source,
        provenance=dict(provenance), event_id=event_id if alpha else None,
        failure_probability=alpha, assumptions=tuple(assumptions),
    )


def gap_evidence_from_profile(
        profile: Mapping[str, object], *, index: int, context: EvidenceContext,
        poorer: str, richer: str, evidence_id: str, source: str,
        provenance: Mapping[str, object]) -> GapEvidence:
    """Select one predeclared point from a sensitivity profile as an edge.

    The selected value remains a declared conditional assumption.  This
    adapter does not estimate or validate it and therefore spends zero alpha.
    """
    if profile.get("status") != "sensitivity_only":
        raise ValueError("a sensitivity profile is required")
    _check_result_context(profile, context)
    if "gap_caps" in profile:
        caps = list(profile["gap_caps"])
    elif "epsilon" in profile and "maximum_clipped_bayes_risk" in profile:
        caps = [float(epsilon) * float(profile["maximum_clipped_bayes_risk"])
                for epsilon in profile["epsilon"]]
    elif "group_reference_bound" in profile and "bounds" in profile:
        base = float(profile["group_reference_bound"])
        caps = [max(0., base - float(bound)) for bound in profile["bounds"]]
    else:
        raise ValueError("profile does not expose a risk-gap sensitivity grid")
    if not isinstance(index, int) or not 0 <= index < len(caps):
        raise ValueError("predeclared profile index is out of range")
    return gap_evidence_from_cap(
        context=context, poorer=poorer, richer=richer,
        upper_gap=float(caps[index]), evidence_id=evidence_id,
        kind="declared_sensitivity", source=source, provenance=provenance,
        claim_tier=str(profile.get("claim_tier", "conditional")),
        assumptions=tuple(map(str, profile.get("assumptions", ()))),
    )


def structural_zero_gap(
        *, context: EvidenceContext, poorer: str, richer: str,
        evidence_id: str, source: str, provenance: Mapping[str, object],
        assumptions: tuple[str, ...]) -> GapEvidence:
    """Register a design/provenance assertion of no added decision value."""
    if not assumptions:
        raise ValueError("structural-zero justification is required")
    return gap_evidence_from_cap(
        context=context, poorer=poorer, richer=richer, upper_gap=0.,
        evidence_id=evidence_id, kind="structural_zero", source=source,
        provenance=provenance, claim_tier="conditional", assumptions=assumptions)


def gap_evidence_from_finite_cmi(
        result: Mapping[str, object], *, context: EvidenceContext,
        poorer: str, richer: str, evidence_id: str, source: str,
        provenance: Mapping[str, object], event_id: str) -> GapEvidence:
    """Adapt a finite-alphabet CMI risk-gap upper endpoint."""
    if result.get("status") != "conditional" or result.get("gap_cap") is None:
        raise ValueError("a completed finite-CMI risk-gap result is required")
    _check_result_context(result, context)
    alpha = _infer_alpha(result, None)
    if alpha <= 0 or not event_id:
        raise ValueError("finite-CMI gap evidence requires a stochastic event id")
    return gap_evidence_from_cap(
        context=context, poorer=poorer, richer=richer,
        upper_gap=float(result["gap_cap"]), evidence_id=evidence_id,
        kind="statistical_upper", source=source, provenance=provenance,
        claim_tier=str(result.get("claim_tier", "conditional")),
        event_id=event_id, failure_probability=alpha,
        assumptions=tuple(map(str, result.get("assumptions", ()))),
    )


def gap_evidence_from_finite_decision(
        result: Mapping[str, object], *, context: EvidenceContext,
        poorer: str, richer: str, evidence_id: str, source: str,
        provenance: Mapping[str, object], event_id: str) -> GapEvidence:
    """Adapt a direct finite-alphabet decision-gap upper endpoint."""
    return gap_evidence_from_finite_cmi(
        result, context=context, poorer=poorer, richer=richer,
        evidence_id=evidence_id, source=source, provenance=provenance,
        event_id=event_id)
