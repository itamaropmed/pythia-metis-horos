"""Compile lower-risk certificates through a nested representation graph.

For a decision problem with Bayes risks ``R(node)``, an edge from a poorer
representation ``u`` to a richer representation ``v`` carries an upper bound

    R(u) - R(v) <= gap(u, v).

Thus any base lower endpoint at ``u`` transfers along a path to ``v`` after
subtracting the path's gap bounds.  Taking the maximum over predeclared paths
is valid when every stochastic evidence event is simultaneous.  The compiler
enforces that condition with an explicit event ledger; repeated uses of one
simultaneous event are charged once.

The graph is an accounting and theorem-composition layer.  It does not infer
representation sufficiency, estimate a gap, fit a model, or validate a
provenance claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping

import math


_TIER_STRENGTH = {
    "proved": 4,
    "conditional": 3,
    "calibrated": 2,
    "exploratory": 1,
    "refused": 0,
}


@dataclass(frozen=True)
class LowerEvidence:
    evidence_id: str
    problem_id: str
    node: str
    lower: float
    claim_tier: str
    source: str
    provenance: Mapping[str, object]
    event_id: str | None = None
    failure_probability: float = 0.0
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class GapEvidence:
    evidence_id: str
    problem_id: str
    poorer: str
    richer: str
    upper_gap: float
    claim_tier: str
    kind: str
    source: str
    provenance: Mapping[str, object]
    event_id: str | None = None
    failure_probability: float = 0.0
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompiledCertificate:
    status: str
    claim_tier: str
    target_node: str
    bound: float | None
    family_failure_probability: float
    selected_path: tuple[str, ...] = ()
    selected_evidence: tuple[str, ...] = ()
    candidates: tuple[Mapping[str, object], ...] = ()
    alpha_ledger: Mapping[str, float] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    refusal: Mapping[str, str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_common(evidence) -> None:
    if (not evidence.evidence_id or not evidence.problem_id or not evidence.source
            or not evidence.provenance):
        raise ValueError("evidence id, source, and provenance are required")
    if evidence.claim_tier not in _TIER_STRENGTH:
        raise ValueError(f"unknown claim tier: {evidence.claim_tier!r}")
    if evidence.claim_tier == "refused":
        raise ValueError("refused evidence cannot enter a compiled certificate")
    alpha = float(evidence.failure_probability)
    if not math.isfinite(alpha) or not 0 <= alpha < 1:
        raise ValueError("failure probability must lie in [0,1)")
    if alpha > 0 and not evidence.event_id:
        raise ValueError("stochastic evidence requires an event id")
    if alpha == 0 and evidence.event_id is not None:
        raise ValueError("zero-cost deterministic evidence must not declare an event id")


def _alpha_ledger(evidence) -> dict[str, float]:
    ledger: dict[str, float] = {}
    for item in evidence:
        if item.event_id is None:
            continue
        alpha = float(item.failure_probability)
        if item.event_id in ledger and not math.isclose(
                ledger[item.event_id], alpha, rel_tol=0, abs_tol=1e-15):
            raise ValueError("one simultaneous event id cannot carry conflicting alpha")
        ledger[item.event_id] = alpha
    return ledger


def compile_certificate_graph(
        *, target_node: str, lower_evidence: tuple[LowerEvidence, ...],
        gap_evidence: tuple[GapEvidence, ...], family_alpha: float = .05,
        permitted_tiers: tuple[str, ...] = ("proved", "conditional")) -> CompiledCertificate:
    """Compile the strongest simultaneous pathwise lower endpoint.

    All candidate evidence is charged in the event ledger because the maximum
    is selected after observing endpoints.  ``event_id`` may be shared only
    when its source procedure genuinely produced one simultaneous event.
    """
    if not target_node or not 0 < family_alpha < 1:
        raise ValueError("target node and family alpha are required")
    if not lower_evidence:
        return CompiledCertificate(
            status="refused", claim_tier="refused", target_node=target_node,
            bound=None, family_failure_probability=0.,
            refusal={"reason": "no base lower-risk evidence supplied",
                     "referral": "register a valid base witness"})
    all_evidence = tuple(lower_evidence) + tuple(gap_evidence)
    ids = [item.evidence_id for item in all_evidence]
    if len(ids) != len(set(ids)):
        raise ValueError("evidence ids must be unique")
    if not permitted_tiers or any(tier not in _TIER_STRENGTH for tier in permitted_tiers):
        raise ValueError("known permitted claim tiers required")
    for item in all_evidence:
        _validate_common(item)
    problem_ids = {item.problem_id for item in all_evidence}
    if len(problem_ids) != 1:
        return CompiledCertificate(
            status="refused", claim_tier="refused", target_node=target_node,
            bound=None, family_failure_probability=0.,
            diagnostics={"problem_ids": tuple(sorted(problem_ids))},
            refusal={"reason": "evidence belongs to incompatible decision problems",
                     "referral": "align population, loss, outcome scale, clip, and decision timing"})
    for base in lower_evidence:
        if not base.node or not math.isfinite(base.lower) or base.lower < 0:
            raise ValueError("base nodes and finite nonnegative lower bounds required")
    for edge in gap_evidence:
        if (not edge.poorer or not edge.richer or edge.poorer == edge.richer
                or not math.isfinite(edge.upper_gap) or edge.upper_gap < 0):
            raise ValueError("valid directed edges and nonnegative gap bounds required")
        if edge.kind not in {"structural_zero", "statistical_upper", "declared_sensitivity"}:
            raise ValueError("unknown gap evidence kind")
        if edge.kind == "structural_zero" and edge.upper_gap != 0:
            raise ValueError("structural-zero edges must have zero gap")

    adjacency: dict[str, list[GapEvidence]] = {}
    for edge in gap_evidence:
        adjacency.setdefault(edge.poorer, []).append(edge)

    # Enumerate simple paths.  A cycle is rejected even if no chosen path uses
    # it because outcome-dependent path exclusion would invalidate selection.
    colors: dict[str, int] = {}
    nodes = ({target_node} | {base.node for base in lower_evidence}
             | {edge.poorer for edge in gap_evidence}
             | {edge.richer for edge in gap_evidence})

    def visit(node: str) -> None:
        state = colors.get(node, 0)
        if state == 1:
            raise ValueError("representation gap graph must be acyclic")
        if state == 2:
            return
        colors[node] = 1
        for edge in adjacency.get(node, ()):
            visit(edge.richer)
        colors[node] = 2

    for node in nodes:
        visit(node)

    candidates: list[dict[str, object]] = []

    def paths(node: str, path_nodes: tuple[str, ...], path_edges: tuple[GapEvidence, ...]):
        if node == target_node:
            yield path_nodes, path_edges
            return
        for edge in adjacency.get(node, ()):
            yield from paths(edge.richer, path_nodes + (edge.richer,), path_edges + (edge,))

    for base in lower_evidence:
        for path_nodes, path_edges in paths(base.node, (base.node,), ()):
            total_gap = float(sum(edge.upper_gap for edge in path_edges))
            candidates.append({
                "bound": float(max(0., base.lower - total_gap)),
                "base_lower": float(base.lower), "total_gap": total_gap,
                "path": path_nodes,
                "evidence": (base.evidence_id,) + tuple(edge.evidence_id for edge in path_edges),
            })
    if not candidates:
        return CompiledCertificate(
            status="refused", claim_tier="refused", target_node=target_node,
            bound=None, family_failure_probability=0., alpha_ledger={},
            diagnostics={"nodes": tuple(sorted(nodes)), "family_alpha": family_alpha},
            refusal={"reason": "no evidence path reaches the target representation",
                     "referral": "add a justified gap edge or choose a supported target"})

    # Graph topology, not endpoint magnitude, determines the candidate union.
    # Evidence disconnected from every base-to-target path is irrelevant to
    # this compilation and must not consume alpha or weaken its claim tier.
    active_ids = {
        evidence_id for candidate in candidates for evidence_id in candidate["evidence"]
    }
    active_evidence = tuple(item for item in all_evidence if item.evidence_id in active_ids)
    ignored_ids = tuple(item.evidence_id for item in all_evidence
                        if item.evidence_id not in active_ids)
    disallowed = tuple(item.evidence_id for item in active_evidence
                       if item.claim_tier not in permitted_tiers)
    if disallowed:
        return CompiledCertificate(
            status="refused", claim_tier="refused", target_node=target_node,
            bound=None, family_failure_probability=0.,
            diagnostics={"disallowed_evidence": disallowed,
                         "ignored_evidence": ignored_ids},
            refusal={"reason": "an active evidence path has a non-permitted claim tier",
                     "referral": "compile that tier separately or strengthen the evidence"})
    ledger = _alpha_ledger(active_evidence)
    spent = float(sum(ledger.values()))
    if spent > family_alpha + 1e-15:
        return CompiledCertificate(
            status="refused", claim_tier="refused", target_node=target_node,
            bound=None, family_failure_probability=spent, alpha_ledger=ledger,
            diagnostics={"family_alpha": family_alpha, "overspend": spent - family_alpha,
                         "ignored_evidence": ignored_ids},
            refusal={"reason": "alpha ledger exceeds the registered family budget",
                     "referral": "preallocate a valid simultaneous evidence budget"})

    chosen = max(candidates, key=lambda row: float(row["bound"]))
    weakest = min((item.claim_tier for item in active_evidence), key=_TIER_STRENGTH.get)
    assumptions = tuple(dict.fromkeys(
        assumption for item in active_evidence for assumption in item.assumptions))
    return CompiledCertificate(
        status="complete", claim_tier=weakest, target_node=target_node,
        bound=float(chosen["bound"]), family_failure_probability=spent,
        selected_path=tuple(chosen["path"]),
        selected_evidence=tuple(chosen["evidence"]),
        candidates=tuple(candidates), alpha_ledger=ledger,
        assumptions=assumptions,
        diagnostics={
            "family_alpha": float(family_alpha),
            "unspent_alpha": float(family_alpha - spent),
            "candidate_count": len(candidates),
            "active_evidence_charged": True,
            "ignored_evidence": ignored_ids,
            "selection_rule": "maximum of simultaneously valid pathwise lower endpoints",
        },
    )
