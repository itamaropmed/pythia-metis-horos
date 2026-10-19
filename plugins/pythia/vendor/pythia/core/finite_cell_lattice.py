"""Simultaneous finite-cell certificates on an information lattice.

The module composes two already-valid facts without fitting a predictor:

1. a finite-cell SAA endpoint lower-bounds the Bayes risk at one frozen
   representation; and
2. if ``poor = g(rich)`` is a declared deterministic garbling, then
   ``R*(rich) <= R*(poor)`` for every common decision problem.

An equal, outcome-blind Bonferroni ledger makes all node endpoints
simultaneous.  Lower endpoints and plug-in scores are then closed under the
declared information order.  The sample can falsify deterministic nesting but
cannot prove its population or timing contract, so an external attestation and
provenance record remain mandatory.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import math
import numpy as np

from ..information_order import (
    InformationOrderResult,
    RiskInterval,
    close_information_order,
)
from .finite_cell_saa import FiniteCellSAAResult, finite_cell_saa_lower_bound


@dataclass(frozen=True)
class FiniteCellLatticeResult:
    status: str
    claim_tier: str
    alpha: float
    spent_alpha: float
    node_results: Mapping[str, FiniteCellSAAResult]
    order_result: InformationOrderResult | None
    assumptions: tuple[str, ...]
    diagnostics: Mapping[str, object]
    refusal: Mapping[str, str] | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "claim_tier": self.claim_tier,
            "alpha": self.alpha,
            "spent_alpha": self.spent_alpha,
            "node_results": {
                name: asdict(result) for name, result in self.node_results.items()
            },
            "order_result": (
                None if self.order_result is None else self.order_result.to_dict()
            ),
            "assumptions": list(self.assumptions),
            "diagnostics": dict(self.diagnostics),
            "refusal": None if self.refusal is None else dict(self.refusal),
        }


def _observed_deterministic_garbling(rich, poor) -> tuple[bool, str | None]:
    mapping: dict[str, str] = {}
    for rich_value, poor_value in zip(rich, poor):
        previous = mapping.setdefault(str(rich_value), str(poor_value))
        if previous != str(poor_value):
            return False, str(rich_value)
    return True, None


def finite_cell_information_lattice(
        representations: Mapping[str, Sequence[object]], outcomes, entities, *,
        richer_to_poorer: Sequence[tuple[str, str]], loss: str,
        equal_risk_pairs: Sequence[tuple[str, str]] = (),
        outcome_range_bound: float | None = None,
        quantile: float | None = None, window: float | None = None,
        alpha: float = .05,
        representation_provenance: Mapping[str, str],
        support_provenance: str, nesting_provenance: str,
        iid_rows_attested: bool, outcome_blind_attested: bool,
        nesting_attested: bool,
        equivalence_provenance: str | None = None,
        equivalence_attested: bool = False,
        exploratory_sampling_override: bool = False,
        cell_outcome_range_bounds: Mapping[str, Mapping[str, float]] | None = None,
        cell_support_provenance: Mapping[str, str] | None = None,
        ) -> FiniteCellLatticeResult:
    """Certify a predeclared finite representation hierarchy simultaneously.

    Edges point from a richer representation to a poorer deterministic
    garbling.  Node selection, labels, edges and the equal alpha allocation
    must be frozen before certification outcomes are inspected.
    """
    nodes = tuple(representations)
    if not nodes:
        raise ValueError("at least one representation node is required")
    if len(nodes) != len(set(nodes)) or any(not str(name) for name in nodes):
        raise ValueError("representation names must be unique and nonempty")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if not support_provenance or not nesting_provenance:
        raise ValueError("support and nesting provenance are required")
    if set(representation_provenance) != set(nodes):
        raise ValueError("representation provenance must cover every node exactly")

    y = np.asarray(outcomes, dtype=float).reshape(-1)
    entity = np.asarray(entities).reshape(-1)
    labels = {name: np.asarray(values).astype(str).reshape(-1)
              for name, values in representations.items()}
    if any(len(value) != len(y) for value in labels.values()):
        raise ValueError("all representation labels must align with outcomes")

    edges = tuple((str(rich), str(poor)) for rich, poor in richer_to_poorer)
    equalities = tuple((str(left), str(right)) for left, right in equal_risk_pairs)
    unknown = tuple(edge for edge in edges
                    if edge[0] not in labels or edge[1] not in labels)
    unknown_equalities = tuple(pair for pair in equalities
                               if pair[0] not in labels or pair[1] not in labels)
    if unknown or unknown_equalities:
        raise ValueError(
            "information-order contracts reference unknown nodes: "
            f"{unknown + unknown_equalities}")
    if any(left == right for left, right in equalities):
        raise ValueError("equal-risk pairs cannot be self pairs")
    if equalities and (not equivalence_attested or not equivalence_provenance):
        return FiniteCellLatticeResult(
            status="refused", claim_tier="refused", alpha=float(alpha),
            spent_alpha=0., node_results={}, order_result=None, assumptions=(),
            diagnostics={"edges": edges, "equal_risk_pairs": equalities},
            refusal={
                "reason": "risk-equivalence contracts lack independent attestation",
                "referral": "prove bijectivity or conditional-independent nuisance and record provenance",
            })
    if not nesting_attested:
        return FiniteCellLatticeResult(
            status="refused", claim_tier="refused", alpha=float(alpha),
            spent_alpha=0., node_results={}, order_result=None, assumptions=(),
            diagnostics={"edges": edges, "nesting_provenance": nesting_provenance},
            refusal={
                "reason": "population deterministic-garbling contract was not attested",
                "referral": "freeze and audit every representation map before outcomes",
            })
    for rich, poor in edges:
        compatible, offending = _observed_deterministic_garbling(
            labels[rich], labels[poor])
        if not compatible:
            return FiniteCellLatticeResult(
                status="refused", claim_tier="refused", alpha=float(alpha),
                spent_alpha=0., node_results={}, order_result=None, assumptions=(),
                diagnostics={
                    "edges": edges, "nesting_provenance": nesting_provenance,
                    "offending_edge": (rich, poor),
                    "offending_rich_label": offending,
                },
                refusal={
                    "reason": "observed labels contradict deterministic nesting",
                    "referral": "repair the representation map or remove the edge",
                })

    per_node_alpha = float(alpha / len(nodes))
    results: dict[str, FiniteCellSAAResult] = {}
    for name in nodes:
        node_caps = (None if cell_outcome_range_bounds is None
                     else cell_outcome_range_bounds.get(name))
        node_cap_provenance = (None if cell_support_provenance is None
                               else cell_support_provenance.get(name))
        result = finite_cell_saa_lower_bound(
            labels[name], y, entity, loss=loss,
            outcome_range_bound=outcome_range_bound,
            quantile=quantile, window=window, alpha=per_node_alpha,
            representation_provenance=representation_provenance[name],
            support_provenance=support_provenance,
            iid_rows_attested=iid_rows_attested,
            outcome_blind_attested=outcome_blind_attested,
            cell_outcome_range_bounds=node_caps,
            cell_support_provenance=node_cap_provenance,
            exploratory_sampling_override=exploratory_sampling_override,
        )
        results[name] = result
        if result.status != "complete":
            return FiniteCellLatticeResult(
                status="refused", claim_tier="refused", alpha=float(alpha),
                spent_alpha=float(per_node_alpha * len(results)),
                node_results=results, order_result=None, assumptions=(),
                diagnostics={
                    "edges": edges, "nesting_provenance": nesting_provenance,
                    "failed_node": name,
                },
                refusal={
                    "reason": result.reason or "a node certificate was refused",
                    "referral": "satisfy every node contract before lattice closure",
                })

    intervals = {
        name: RiskInterval(
            lower=float(result.bound),
            estimate=float(max(result.bound, result.empirical_saa_score)),
            upper=float(result.loss_range_bound),
            claim_tier=result.claim_tier,
            simultaneous=True,
            source=f"finite_cell_saa:{name}",
        )
        for name, result in results.items()
    }
    # Equality closure is stronger than Blackwell monotonicity.  It is valid
    # for a bijection or an independently attested conditionally independent
    # nuisance channel, but is never inferred merely from similar endpoints.
    parent = {name: name for name in nodes}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in equalities:
        union(left, right)
    components: dict[str, list[str]] = {}
    for name in nodes:
        components.setdefault(find(name), []).append(name)
    equality_tightened = []
    equality_estimator_source = {}
    for members in components.values():
        if len(members) == 1:
            continue
        lower = max(intervals[name].lower for name in members)
        upper = min(intervals[name].upper for name in members)
        if lower > upper + 1e-12:
            return FiniteCellLatticeResult(
                status="refused", claim_tier="refused", alpha=float(alpha),
                spent_alpha=float(alpha), node_results=results, order_result=None,
                assumptions=(), diagnostics={
                    "edges": edges, "equal_risk_pairs": equalities,
                    "infeasible_equivalence_component": tuple(members),
                    "equivalence_provenance": equivalence_provenance,
                }, refusal={
                    "reason": "simultaneous intervals contradict a declared risk equality",
                    "referral": "audit the equality contract, population alignment and support",
                })
        # For equal-risk representations, refining cells can only decrease the
        # empirical SAA optimum while the population optimum is unchanged.
        # Select the least fragmented member by labels alone to reduce that
        # one-sided optimization bias.  This is an outcome-blind estimator
        # route, not an additional confidence event.
        source = min(members, key=lambda name: (results[name].n_cells, nodes.index(name)))
        estimate = float(np.clip(intervals[source].estimate, lower, upper))
        for name in members:
            equality_estimator_source[name] = source
        for name in members:
            previous = intervals[name]
            if lower > previous.lower + 1e-12 or upper < previous.upper - 1e-12:
                equality_tightened.append(name)
            intervals[name] = RiskInterval(
                lower=float(lower), estimate=estimate, upper=float(upper),
                claim_tier=previous.claim_tier, simultaneous=True,
                source=previous.source)
    order = close_information_order(intervals, edges)
    if order.status != "complete":
        return FiniteCellLatticeResult(
            status="refused", claim_tier="refused", alpha=float(alpha),
            spent_alpha=float(alpha), node_results=results, order_result=order,
            assumptions=(), diagnostics={
                "edges": edges, "nesting_provenance": nesting_provenance,
                "equal_risk_pairs": equalities,
                "equivalence_provenance": equivalence_provenance,
                "alpha_allocation": "equal Bonferroni over frozen nodes",
            }, refusal=order.refusal)

    assumptions = (
        "the node registry and edges were frozen before certification outcomes",
        "every edge is a population deterministic garbling with common timing",
        "all nodes target the same population, outcome, action space and loss",
        "the equal Bonferroni ledger makes all finite-cell events simultaneous",
        "declared equal-risk pairs are bijections or conditionally independent nuisance enrichments",
    ) + tuple(dict.fromkeys(
        assumption for result in results.values() for assumption in result.assumptions
    ))
    lattice_tier = "conditional" if all(
        result.claim_tier == "conditional" for result in results.values()) else "exploratory"
    return FiniteCellLatticeResult(
        status="complete", claim_tier=lattice_tier, alpha=float(alpha),
        spent_alpha=float(alpha), node_results=results, order_result=order,
        assumptions=assumptions,
        diagnostics={
            "edges": edges,
            "equal_risk_pairs": equalities,
            "nesting_provenance": nesting_provenance,
            "equivalence_provenance": equivalence_provenance,
            "alpha_allocation": "equal Bonferroni over frozen nodes",
            "per_node_alpha": per_node_alpha,
            "node_count": len(nodes),
            "observed_deterministic_nesting_checked": True,
            "model_training_performed": False,
            "order_closed_nodes": tuple(order.intervals),
            "equality_tightened_nodes": tuple(equality_tightened),
            "equality_estimator_source": equality_estimator_source,
            "equality_estimator_rule": "fewest observed cells, registry order tie-break",
        }, refusal=None)
