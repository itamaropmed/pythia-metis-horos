"""Finite-cell Bayes-risk witnesses with independent entity clusters."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import math
import numpy as np

from .finite_cell_saa import _cell_score


@dataclass(frozen=True)
class ClusteredFiniteCellSAAResult:
    status: str
    claim_tier: str
    loss: str
    bound: float | None
    empirical_saa_score: float | None
    concentration_width: float | None
    n_rows: int
    n_clusters: int
    n_cells: int
    maximum_cluster_size: int
    effective_cluster_count: float
    alpha: float
    loss_range_bound: float | None
    concentration_route: str
    sum_squared_cluster_sensitivities: float | None
    assumptions: tuple[str, ...]
    reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def clustered_finite_cell_saa_lower_bound(
        cell_labels, outcomes, clusters, *, loss: str,
        outcome_range_bound: float | None = None,
        quantile: float | None = None, window: float | None = None,
        alpha: float = .05, representation_provenance: str,
        support_provenance: str, independent_clusters_attested: bool,
        outcome_blind_attested: bool,
        cell_outcome_range_bounds: Mapping[str, float] | None = None,
        cell_support_provenance: str | None = None,
        exploratory_cluster_override: bool = False,
        ) -> ClusteredFiniteCellSAAResult:
    """Lower-bound case-weighted Bayes risk with arbitrary within-cluster dependence."""
    labels = np.asarray(cell_labels).astype(str).reshape(-1)
    y = np.asarray(outcomes, dtype=float).reshape(-1)
    group = np.asarray(clusters).astype(str).reshape(-1)
    selected = str(loss).lower()
    n = len(y)
    if (n < 2 or len(labels) != n or len(group) != n or not np.isfinite(y).all()):
        raise ValueError("aligned finite outcomes, labels and clusters are required")
    unique_groups, group_inverse, group_counts = np.unique(
        group, return_inverse=True, return_counts=True)
    if len(unique_groups) < 2:
        raise ValueError("at least two independent clusters are required")
    if selected not in {"mae", "pinball", "window_miss"}:
        raise ValueError("loss must be mae, pinball, or window_miss")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if not representation_provenance or not support_provenance:
        raise ValueError("representation and support provenance are required")
    if ((not independent_clusters_attested and not exploratory_cluster_override)
            or not outcome_blind_attested):
        missing = []
        if not independent_clusters_attested:
            missing.append("independent clusters")
        if not outcome_blind_attested:
            missing.append("outcome-blind labels")
        return ClusteredFiniteCellSAAResult(
            status="refused", claim_tier="refused", loss=selected,
            bound=None, empirical_saa_score=None, concentration_width=None,
            n_rows=n, n_clusters=len(unique_groups), n_cells=len(np.unique(labels)),
            maximum_cluster_size=int(np.max(group_counts)),
            effective_cluster_count=float(n * n / np.sum(group_counts ** 2)),
            alpha=float(alpha), loss_range_bound=None,
            concentration_route="cluster_bounded_differences",
            sum_squared_cluster_sensitivities=None, assumptions=(),
            reason="missing required attestation: " + ", ".join(missing))

    multiplier = 1.
    if selected == "pinball":
        if quantile is None or not math.isfinite(quantile) or not 0 < quantile < 1:
            raise ValueError("pinball requires a quantile strictly between zero and one")
        multiplier = max(float(quantile), 1 - float(quantile))
    if selected == "window_miss":
        if window is None or not math.isfinite(window) or window <= 0:
            raise ValueError("window miss requires a finite positive half-window")
        row_caps = np.ones(n)
        loss_cap = 1.
        route = "cluster_natural_unit_range"
    elif cell_outcome_range_bounds is not None:
        if not cell_support_provenance:
            raise ValueError("cell-specific ranges require cell support provenance")
        caps = {str(key): float(value) for key, value in cell_outcome_range_bounds.items()}
        needed = set(labels)
        if (not needed.issubset(caps)
                or any(not math.isfinite(caps[key]) or caps[key] <= 0 for key in needed)):
            raise ValueError("finite positive cell ranges must cover every observed label")
        for key in needed:
            if np.ptp(y[labels == key]) > caps[key] + 1e-12:
                return ClusteredFiniteCellSAAResult(
                    status="refused", claim_tier="refused", loss=selected,
                    bound=None, empirical_saa_score=None, concentration_width=None,
                    n_rows=n, n_clusters=len(unique_groups), n_cells=len(needed),
                    maximum_cluster_size=int(np.max(group_counts)),
                    effective_cluster_count=float(n * n / np.sum(group_counts ** 2)),
                    alpha=float(alpha), loss_range_bound=max(caps.values()),
                    concentration_route="cluster_cell_specific_range",
                    sum_squared_cluster_sensitivities=None, assumptions=(),
                    reason=f"observed outcomes exceed hard range for cell {key!r}")
        row_caps = np.asarray([multiplier * caps[label] for label in labels])
        loss_cap = float(np.max(row_caps))
        route = "cluster_cell_specific_range"
    else:
        if (outcome_range_bound is None or not math.isfinite(outcome_range_bound)
                or outcome_range_bound <= 0):
            raise ValueError("MAE and pinball require a positive hard outcome range")
        if np.ptp(y) > outcome_range_bound + 1e-12:
            return ClusteredFiniteCellSAAResult(
                status="refused", claim_tier="refused", loss=selected,
                bound=None, empirical_saa_score=None, concentration_width=None,
                n_rows=n, n_clusters=len(unique_groups), n_cells=len(np.unique(labels)),
                maximum_cluster_size=int(np.max(group_counts)),
                effective_cluster_count=float(n * n / np.sum(group_counts ** 2)),
                alpha=float(alpha), loss_range_bound=float(outcome_range_bound),
                concentration_route="cluster_global_range",
                sum_squared_cluster_sensitivities=None, assumptions=(),
                reason="observed outcomes exceed the declared hard range")
        loss_cap = multiplier * float(outcome_range_bound)
        row_caps = np.full(n, loss_cap)
        route = "cluster_global_range"

    cells, inverse = np.unique(labels, return_inverse=True)
    pieces = []
    for cell in range(len(cells)):
        values = y[inverse == cell]
        pieces.append(len(values) * _cell_score(values, selected, quantile, window))
    score = math.fsum(pieces) / n
    sensitivities = np.asarray([
        np.sum(row_caps[group_inverse == cluster]) / n
        for cluster in range(len(unique_groups))])
    sum_squared = float(np.sum(sensitivities ** 2))
    width = math.sqrt(.5 * math.log(1 / alpha) * sum_squared)
    tier = "conditional" if independent_clusters_attested else "exploratory"
    independence = (
        "entity clusters are independent; dependence within a cluster is arbitrary"
        if independent_clusters_attested else
        "cluster independence is not attested; the numerical endpoint has no coverage claim"
    )
    return ClusteredFiniteCellSAAResult(
        status="complete", claim_tier=tier, loss=selected,
        bound=float(max(0., score - width)), empirical_saa_score=float(score),
        concentration_width=float(width), n_rows=n,
        n_clusters=len(unique_groups), n_cells=len(cells),
        maximum_cluster_size=int(np.max(group_counts)),
        effective_cluster_count=float(n * n / np.sum(group_counts ** 2)),
        alpha=float(alpha), loss_range_bound=float(loss_cap),
        concentration_route=route,
        sum_squared_cluster_sensitivities=sum_squared,
        assumptions=(
            independence,
            "the estimand is case-weighted over the audited empirical cell mixture",
            "representation labels and cluster membership were frozen without outcomes",
            "conditional outcome marginals are shared within each representation cell",
            "hard ranges cover population losses, not only observed outcomes",
        ), reason=None)


@dataclass(frozen=True)
class ClusteredFixedPolicyInformationValue:
    status: str
    claim_tier: str
    value_lower_bound: float | None
    raw_value_endpoint: float | None
    coarse_risk_lower: float | None
    policy_risk_estimate: float | None
    policy_risk_upper: float | None
    policy_concentration_width: float | None
    alpha: float
    coarse_result: ClusteredFiniteCellSAAResult | None
    assumptions: tuple[str, ...]
    refusal: Mapping[str, str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def clustered_fixed_policy_information_value_lower_bound(
        cell_labels, outcomes, clusters, policy_losses, *, loss: str,
        loss_range_bound: float, outcome_range_bound: float | None = None,
        quantile: float | None = None, window: float | None = None,
        alpha: float = .05, representation_provenance: str,
        support_provenance: str, policy_provenance: str,
        independent_clusters_attested: bool, outcome_blind_coarse_attested: bool,
        policy_frozen_before_evaluation: bool,
        policy_refinement_measurable_attested: bool,
        policy_preoutcome_attested: bool,
        cell_outcome_range_bounds: Mapping[str, float] | None = None,
        cell_support_provenance: str | None = None,
        exploratory_cluster_override: bool = False,
        ) -> ClusteredFixedPolicyInformationValue:
    """Cluster-valid fixed-policy lower endpoint for case-weighted information value."""
    y = np.asarray(outcomes, dtype=float).reshape(-1)
    group = np.asarray(clusters).astype(str).reshape(-1)
    losses = np.asarray(policy_losses, dtype=float).reshape(-1)
    cap = float(loss_range_bound)
    if (len(y) < 2 or len(group) != len(y) or len(losses) != len(y)
            or not np.isfinite(losses).all() or not math.isfinite(cap) or cap <= 0
            or np.any(losses < -1e-12) or np.any(losses > cap + 1e-12)):
        raise ValueError("aligned bounded policy losses and clusters required")
    missing = []
    if not policy_frozen_before_evaluation:
        missing.append("policy frozen before evaluation")
    if not policy_refinement_measurable_attested:
        missing.append("policy measurable from refined information")
    if not policy_preoutcome_attested:
        missing.append("pre-outcome policy inputs")
    if not policy_provenance:
        raise ValueError("policy provenance is required")
    if missing:
        return ClusteredFixedPolicyInformationValue(
            status="refused", claim_tier="refused", value_lower_bound=None,
            raw_value_endpoint=None, coarse_risk_lower=None,
            policy_risk_estimate=None, policy_risk_upper=None,
            policy_concentration_width=None, alpha=float(alpha),
            coarse_result=None, assumptions=(),
            refusal={"reason": "missing policy contract: " + ", ".join(missing)})
    coarse = clustered_finite_cell_saa_lower_bound(
        cell_labels, y, group, loss=loss,
        outcome_range_bound=outcome_range_bound, quantile=quantile, window=window,
        alpha=alpha / 2, representation_provenance=representation_provenance,
        support_provenance=support_provenance,
        independent_clusters_attested=independent_clusters_attested,
        outcome_blind_attested=outcome_blind_coarse_attested,
        cell_outcome_range_bounds=cell_outcome_range_bounds,
        cell_support_provenance=cell_support_provenance,
        exploratory_cluster_override=exploratory_cluster_override)
    if coarse.status != "complete":
        return ClusteredFixedPolicyInformationValue(
            status="refused", claim_tier="refused", value_lower_bound=None,
            raw_value_endpoint=None, coarse_risk_lower=None,
            policy_risk_estimate=None, policy_risk_upper=None,
            policy_concentration_width=None, alpha=float(alpha),
            coarse_result=coarse, assumptions=(),
            refusal={"reason": coarse.reason or "coarse endpoint refused"})
    unique, inverse = np.unique(group, return_inverse=True)
    counts = np.bincount(inverse)
    sensitivities = counts * cap / len(y)
    width = math.sqrt(.5 * math.log(2 / alpha) * float(np.sum(sensitivities ** 2)))
    estimate = float(np.mean(losses))
    upper = float(min(cap, estimate + width))
    raw = float(coarse.bound - upper)
    return ClusteredFixedPolicyInformationValue(
        status="complete", claim_tier=coarse.claim_tier,
        value_lower_bound=float(max(0., raw)), raw_value_endpoint=raw,
        coarse_risk_lower=float(coarse.bound), policy_risk_estimate=estimate,
        policy_risk_upper=upper, policy_concentration_width=float(width),
        alpha=float(alpha), coarse_result=coarse,
        assumptions=coarse.assumptions + (
            "the fixed refined policy uses only pre-outcome declared information",
            "case-weighted policy risk concentrates across independent clusters",
            "the policy upper endpoint uses cluster bounded differences",
        ), refusal=None)
