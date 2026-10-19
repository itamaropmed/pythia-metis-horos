"""Training-free lower risk endpoints for frozen finite representations."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class FiniteCellSAAResult:
    status: str
    claim_tier: str
    loss: str
    bound: float | None
    empirical_saa_score: float | None
    concentration_width: float | None
    n_entities: int
    n_cells: int
    singleton_fraction: float
    effective_cell_size: float
    alpha: float
    loss_range_bound: float | None
    representation_provenance: str
    support_provenance: str
    assumptions: tuple[str, ...]
    reason: str | None = None
    concentration_route: str = "global_range"
    sum_squared_entity_sensitivities: float | None = None


def _cell_score(values: np.ndarray, loss: str, quantile: float | None,
                window: float | None) -> float:
    if loss == "mae":
        return float(np.mean(np.abs(values - np.median(values))))
    if loss == "pinball":
        action = float(np.quantile(values, quantile, method="higher"))
        residual = values - action
        return float(np.mean(np.where(
            residual >= 0, quantile * residual, (quantile - 1) * residual)))
    ordered = np.sort(values)
    left = 0
    best = 0
    for right in range(len(ordered)):
        while ordered[right] - ordered[left] > 2 * window:
            left += 1
        best = max(best, right - left + 1)
    return float(1 - best / len(values))


def finite_cell_saa_lower_bound(
        cell_labels, outcomes, entities, *, loss: str,
        outcome_range_bound: float | None = None,
        quantile: float | None = None, window: float | None = None,
        alpha: float = .05, representation_provenance: str,
        support_provenance: str, iid_rows_attested: bool,
        outcome_blind_attested: bool,
        cell_outcome_range_bounds: dict[str, float] | None = None,
        cell_support_provenance: str | None = None,
        exploratory_sampling_override: bool = False) -> FiniteCellSAAResult:
    """Lower-bound Bayes risk for a frozen finite representation.

    Conditional on the observed cell counts, the empirical optimum in each
    cell is a sample-average-approximation lower witness for that cell's Bayes
    risk. With loss range ``M``, changing one entity changes the weighted
    witness by at most ``M/n``; McDiarmid therefore supplies the one-sided
    endpoint used here. Unequal cells and singletons are supported exactly.

    The estimand is population Bayes risk conditional on the finite label,
    weighted by the audited cell counts. Rows must be independent draws and
    labels must be fixed without their outcomes.
    """
    labels = np.asarray(cell_labels).astype(str).reshape(-1)
    y = np.asarray(outcomes, dtype=float).reshape(-1)
    entity = np.asarray(entities).astype(str).reshape(-1)
    selected = str(loss).lower()
    n = len(y)
    if (n < 2 or len(labels) != n or len(entity) != n
            or not np.isfinite(y).all() or np.unique(entity).size != n):
        raise ValueError("aligned finite outcomes with distinct entities are required")
    if selected not in {"mae", "pinball", "window_miss"}:
        return FiniteCellSAAResult(
            status="refused", claim_tier="refused", loss=selected, bound=None,
            empirical_saa_score=None, concentration_width=None, n_entities=n,
            n_cells=int(np.unique(labels).size), singleton_fraction=math.nan,
            effective_cell_size=math.nan, alpha=float(alpha),
            loss_range_bound=None, representation_provenance=str(representation_provenance),
            support_provenance=str(support_provenance), assumptions=(),
            reason="finite-cell SAA supports MAE, pinball, and window_miss")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if not representation_provenance or not support_provenance:
        raise ValueError("representation and support provenance are required")
    if ((not iid_rows_attested and not exploratory_sampling_override)
            or not outcome_blind_attested):
        missing = []
        if not iid_rows_attested:
            missing.append("iid rows")
        if not outcome_blind_attested:
            missing.append("outcome-blind labels")
        return FiniteCellSAAResult(
            status="refused", claim_tier="refused", loss=selected, bound=None,
            empirical_saa_score=None, concentration_width=None, n_entities=n,
            n_cells=int(np.unique(labels).size), singleton_fraction=math.nan,
            effective_cell_size=math.nan, alpha=float(alpha),
            loss_range_bound=None, representation_provenance=str(representation_provenance),
            support_provenance=str(support_provenance), assumptions=(),
            reason="missing required attestation: " + ", ".join(missing))

    cell_caps = None
    route = "global_range"
    if selected in {"mae", "pinball"}:
        if cell_outcome_range_bounds is not None:
            if not cell_support_provenance:
                raise ValueError("cell-specific ranges require cell support provenance")
            raw_caps = {str(key): float(value)
                        for key, value in cell_outcome_range_bounds.items()}
            needed = set(labels)
            if (not needed.issubset(raw_caps)
                    or any(not math.isfinite(raw_caps[key]) or raw_caps[key] <= 0
                           for key in needed)):
                raise ValueError("finite positive cell ranges must cover every observed label")
            for key in needed:
                if np.ptp(y[labels == key]) > raw_caps[key] + 1e-12:
                    return FiniteCellSAAResult(
                        status="refused", claim_tier="refused", loss=selected, bound=None,
                        empirical_saa_score=None, concentration_width=None, n_entities=n,
                        n_cells=int(np.unique(labels).size), singleton_fraction=math.nan,
                        effective_cell_size=math.nan, alpha=float(alpha),
                        loss_range_bound=max(raw_caps.values()),
                        representation_provenance=str(representation_provenance),
                        support_provenance=str(cell_support_provenance), assumptions=(),
                        reason=f"observed outcomes exceed declared hard range for cell {key!r}",
                        concentration_route="cell_specific_range")
            cell_caps = raw_caps
            route = "cell_specific_range"
        elif (outcome_range_bound is None or not math.isfinite(outcome_range_bound)
              or outcome_range_bound <= 0):
            raise ValueError("MAE and pinball require a global or cell-specific hard outcome range")
        if (cell_caps is None and np.ptp(y) > outcome_range_bound + 1e-12):
            return FiniteCellSAAResult(
                status="refused", claim_tier="refused", loss=selected, bound=None,
                empirical_saa_score=None, concentration_width=None, n_entities=n,
                n_cells=int(np.unique(labels).size), singleton_fraction=math.nan,
                effective_cell_size=math.nan, alpha=float(alpha),
                loss_range_bound=float(outcome_range_bound),
                representation_provenance=str(representation_provenance),
                support_provenance=str(support_provenance), assumptions=(),
                reason="observed outcomes exceed the declared hard range")
        multiplier = 1.0
        if selected == "pinball":
            if quantile is None or not math.isfinite(quantile) or not 0 < quantile < 1:
                raise ValueError("pinball requires a quantile strictly between zero and one")
            multiplier = max(float(quantile), 1 - float(quantile))
        if cell_caps is None:
            loss_cap = multiplier * float(outcome_range_bound)
        else:
            cell_caps = {key: multiplier * value for key, value in cell_caps.items()}
            loss_cap = max(cell_caps[key] for key in set(labels))
    else:
        if window is None or not math.isfinite(window) or window <= 0:
            raise ValueError("window miss requires a finite positive half-window")
        if outcome_range_bound is not None:
            raise ValueError("window miss is naturally bounded and takes no outcome range")
        loss_cap = 1.0

    unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    contributions = []
    for cell in range(len(unique)):
        values = y[inverse == cell]
        contributions.append(len(values) * _cell_score(
            values, selected, quantile, window))
    # ``fsum`` makes a bijective relabeling numerically invariant even when it
    # changes NumPy's lexicographic cell traversal order.
    score = math.fsum(contributions) / n
    if cell_caps is None:
        sum_squared = loss_cap * loss_cap / n
    else:
        sum_squared = float(np.sum([cell_caps[label] ** 2 for label in labels]) / (n * n))
    width = math.sqrt(0.5 * math.log(1.0 / alpha) * sum_squared)
    bound = max(0.0, score - width)
    claim_tier = "conditional" if iid_rows_attested else "exploratory"
    sampling_assumption = (
        "rows are independent draws from the registered population"
        if iid_rows_attested else
        "iid sampling is not attested; the numerical endpoint has no finite-sample coverage claim"
    )
    return FiniteCellSAAResult(
        status="complete", claim_tier=claim_tier, loss=selected,
        bound=float(bound), empirical_saa_score=float(score),
        concentration_width=float(width), n_entities=n, n_cells=int(len(unique)),
        singleton_fraction=float(np.sum(counts[counts == 1]) / n),
        effective_cell_size=float(n / len(unique)), alpha=float(alpha),
        loss_range_bound=float(loss_cap),
        representation_provenance=str(representation_provenance),
        support_provenance=str(support_provenance),
        assumptions=(
            sampling_assumption,
            "finite representation labels were frozen without row outcomes",
            "conditional outcome laws are shared within each representation cell",
            "the declared hard support bounds population loss, not only observed outcomes",
            "the endpoint targets the audited empirical mixture of representation cells",
        ) + ((
            "cell-specific hard ranges cover their full population conditional supports"
            if cell_caps is not None else
            "the global hard range covers the full population outcome support"
        ),),
        concentration_route=route,
        sum_squared_entity_sensitivities=sum_squared,
    )
