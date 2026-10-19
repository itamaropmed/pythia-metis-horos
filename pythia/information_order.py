"""Risk-interval closure under declared information order.

If representation ``rich`` Blackwell-dominates ``poor`` (in particular when
``poor`` is a deterministic or randomized garbling of ``rich``), then Bayes
risk obeys ``R*(rich) <= R*(poor)`` for the same decision problem and loss.
This module propagates already-valid intervals through that order.  It never
upgrades the claim tier of the input intervals.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import numpy as np
from scipy.optimize import minimize


_TIER_STRENGTH = {
    "proved": 4,
    "conditional": 3,
    "calibrated": 2,
    "exploratory": 1,
    "refused": 0,
}


@dataclass(frozen=True)
class RiskInterval:
    lower: float
    estimate: float
    upper: float
    claim_tier: str
    simultaneous: bool = False
    source: str | None = None

    def validate(self) -> None:
        values = np.asarray([self.lower, self.estimate, self.upper], float)
        if not np.isfinite(values).all() or self.lower < 0:
            raise ValueError("risk intervals must contain finite nonnegative values")
        if not self.lower <= self.estimate <= self.upper:
            raise ValueError("risk interval must satisfy lower <= estimate <= upper")
        if self.claim_tier not in _TIER_STRENGTH:
            raise ValueError(f"unknown claim tier: {self.claim_tier!r}")


@dataclass(frozen=True)
class InformationOrderResult:
    status: str
    claim_tier: str
    intervals: Mapping[str, RiskInterval]
    edges: tuple[tuple[str, str], ...]
    assumptions: tuple[str, ...]
    diagnostics: Mapping[str, object]
    refusal: Mapping[str, str] | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "claim_tier": self.claim_tier,
            "intervals": {name: asdict(value) for name, value in self.intervals.items()},
            "edges": [list(edge) for edge in self.edges],
            "assumptions": list(self.assumptions),
            "diagnostics": dict(self.diagnostics),
            "refusal": None if self.refusal is None else dict(self.refusal),
        }


def _reachability(names: tuple[str, ...], edges: tuple[tuple[str, str], ...]) -> np.ndarray:
    index = {name: i for i, name in enumerate(names)}
    reach = np.eye(len(names), dtype=bool)
    for rich, poor in edges:
        if rich not in index or poor not in index:
            raise ValueError(f"information-order edge references an unknown node: {(rich, poor)}")
        if rich == poor:
            raise ValueError("information-order edges cannot be self loops")
        reach[index[rich], index[poor]] = True
    for k in range(len(names)):
        reach |= reach[:, [k]] & reach[[k], :]
    if any(reach[i, j] and reach[j, i] for i in range(len(names)) for j in range(i + 1, len(names))):
        raise ValueError("information-order graph must be acyclic")
    return reach


def _weakest_tier(intervals: Mapping[str, RiskInterval]) -> str:
    return min((value.claim_tier for value in intervals.values()), key=_TIER_STRENGTH.get)


def close_information_order(
    intervals: Mapping[str, RiskInterval],
    richer_to_poorer: Iterable[tuple[str, str]],
    *,
    weights: Mapping[str, float] | None = None,
    tolerance: float = 1e-9,
) -> InformationOrderResult:
    """Close risk intervals and estimates under a representation partial order.

    Edges point from a more informative representation to a garbling. Input
    intervals must be jointly/simultaneously valid if the closed intervals are
    to inherit simultaneous coverage.
    """
    if not intervals:
        raise ValueError("at least one risk interval is required")
    for value in intervals.values():
        value.validate()
    refused_inputs = tuple(
        name for name, value in intervals.items() if value.claim_tier == "refused"
    )
    if refused_inputs:
        return InformationOrderResult(
            status="refused", claim_tier="refused", intervals=dict(intervals),
            edges=tuple(dict.fromkeys(tuple(edge) for edge in richer_to_poorer)),
            assumptions=(), diagnostics={"refused_inputs": refused_inputs},
            refusal={
                "reason": "information-order closure cannot repair a refused input interval",
                "referral": "resolve or remove the refused representation before closure",
            },
        )
    names = tuple(intervals)
    edges = tuple(dict.fromkeys(tuple(edge) for edge in richer_to_poorer))
    reach = _reachability(names, edges)
    lower = np.asarray([intervals[name].lower for name in names], float)
    upper = np.asarray([intervals[name].upper for name in names], float)
    raw_estimate = np.asarray([intervals[name].estimate for name in names], float)

    # Ancestors are more informative and supply valid lower bounds downstream.
    closed_lower = np.asarray([
        np.max(lower[reach[:, j]]) for j in range(len(names))
    ])
    # Descendants are less informative and supply valid upper bounds upstream.
    closed_upper = np.asarray([
        np.min(upper[reach[i, :]]) for i in range(len(names))
    ])
    infeasible = np.flatnonzero(closed_lower > closed_upper + tolerance)
    if infeasible.size:
        bad = tuple(names[i] for i in infeasible)
        return InformationOrderResult(
            status="refused", claim_tier="refused", intervals=dict(intervals), edges=edges,
            assumptions=("declared edges are valid Blackwell/garbling relations",),
            diagnostics={"infeasible_nodes": bad},
            refusal={
                "reason": "risk intervals conflict with the declared information order",
                "referral": "audit representation nesting and use simultaneous risk intervals",
            },
        )

    w = np.ones(len(names), float) if weights is None else np.asarray([
        float(weights.get(name, 1.0)) for name in names
    ])
    if np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("isotonic projection weights must be finite and positive")
    constraints = [
        {"type": "ineq", "fun": lambda x, i=i, j=j: x[j] - x[i]}
        for i, j in zip(*np.where(reach & ~np.eye(len(names), dtype=bool)))
    ]
    start = np.clip(raw_estimate, closed_lower, closed_upper)
    fit = minimize(
        lambda x: float(np.sum(w * (x - raw_estimate) ** 2)),
        start, method="SLSQP", bounds=list(zip(closed_lower, closed_upper)),
        constraints=constraints, options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not fit.success:
        return InformationOrderResult(
            status="refused", claim_tier="refused", intervals=dict(intervals), edges=edges,
            assumptions=("declared edges are valid Blackwell/garbling relations",),
            diagnostics={"optimizer_message": str(fit.message)},
            refusal={"reason": "order-constrained estimator projection failed",
                     "referral": "inspect interval geometry and optimizer diagnostics"},
        )

    simultaneous = all(value.simultaneous for value in intervals.values())
    projected = {
        name: RiskInterval(
            lower=float(closed_lower[i]), estimate=float(fit.x[i]),
            upper=float(closed_upper[i]), claim_tier=intervals[name].claim_tier,
            simultaneous=simultaneous, source=intervals[name].source,
        )
        for i, name in enumerate(names)
    }
    return InformationOrderResult(
        status="complete", claim_tier=_weakest_tier(intervals), intervals=projected,
        edges=edges,
        assumptions=(
            "every directed edge is a valid deterministic or randomized garbling",
            "all nodes use the same population, endpoint, loss, and decision timing",
            "coverage is simultaneous only when all supplied intervals are simultaneous",
        ),
        diagnostics={
            "simultaneous_coverage_inherited": simultaneous,
            "lower_tightened": tuple(names[i] for i in np.flatnonzero(closed_lower > lower + tolerance)),
            "upper_tightened": tuple(names[i] for i in np.flatnonzero(closed_upper < upper - tolerance)),
            "estimate_projected": tuple(names[i] for i in np.flatnonzero(np.abs(fit.x - raw_estimate) > tolerance)),
            "objective": float(fit.fun),
        },
    )


def information_value_interval(
    result: InformationOrderResult, *, richer: str, poorer: str
) -> dict[str, object]:
    """Bound the Bayes-risk reduction attributable to added information."""
    if result.status != "complete":
        raise ValueError("information value requires a completed order closure")
    if (richer, poorer) not in result.edges:
        names = tuple(result.intervals)
        reach = _reachability(names, result.edges)
        index = {name: i for i, name in enumerate(names)}
        if richer not in index or poorer not in index or not reach[index[richer], index[poorer]]:
            raise ValueError(f"{richer!r} is not declared more informative than {poorer!r}")
    rich = result.intervals[richer]
    poor = result.intervals[poorer]
    return {
        "richer": richer,
        "poorer": poorer,
        "value_definition": "R*(poorer) - R*(richer)",
        "lower": float(max(0.0, poor.lower - rich.upper)),
        "estimate": float(max(0.0, poor.estimate - rich.estimate)),
        "upper": float(max(0.0, poor.upper - rich.lower)),
        "claim_tier": result.claim_tier,
        "simultaneous": rich.simultaneous and poor.simultaneous,
    }


def decompose_achieved_gain(
    result: InformationOrderResult,
    *,
    richer: str,
    poorer: str,
    achieved_richer: float,
    achieved_poorer: float,
) -> dict[str, object]:
    """Separate achieved improvement into information and learner-gap terms.

    With ``L = R* + H`` and ``H >= 0``, the achieved gain satisfies
    ``L_poor-L_rich = information_value + (H_poor-H_rich)``.  A negative
    learner-gap term is the precise signature of nuisance sensitivity: the
    richer representation is at least as informative, but the fitted learner
    moved farther from its own attainable floor.
    """
    achieved = np.asarray([achieved_richer, achieved_poorer], float)
    if not np.isfinite(achieved).all() or np.any(achieved < 0):
        raise ValueError("achieved risks must be finite and nonnegative")
    value = information_value_interval(result, richer=richer, poorer=poorer)
    achieved_gain = float(achieved_poorer - achieved_richer)
    learner_estimate = achieved_gain - float(value["estimate"])
    learner_interval = (
        achieved_gain - float(value["upper"]),
        achieved_gain - float(value["lower"]),
    )
    if learner_interval[1] < 0:
        interpretation = "richer representation provably increased the learner gap"
    elif learner_interval[0] > 0:
        interpretation = "richer representation provably reduced the learner gap"
    else:
        interpretation = "learner-gap movement is unresolved"
    return {
        "achieved_gain": achieved_gain,
        "decision_information": value,
        "learner_gap_change": {
            "definition": "H(poorer) - H(richer)",
            "lower": float(learner_interval[0]),
            "estimate": float(learner_estimate),
            "upper": float(learner_interval[1]),
            "interpretation": interpretation,
        },
        "identity": "achieved_gain = decision_information + learner_gap_change",
    }
