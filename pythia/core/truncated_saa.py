"""Original-outcome SAA certificates via safe score truncation.

For any nonnegative loss, an empirical optimum score ``S`` satisfies
``E[S] <= R*``.  For every fixed cap ``c > 0``, ``min(S,c) <= S``; therefore a
bounded-mean lower confidence endpoint for ``E[min(S,c)]`` is also a lower
endpoint for the *original, untruncated* Bayes risk.  Outcome clipping is not
needed.  A finite predeclared cap grid can be selected safely by allocating
alpha across every route/cap candidate.

The diversified-Kelly (dKelly) endpoint follows the positive-capital,
fixed-time construction of Waudby-Smith and Ramdas.  Its AM--GM proof applies
to independent, non-identically distributed observations when the null is on
their average mean; a common score mean is not required.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import math
import numpy as np


@dataclass(frozen=True)
class TruncatedSAACertificate:
    status: str
    claim_tier: str
    loss: str
    bound: float | None
    alpha_family: float
    alpha_per_candidate: float | None
    selected_route: str | None
    selected_score_cap: float | None
    retained_fraction: float | None
    candidates: tuple[Mapping[str, object], ...]
    assumptions: tuple[str, ...]
    diagnostics: Mapping[str, object]
    refusal: Mapping[str, str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def dkelly_lower_mean(values: Sequence[float] | np.ndarray, *, cap: float,
                      alpha: float, diversifications: int = 10,
                      iterations: int = 60) -> float:
    """One-sided dKelly lower endpoint for a bounded average mean."""
    x = np.asarray(values, dtype=float).ravel()
    cap = float(cap)
    if (len(x) < 1 or not np.isfinite(x).all() or not math.isfinite(cap)
            or cap <= 0 or np.any(x < 0) or np.any(x > cap)
            or not 0 < alpha < 1
            or not isinstance(diversifications, (int, np.integer))
            or diversifications < 1 or iterations < 20):
        raise ValueError("bounded finite scores, valid alpha, and diversification required")
    z = x / cap
    upper = float(np.mean(z))
    if upper <= 0:
        return 0.
    fractions = np.arange(1, diversifications + 1, dtype=float) / (diversifications + 1)
    threshold = math.log(1. / alpha)

    def log_evalue(mean):
        logs = np.sum(np.log(
            (1. - fractions[:, None])
            + fractions[:, None] * z[None, :] / mean), axis=1)
        top = float(np.max(logs))
        return top + math.log(float(np.mean(np.exp(logs - top))))

    tiny = np.finfo(float).tiny
    if log_evalue(max(tiny, upper * 1e-15)) < threshold:
        return 0.
    rejected, accepted = 0., upper
    for _ in range(iterations):
        middle = (rejected + accepted) / 2.
        if log_evalue(max(tiny, middle)) >= threshold:
            rejected = middle
        else:
            accepted = middle
    return float(max(0., np.nextafter(accepted * cap, 0.)))


def truncated_saa_portfolio(
        scores_by_route: Mapping[str, Sequence[float] | np.ndarray], *,
        loss: str, score_caps: Sequence[float], alpha_family: float = .05,
        retained_fractions: Mapping[str, float] | None = None,
        route_provenance: Mapping[str, object] | None = None,
        portfolio_frozen_before_evaluation: bool,
        outcome_scale: str = "original loss units") -> TruncatedSAACertificate:
    """Select a score-truncation SAA endpoint under one alpha ledger.

    Routes may correspond to different predeclared block sizes or witness
    constructions.  Scores must be mutually independent *within* each route;
    dependence across routes is harmless because selection is Bonferroni
    protected.  ``retained_fractions`` transfers a route covering only a
    nonnegative-risk subset to the full registered target.
    """
    if not loss or not outcome_scale or not 0 < alpha_family < 1:
        raise ValueError("loss, outcome scale, and valid family alpha are required")
    if not scores_by_route or not score_caps:
        raise ValueError("at least one route and score cap are required")
    if not route_provenance:
        return TruncatedSAACertificate(
            status="refused", claim_tier="refused", loss=loss, bound=None,
            alpha_family=alpha_family, alpha_per_candidate=None,
            selected_route=None, selected_score_cap=None, retained_fraction=None,
            candidates=(), assumptions=(), diagnostics={},
            refusal={"reason": "route provenance is required",
                     "referral": "record block construction and score source hashes"})
    if not portfolio_frozen_before_evaluation:
        return TruncatedSAACertificate(
            status="refused", claim_tier="refused", loss=loss, bound=None,
            alpha_family=alpha_family, alpha_per_candidate=None,
            selected_route=None, selected_score_cap=None, retained_fraction=None,
            candidates=(), assumptions=(), diagnostics={"route_provenance": dict(route_provenance)},
            refusal={"reason": "candidate routes or caps were not frozen before evaluation",
                     "referral": "register the finite portfolio, then evaluate on untouched outcomes"})

    caps = tuple(float(cap) for cap in score_caps)
    if (len(caps) != len(set(caps)) or any(not math.isfinite(cap) or cap <= 0 for cap in caps)):
        raise ValueError("score caps must be unique, finite, and positive")
    names = tuple(str(name) for name in scores_by_route)
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("route names must be unique and nonempty")
    retained_fractions = retained_fractions or {name: 1. for name in names}
    if set(retained_fractions) != set(names):
        raise ValueError("one retained fraction is required for every route")
    if any(not math.isfinite(float(value)) or not 0 < float(value) <= 1
           for value in retained_fractions.values()):
        raise ValueError("retained fractions must lie in (0,1]")

    alpha = float(alpha_family / (len(names) * len(caps)))
    candidates = []
    for name in names:
        raw = np.asarray(scores_by_route[name], dtype=float)
        if (raw.ndim != 1 or raw.size < 1 or not np.isfinite(raw).all()
                or np.any(raw < 0)):
            raise ValueError("each route requires a finite nonnegative score vector")
        retained = float(retained_fractions[name])
        for cap in caps:
            truncated = np.minimum(raw, cap)
            lcb = dkelly_lower_mean(truncated, cap=cap, alpha=alpha)
            candidates.append({
                "route": name, "score_cap": cap, "n_scores": int(raw.size),
                "retained_fraction": retained,
                "untruncated_score_mean": float(np.mean(raw)),
                "truncated_score_mean": float(np.mean(truncated)),
                "lower_truncated_score_mean": lcb,
                "bound": float(retained * lcb),
            })
    selected = max(candidates, key=lambda row: float(row["bound"]))
    return TruncatedSAACertificate(
        status="conditional", claim_tier="conditional", loss=loss,
        bound=float(selected["bound"]), alpha_family=float(alpha_family),
        alpha_per_candidate=alpha, selected_route=str(selected["route"]),
        selected_score_cap=float(selected["score_cap"]),
        retained_fraction=float(selected["retained_fraction"]),
        candidates=tuple(candidates),
        assumptions=(
            "scores are nonnegative empirical optima whose expectations do not exceed the corresponding Bayes risks",
            "scores are mutually independent within each route but may be heterogeneous",
            "routes, score caps, retained fractions, and alpha allocation were frozen before evaluation",
            "any omitted target units have nonnegative risk",
        ),
        diagnostics={
            "concentration": "diversified-Kelly fixed-time e-value",
            "selection": "maximum over a Bonferroni-simultaneous finite portfolio",
            "outcome_clipping_used": False,
            "score_truncation_used": True,
            "target_scale": "original outcome and original loss",
            "moment_assumption_required": False,
            "common_mean_required": False,
            "route_provenance": dict(route_provenance),
        })
