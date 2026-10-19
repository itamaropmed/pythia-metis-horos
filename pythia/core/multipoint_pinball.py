"""Capped overlapping multi-point witnesses for pinball Bayes risk.

Each registered partition covers every independent entity exactly once.  A
portfolio may average many such partitions, so groups overlap across
partitions but not within one partition.  Bounded differences is charged at
the entity level and remains valid under this overlap.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MultipointPinballCertificate:
    status: str
    claim_tier: str
    loss: str
    bound: float | None
    quantile: float
    selected_k: int | None
    selected_score_cap: float | None
    selected_center: float | None
    selected_radius: float | None
    selected_penalty: float | None
    alpha_family: float
    alpha_per_candidate: float | None
    candidates: tuple[Mapping[str, float | int], ...]
    assumptions: tuple[str, ...]
    diagnostics: Mapping[str, object]
    refusal: Mapping[str, str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _pinball_scores(block_outcomes: np.ndarray, quantile: float) -> np.ndarray:
    action = np.quantile(
        block_outcomes, quantile, axis=1, method="inverted_cdf")
    residual = block_outcomes - action[:, None]
    return np.mean(np.maximum(
        quantile * residual, (quantile - 1) * residual), axis=1)


def overlapping_pinball_barycentric_portfolio(
        outcomes, partitions_by_k: Mapping[int, np.ndarray], *,
        quantile: float, clip: tuple[float, float],
        declared_discrepancy_by_k: Mapping[int, float],
        score_caps: Sequence[float], alpha_family: float = .05,
        partition_provenance: str | None,
        discrepancy_provenance: str | None,
        portfolio_frozen_before_outcomes: bool,
        shift=None, shift_provenance: str | None = None,
        ) -> MultipointPinballCertificate:
    """Selection-safe capped multi-partition pinball lower certificate.

    For every partition and block, SAA optimism plus the barycentric theorem
    gives ``E S <= average risk + P_k/2``.  Averaging registered partitions
    preserves this inequality.  If a block score is capped at ``c``, changing
    one clipped outcome changes its block score by at most
    ``min(c, max(q,1-q)*D/k)``.  Since the entity occurs once in each balanced
    partition, the sensitivity of the grand mean is
    ``min(k*c, max(q,1-q)*D)/n``.  McDiarmid then yields the endpoint below.
    """
    y = np.asarray(outcomes, dtype=float).ravel()
    lo, hi = map(float, clip)
    if (len(y) < 4 or not np.isfinite(y).all() or not np.isfinite([lo, hi]).all()
            or lo >= hi or not math.isfinite(quantile) or not 0 < quantile < 1
            or not 0 < alpha_family < 1):
        raise ValueError("finite outcomes, clip, quantile, and alpha required")
    if not partition_provenance or not discrepancy_provenance:
        return MultipointPinballCertificate(
            status="refused", claim_tier="refused", loss="pinball", bound=None,
            quantile=float(quantile), selected_k=None, selected_score_cap=None,
            selected_center=None, selected_radius=None, selected_penalty=None,
            alpha_family=float(alpha_family), alpha_per_candidate=None,
            candidates=(), assumptions=(), diagnostics={},
            refusal={"reason": "partition and discrepancy provenance are required"})
    if shift is not None and not shift_provenance:
        raise ValueError("shift provenance required when a shift is supplied")
    if not portfolio_frozen_before_outcomes:
        return MultipointPinballCertificate(
            status="refused", claim_tier="refused", loss="pinball", bound=None,
            quantile=float(quantile), selected_k=None, selected_score_cap=None,
            selected_center=None, selected_radius=None, selected_penalty=None,
            alpha_family=float(alpha_family), alpha_per_candidate=None,
            candidates=(), assumptions=(), diagnostics={},
            refusal={"reason": "partitions, caps, or block sizes were outcome-selected"})
    if set(partitions_by_k) != set(declared_discrepancy_by_k) or not partitions_by_k:
        raise ValueError("one declared discrepancy is required for every block size")
    caps = tuple(float(c) for c in score_caps)
    if (not caps or len(caps) != len(set(caps))
            or any(not math.isfinite(c) or c <= 0 for c in caps)):
        raise ValueError("unique positive finite score caps required")
    if shift is not None:
        s = np.asarray(shift, dtype=float).ravel()
        if len(s) != len(y) or not np.isfinite(s).all():
            raise ValueError("finite aligned shift required")
        y = y - s
    y = np.clip(y, lo, hi)
    width = hi - lo
    alpha = float(alpha_family / (len(partitions_by_k) * len(caps)))
    candidates = []
    for key in sorted(partitions_by_k):
        k = int(key)
        raw = np.asarray(partitions_by_k[key])
        idx = np.asarray(raw, dtype=int)
        if (raw.shape != idx.shape or not np.array_equal(raw, idx)
                or idx.ndim != 3 or idx.shape[0] < 1 or idx.shape[1] < 2
                or idx.shape[2] != k or idx.shape[1] * k != len(y)
                or idx.min() < 0 or idx.max() >= len(y)):
            raise ValueError("each route requires balanced integer partitions of the cohort")
        target = np.arange(len(y))
        if any(not np.array_equal(np.sort(partition.ravel()), target)
               for partition in idx):
            raise ValueError("every registered partition must cover each entity exactly once")
        declaration = float(declared_discrepancy_by_k[key])
        if not math.isfinite(declaration) or declaration < 0:
            raise ValueError("finite nonnegative discrepancy declarations required")
        raw_scores = _pinball_scores(y[idx].reshape(-1, k), quantile)
        for cap in caps:
            scores = np.minimum(raw_scores, cap)
            center = float(np.mean(scores))
            entity_scale = min(k * cap, max(quantile, 1-quantile) * width)
            radius = entity_scale * math.sqrt(
                math.log(1 / alpha) / (2 * len(y)))
            penalty = .5 * declaration
            candidates.append({
                "k": k, "n_partitions": int(idx.shape[0]),
                "n_block_scores": int(len(raw_scores)), "score_cap": cap,
                "center": center, "radius": float(radius),
                "declared_discrepancy": declaration,
                "penalty": float(penalty),
                "entity_bounded_difference_scale": float(entity_scale),
                "bound": float(max(0., center - radius - penalty)),
            })
    selected = max(candidates, key=lambda row: float(row["bound"]))
    return MultipointPinballCertificate(
        status="conditional", claim_tier="conditional", loss="pinball",
        bound=float(selected["bound"]), quantile=float(quantile),
        selected_k=int(selected["k"]),
        selected_score_cap=float(selected["score_cap"]),
        selected_center=float(selected["center"]),
        selected_radius=float(selected["radius"]),
        selected_penalty=float(selected["penalty"]),
        alpha_family=float(alpha_family), alpha_per_candidate=alpha,
        candidates=tuple(candidates),
        assumptions=(
            "entities are independent conditional on the frozen covariates",
            "every partition covers every entity exactly once and was frozen without outcomes",
            "the declared discrepancy upper-bounds the average barycentric residual-law discrepancy for every selected route",
            "the outcome clip and any shift were frozen without evaluation outcomes",
            "the complete block-size and score-cap portfolio was frozen",
        ),
        diagnostics={
            "concentration": "entity-level McDiarmid under overlapping balanced partitions",
            "overlap_across_partitions_allowed": True,
            "outcome_width": float(width),
            "penalty_coefficient": .5,
            "partition_provenance": str(partition_provenance),
            "discrepancy_provenance": str(discrepancy_provenance),
            "shift_provenance": str(shift_provenance or "none"),
        })


def _mixture_dkelly_lower_target(
        score_matrix: np.ndarray, *, cap: float, alpha: float,
        penalties: np.ndarray, partition_weights: np.ndarray | None = None,
        diversifications: int = 10,
        iterations: int = 60) -> float:
    """Invert an equal mixture of valid partition-level dKelly e-values."""
    scores = np.asarray(score_matrix, dtype=float)
    penalties = np.asarray(penalties, dtype=float).ravel()
    if (scores.ndim != 2 or scores.shape[0] != len(penalties)
            or scores.shape[1] < 2 or not np.isfinite(scores).all()
            or np.any(scores < 0) or np.any(scores > cap)
            or np.any(~np.isfinite(penalties)) or np.any(penalties < 0)
            or not 0 < alpha < 1 or cap <= 0 or diversifications < 1):
        raise ValueError("bounded partition score matrix and penalties required")
    z = scores / cap
    if partition_weights is None:
        weights = np.full(scores.shape[0], 1/scores.shape[0])
    else:
        weights = np.asarray(partition_weights, dtype=float).ravel()
        if (len(weights) != scores.shape[0] or not np.isfinite(weights).all()
                or np.any(weights < 0) or float(np.sum(weights)) <= 0):
            raise ValueError("nonnegative partition weights with positive sum required")
        weights = weights/np.sum(weights)
    fractions = np.arange(1, diversifications+1, dtype=float)/(diversifications+1)
    threshold = math.log(1/alpha)
    tiny = np.finfo(float).tiny

    def log_evalue(target: float) -> float:
        null_means = np.maximum((target+penalties)/cap, tiny)
        logs = np.sum(np.log(
            (1-fractions[:, None, None])
            + fractions[:, None, None]*z[None, :, :]/null_means[None, :, None]
        ), axis=2)
        component_weights = np.broadcast_to(
            weights[None, :]/diversifications, logs.shape)
        positive = component_weights > 0
        weighted_logs = logs[positive] + np.log(component_weights[positive])
        top = float(np.max(weighted_logs))
        return top + math.log(float(np.sum(np.exp(weighted_logs-top))))

    if log_evalue(0.) < threshold:
        return 0.
    rejected, accepted = 0., cap
    for _ in range(iterations):
        middle = (rejected+accepted)/2
        if log_evalue(middle) >= threshold:
            rejected = middle
        else:
            accepted = middle
    return float(max(0., np.nextafter(accepted, 0.)))


def mixture_dkelly_pinball_portfolio(
        outcomes, partitions_by_k: Mapping[int, np.ndarray], *,
        quantile: float, clip: tuple[float, float],
        declared_partition_discrepancies: Mapping[int, float | Sequence[float]],
        score_caps: Sequence[float], alpha_family: float = .05,
        partition_provenance: str | None,
        discrepancy_provenance: str | None,
        portfolio_frozen_before_outcomes: bool,
        shift=None, shift_provenance: str | None = None,
        partition_weights_by_k: Mapping[int, Sequence[float]] | None = None,
        weight_provenance: str | None = None,
        ) -> MultipointPinballCertificate:
    """Average partition-level dKelly e-values, then invert the mixture.

    Every partition contains disjoint entities, so its dKelly process is a
    valid e-value. Averaging over partitions remains valid despite arbitrary
    dependence between partitions. A scalar discrepancy is a uniform cap;
    a vector registers one cap per partition.
    """
    y = np.asarray(outcomes, dtype=float).ravel()
    lo, hi = map(float, clip)
    if (len(y) < 4 or not np.isfinite(y).all() or lo >= hi
            or not 0 < quantile < 1 or not 0 < alpha_family < 1):
        raise ValueError("finite outcomes, clip, quantile, and alpha required")
    refusal = None
    if not partition_provenance or not discrepancy_provenance:
        refusal = "partition and discrepancy provenance are required"
    elif not portfolio_frozen_before_outcomes:
        refusal = "partition mixture or caps were outcome-selected"
    if refusal:
        return MultipointPinballCertificate(
            status="refused", claim_tier="refused", loss="pinball", bound=None,
            quantile=float(quantile), selected_k=None, selected_score_cap=None,
            selected_center=None, selected_radius=None, selected_penalty=None,
            alpha_family=float(alpha_family), alpha_per_candidate=None,
            candidates=(), assumptions=(), diagnostics={},
            refusal={"reason": refusal})
    if shift is not None:
        if not shift_provenance:
            raise ValueError("shift provenance required when a shift is supplied")
        s = np.asarray(shift, dtype=float).ravel()
        if len(s) != len(y) or not np.isfinite(s).all():
            raise ValueError("finite aligned shift required")
        y = y-s
    y = np.clip(y, lo, hi)
    if set(partitions_by_k) != set(declared_partition_discrepancies) or not partitions_by_k:
        raise ValueError("partition discrepancy declarations must match block routes")
    if partition_weights_by_k is not None:
        if set(partition_weights_by_k) != set(partitions_by_k) or not weight_provenance:
            raise ValueError("weighted routes require aligned weights and provenance")
    caps = tuple(float(c) for c in score_caps)
    if (not caps or len(caps) != len(set(caps))
            or any(not math.isfinite(c) or c <= 0 for c in caps)):
        raise ValueError("unique positive finite score caps required")
    alpha = float(alpha_family/(len(partitions_by_k)*len(caps)))
    candidates = []
    for key in sorted(partitions_by_k):
        k = int(key)
        raw = np.asarray(partitions_by_k[key])
        idx = np.asarray(raw, dtype=int)
        if (raw.shape != idx.shape or not np.array_equal(raw, idx)
                or idx.ndim != 3 or idx.shape[1]*idx.shape[2] != len(y)
                or idx.shape[2] != k or idx.shape[1] < 2
                or any(not np.array_equal(np.sort(p.ravel()), np.arange(len(y)))
                       for p in idx)):
            raise ValueError("every route must contain balanced cohort partitions")
        declaration = np.asarray(declared_partition_discrepancies[key], dtype=float)
        if declaration.ndim == 0:
            declaration = np.full(idx.shape[0], float(declaration))
        declaration = declaration.ravel()
        if (len(declaration) != idx.shape[0] or not np.isfinite(declaration).all()
                or np.any(declaration < 0)):
            raise ValueError("one nonnegative discrepancy cap per partition required")
        raw_scores = np.stack([
            _pinball_scores(y[partition], quantile) for partition in idx])
        penalties = .5*declaration
        if partition_weights_by_k is None:
            weights = np.full(idx.shape[0], 1/idx.shape[0])
        else:
            weights = np.asarray(partition_weights_by_k[key], dtype=float).ravel()
            if (len(weights) != idx.shape[0] or not np.isfinite(weights).all()
                    or np.any(weights < 0) or float(np.sum(weights)) <= 0):
                raise ValueError("valid nonnegative weights required for every partition")
            weights = weights/np.sum(weights)
        for cap in caps:
            bounded = np.minimum(raw_scores, cap)
            bound = _mixture_dkelly_lower_target(
                bounded, cap=cap, alpha=alpha, penalties=penalties,
                partition_weights=weights)
            candidates.append({
                "k": k, "n_partitions": int(idx.shape[0]),
                "n_blocks_per_partition": int(idx.shape[1]),
                "score_cap": cap, "center": float(np.mean(bounded)),
                "radius": float(max(0., np.mean(bounded)-bound-np.mean(penalties))),
                "declared_discrepancy": float(np.max(declaration)),
                "penalty": float(np.dot(weights, penalties)), "bound": float(bound),
                "effective_partition_count": float(1/np.sum(weights*weights)),
            })
    selected = max(candidates, key=lambda row: float(row["bound"]))
    return MultipointPinballCertificate(
        status="conditional", claim_tier="conditional", loss="pinball",
        bound=float(selected["bound"]), quantile=float(quantile),
        selected_k=int(selected["k"]), selected_score_cap=float(selected["score_cap"]),
        selected_center=float(selected["center"]),
        selected_radius=float(selected["radius"]),
        selected_penalty=float(selected["penalty"]),
        alpha_family=float(alpha_family), alpha_per_candidate=alpha,
        candidates=tuple(candidates),
        assumptions=(
            "entities are independent within every registered partition",
            "partitions, caps, and alpha allocation were frozen without outcomes",
            "each partition discrepancy declaration upper-bounds its barycentric residual-law discrepancy",
            "the outcome clip and any shift were frozen without evaluation outcomes",
        ),
        diagnostics={
            "concentration": "equal mixture of partition-level diversified-Kelly e-values",
            "products_across_overlapping_partitions_used": False,
            "overlap_between_partitions_requires_independence": False,
            "penalty_coefficient": .5,
            "partition_provenance": str(partition_provenance),
            "discrepancy_provenance": str(discrepancy_provenance),
            "weight_provenance": str(weight_provenance or "equal weights"),
            "shift_provenance": str(shift_provenance or "none"),
        })
