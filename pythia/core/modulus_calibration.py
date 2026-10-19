"""Non-oracle barycentric-discrepancy calibration from repeated environments.

Each environment row supplies one jointly coupled outcome for every registered
unit.  Dependence within a row is arbitrary.  Rows must be independent and the
coordinate marginals must either be the target unit-conditional laws or come
from an externally justified upper-envelope coupling.  Ordinary observational
rows are not repeated environments.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np

from .concentration import bounded_upper_mean
from .declared_modulus import feature_barycentric_dispersion
from .truncated_saa import dkelly_lower_mean


SUPPORTED_UPPER_METHODS = ("Hoeffding", "empirical_Bernstein", "Bentkus", "dKelly")
REPEATED_MODULUS_ROUTER_V1 = {
    "absolute": ("dKelly",),
    "squared": ("Bentkus",),
}
SUPPORTED_MARGINAL_CONTRACTS = (
    "target_unit_conditional",
    "external_upper_envelope",
)


@dataclass(frozen=True)
class RepeatedEnvironmentDiscrepancyCalibration:
    status: str
    claim_tier: str
    absolute_discrepancy_cap: float | None
    squared_discrepancy_cap: float | None
    effective_absolute_modulus_cap: float | None
    effective_squared_modulus_cap: float | None
    feature_dispersion_g: float | None
    feature_dispersion_h: float | None
    alpha_family: float
    alpha_per_endpoint: float | None
    n_environments: int
    marginal_contract: str | None
    candidates: Mapping[str, tuple[Mapping[str, float | str], ...]]
    assumptions: tuple[str, ...]
    diagnostics: Mapping[str, object]
    refusal: Mapping[str, str] | None = None


@dataclass(frozen=True)
class CalibratedBarycentricCertificate:
    status: str
    claim_tier: str
    loss: str
    bound: float | None
    lower_score_mean: float | None
    penalty_cap: float | None
    retained_fraction: float | None
    alpha_family: float
    score_failure_probability: float
    calibration_failure_probability: float
    event_ids: tuple[str, ...]
    assumptions: tuple[str, ...]
    diagnostics: Mapping[str, object]
    refusal: Mapping[str, str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _content_hash(array) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    payload = (str(value.shape) + "|" + str(value.dtype) + "|").encode() + value.tobytes()
    return hashlib.sha256(payload).hexdigest()


def repeated_environment_barycentric_costs(
        replicates, blocks, *, clip: tuple[float, float], shift=None
        ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return absolute and squared barycentric costs for every environment."""
    y = np.asarray(replicates, dtype=float)
    raw_idx = np.asarray(blocks)
    idx = np.asarray(blocks, dtype=int)
    lo, hi = map(float, clip)
    if (y.ndim != 2 or y.shape[0] < 2 or y.shape[1] < 2
            or not np.isfinite(y).all() or not np.isfinite([lo, hi]).all()
            or lo >= hi):
        raise ValueError("finite repeated-environment matrix and increasing clip required")
    if (raw_idx.shape != idx.shape or not np.array_equal(raw_idx, idx)
            or idx.ndim != 2 or idx.shape[0] < 2 or idx.shape[1] < 2
            or idx.size != np.unique(idx).size or idx.min() < 0
            or idx.max() >= y.shape[1]):
        raise ValueError("at least two globally disjoint integer blocks required")
    if shift is not None:
        s = np.asarray(shift, dtype=float).ravel()
        if len(s) != y.shape[1] or not np.isfinite(s).all():
            raise ValueError("finite unit-aligned shift required")
        y = y - s[None, :]
    values = np.clip(y, lo, hi)[:, idx]
    deviation = values - values.mean(axis=2, keepdims=True)
    absolute = np.mean(np.abs(deviation), axis=(1, 2))
    squared = np.mean(deviation * deviation, axis=(1, 2))
    width = hi - lo
    absolute_range, squared_range = width / 2, width * width / 4
    if (np.any(absolute > absolute_range + 1e-10)
            or np.any(squared > squared_range + 1e-10)):
        raise AssertionError("barycentric cost exceeded its deterministic range")
    return absolute, squared, float(absolute_range), float(squared_range)


def support_aware_barycentric_cost_ranges(
        unit_supports, blocks, *, clip: tuple[float, float],
        maximum_block_size: int = 16) -> tuple[float, float]:
    """Exact aggregate cost ranges over registered per-unit support boxes.

    Both barycentric costs are convex in the block outcomes, so their maximum
    over a rectangular support is attained at a vertex.  Blocks are disjoint,
    hence the maximum of their average is the average of block maxima.
    Enumeration is fail-closed above ``maximum_block_size``.
    """
    support = np.asarray(unit_supports, dtype=float)
    raw_idx = np.asarray(blocks)
    idx = np.asarray(blocks, dtype=int)
    lo, hi = map(float, clip)
    if (support.ndim != 2 or support.shape[1] != 2
            or not np.isfinite(support).all()
            or np.any(support[:, 0] > support[:, 1])
            or np.any(support[:, 0] < lo) or np.any(support[:, 1] > hi)):
        raise ValueError("finite ordered unit supports inside the clip required")
    if (raw_idx.shape != idx.shape or not np.array_equal(raw_idx, idx)
            or idx.ndim != 2 or idx.shape[0] < 2 or idx.shape[1] < 2
            or idx.size != np.unique(idx).size or idx.min() < 0
            or idx.max() >= len(support)):
        raise ValueError("globally disjoint integer blocks aligned to supports required")
    k = idx.shape[1]
    if (not isinstance(maximum_block_size, int) or maximum_block_size < 2
            or k > maximum_block_size):
        raise ValueError("block too large for registered exact vertex enumeration")
    bits = ((np.arange(1 << k)[:, None] >> np.arange(k)) & 1).astype(bool)
    absolute_maxima, squared_maxima = [], []
    for block in idx:
        interval = support[block]
        vertices = np.where(bits, interval[None, :, 1], interval[None, :, 0])
        deviation = vertices - vertices.mean(axis=1, keepdims=True)
        absolute_maxima.append(float(np.max(np.mean(np.abs(deviation), axis=1))))
        squared_maxima.append(float(np.max(np.mean(deviation * deviation, axis=1))))
    return float(np.mean(absolute_maxima)), float(np.mean(squared_maxima))


def calibrate_repeated_environment_discrepancy(
        replicates, blocks, features, *, clip: tuple[float, float],
        alpha_family: float = .05,
        methods: Sequence[str] | Mapping[str, Sequence[str]] = SUPPORTED_UPPER_METHODS,
        marginal_contract: str | None,
        coupling_provenance: str | None,
        metric_provenance: str | None,
        portfolio_frozen_before_replicates: bool,
        shift=None, shift_provenance: str | None = None,
        unit_supports=None, support_provenance: str | None = None,
        ) -> RepeatedEnvironmentDiscrepancyCalibration:
    """Calibrate aggregate discrepancy and its effective feature modulus.

    The caps target the average barycentric cost under the supplied coupling.
    Since the loss theorem accepts *any* coupling with the correct marginals,
    these caps upper-bound a valid MAE/pinball or MSE transfer penalty.  The
    effective moduli are aggregate ratios ``P/G`` and ``sqrt(P2/H)``; they are
    not uniform pairwise Lipschitz constants.
    """
    if marginal_contract not in SUPPORTED_MARGINAL_CONTRACTS:
        return RepeatedEnvironmentDiscrepancyCalibration(
            status="refused", claim_tier="refused",
            absolute_discrepancy_cap=None, squared_discrepancy_cap=None,
            effective_absolute_modulus_cap=None,
            effective_squared_modulus_cap=None,
            feature_dispersion_g=None, feature_dispersion_h=None,
            alpha_family=float(alpha_family), alpha_per_endpoint=None,
            n_environments=int(np.asarray(replicates).shape[0]) if np.asarray(replicates).ndim else 0,
            marginal_contract=None, candidates={}, assumptions=(), diagnostics={},
            refusal={"reason": "a supported repeated-environment marginal contract is required",
                     "referral": "use target_unit_conditional or a justified external_upper_envelope"})
    if not coupling_provenance or not metric_provenance:
        return RepeatedEnvironmentDiscrepancyCalibration(
            status="refused", claim_tier="refused",
            absolute_discrepancy_cap=None, squared_discrepancy_cap=None,
            effective_absolute_modulus_cap=None,
            effective_squared_modulus_cap=None,
            feature_dispersion_g=None, feature_dispersion_h=None,
            alpha_family=float(alpha_family), alpha_per_endpoint=None,
            n_environments=int(np.asarray(replicates).shape[0]),
            marginal_contract=marginal_contract, candidates={}, assumptions=(), diagnostics={},
            refusal={"reason": "coupling and metric provenance are required",
                     "referral": "hash and register the environment and feature-metric protocols"})
    if shift is not None and not shift_provenance:
        raise ValueError("shift provenance is required when a shift is supplied")
    if unit_supports is not None and not support_provenance:
        raise ValueError("support provenance is required with unit supports")
    if not portfolio_frozen_before_replicates:
        return RepeatedEnvironmentDiscrepancyCalibration(
            status="refused", claim_tier="refused",
            absolute_discrepancy_cap=None, squared_discrepancy_cap=None,
            effective_absolute_modulus_cap=None,
            effective_squared_modulus_cap=None,
            feature_dispersion_g=None, feature_dispersion_h=None,
            alpha_family=float(alpha_family), alpha_per_endpoint=None,
            n_environments=int(np.asarray(replicates).shape[0]),
            marginal_contract=marginal_contract, candidates={}, assumptions=(), diagnostics={},
            refusal={"reason": "methods or geometry were selected after seeing replicate outcomes",
                     "referral": "freeze the blocks, metric, clip, methods, and alpha ledger first"})
    if isinstance(methods, Mapping):
        if set(methods) != {"absolute", "squared"}:
            raise ValueError("family-specific methods require absolute and squared entries")
        methods_by_family = {
            family: tuple(map(str, methods[family]))
            for family in ("absolute", "squared")}
    else:
        method_tuple = tuple(map(str, methods))
        methods_by_family = {family: method_tuple
                             for family in ("absolute", "squared")}
    if (not 0 < alpha_family < 1
            or any(not route or len(route) != len(set(route))
                   or any(m not in SUPPORTED_UPPER_METHODS for m in route)
                   for route in methods_by_family.values())):
        raise ValueError("valid alpha and unique supported methods required")
    absolute, squared, absolute_range, squared_range = (
        repeated_environment_barycentric_costs(
            replicates, blocks, clip=clip, shift=shift))
    if unit_supports is not None:
        support = np.asarray(unit_supports, dtype=float)
        transformed = np.asarray(replicates, dtype=float)
        if shift is not None:
            transformed = transformed - np.asarray(shift, dtype=float)[None, :]
        transformed = np.clip(transformed, *map(float, clip))
        if (support.shape != (transformed.shape[1], 2)
                or np.any(transformed < support[None, :, 0] - 1e-10)
                or np.any(transformed > support[None, :, 1] + 1e-10)):
            raise ValueError("replicates violate the registered post-shift unit supports")
        absolute_range, squared_range = support_aware_barycentric_cost_ranges(
            support, blocks, clip=clip)
        if (np.any(absolute > absolute_range + 1e-10)
                or np.any(squared > squared_range + 1e-10)):
            raise AssertionError("observed cost exceeds support-aware range")
    g, h = feature_barycentric_dispersion(features, blocks)
    alpha = float(alpha_family / sum(map(len, methods_by_family.values())))
    candidates = {}
    for name, values, upper in (
            ("absolute", absolute, absolute_range),
            ("squared", squared, squared_range)):
        rows = tuple({"method": method,
                      "upper": bounded_upper_mean(
                          values, upper_bound=upper,
                          alpha=alpha, method=method)}
                     for method in methods_by_family[name])
        candidates[name] = rows
    absolute_cap = min(float(row["upper"]) for row in candidates["absolute"])
    squared_cap = min(float(row["upper"]) for row in candidates["squared"])
    absolute_modulus = absolute_cap / g if g > 0 else math.inf
    squared_modulus = math.sqrt(squared_cap / h) if h > 0 else math.inf
    exact = marginal_contract == "target_unit_conditional"
    return RepeatedEnvironmentDiscrepancyCalibration(
        status="conditional", claim_tier="conditional",
        absolute_discrepancy_cap=float(absolute_cap),
        squared_discrepancy_cap=float(squared_cap),
        effective_absolute_modulus_cap=float(absolute_modulus),
        effective_squared_modulus_cap=float(squared_modulus),
        feature_dispersion_g=g, feature_dispersion_h=h,
        alpha_family=float(alpha_family), alpha_per_endpoint=alpha,
        n_environments=int(len(absolute)), marginal_contract=marginal_contract,
        candidates=candidates,
        assumptions=(
            "environment rows are independent and identically distributed",
            "the registered marginal contract upper-bounds the target unit-law barycentric cost",
            "blocks, feature metric, clip, shift, methods, and alpha allocation were frozen before replicate outcomes",
            "downstream evaluation outcomes are independent of the calibration environments",
            "the external upper-envelope contract, when used, is a structural assumption rather than a data-derived fact",
        ),
        diagnostics={
            "empirical_absolute_cost": float(np.mean(absolute)),
            "empirical_squared_cost": float(np.mean(squared)),
            "absolute_cost_range": absolute_range,
            "squared_cost_range": squared_range,
            "metric_provenance": str(metric_provenance),
            "block_design_hash": _content_hash(np.asarray(blocks, dtype=np.int64)),
            "coupling_provenance": str(coupling_provenance),
            "shift_provenance": str(shift_provenance or "none"),
            "support_provenance": str(support_provenance or "global clip only"),
            "support_aware_ranges_used": unit_supports is not None,
            "effective_modulus_is_uniform_pairwise": False,
            "methods_by_family": methods_by_family,
            "exact_target_replicates_available": exact,
            "resource_fairness_warning": (
                "the same exact target replicates support direct repeated-environment SAA; "
                "report that stronger comparator" if exact else
                "validity depends on the registered external upper-envelope transfer"),
        },
    )


def calibrate_repeated_environment_discrepancy_v1(
        replicates, blocks, features, *, clip: tuple[float, float],
        alpha_family: float = .05, marginal_contract: str | None,
        coupling_provenance: str | None, metric_provenance: str | None,
        portfolio_frozen_before_replicates: bool,
        shift=None, shift_provenance: str | None = None,
        unit_supports=None, support_provenance: str | None = None,
        ) -> RepeatedEnvironmentDiscrepancyCalibration:
    """Frozen V1 loss router selected before confirmation seeds 134001+."""
    return calibrate_repeated_environment_discrepancy(
        replicates, blocks, features, clip=clip,
        alpha_family=alpha_family, methods=REPEATED_MODULUS_ROUTER_V1,
        marginal_contract=marginal_contract,
        coupling_provenance=coupling_provenance,
        metric_provenance=metric_provenance,
        portfolio_frozen_before_replicates=portfolio_frozen_before_replicates,
        shift=shift, shift_provenance=shift_provenance,
        unit_supports=unit_supports, support_provenance=support_provenance)


def calibrated_barycentric_lower_bound(
        *, lower_score_mean: float, retained_fraction: float, loss: str,
        calibration: RepeatedEnvironmentDiscrepancyCalibration,
        score_failure_probability: float, score_event_id: str,
        calibration_event_id: str, score_block_design_hash: str,
        score_provenance: str) -> CalibratedBarycentricCertificate:
    """Compose a score LCB and repeated-environment penalty UCB.

    The two failure probabilities are combined by a union bound; independence
    of the score and calibration events is not needed for coverage.  The
    calibration protocol itself still requires downstream evaluation outcomes
    to be independent because its block/shift design must be frozen.
    """
    score_alpha = float(score_failure_probability)
    total_alpha = score_alpha + float(calibration.alpha_family)
    if calibration.status != "conditional":
        return CalibratedBarycentricCertificate(
            status="refused", claim_tier="refused", loss=str(loss), bound=None,
            lower_score_mean=None, penalty_cap=None, retained_fraction=None,
            alpha_family=total_alpha, score_failure_probability=score_alpha,
            calibration_failure_probability=float(calibration.alpha_family),
            event_ids=(), assumptions=(), diagnostics={},
            refusal={"reason": "completed conditional calibration required"})
    lcb, retained = float(lower_score_mean), float(retained_fraction)
    if (not math.isfinite(lcb) or lcb < 0 or not math.isfinite(retained)
            or not 0 <= retained <= 1 or not 0 < score_alpha < 1
            or not 0 < total_alpha < 1 or not score_event_id
            or not calibration_event_id or score_event_id == calibration_event_id
            or not score_provenance):
        raise ValueError("valid score endpoint, distinct events, and provenance required")
    expected_hash = str(calibration.diagnostics.get("block_design_hash", ""))
    if not expected_hash or score_block_design_hash != expected_hash:
        return CalibratedBarycentricCertificate(
            status="refused", claim_tier="refused", loss=str(loss), bound=None,
            lower_score_mean=lcb, penalty_cap=None, retained_fraction=retained,
            alpha_family=total_alpha, score_failure_probability=score_alpha,
            calibration_failure_probability=float(calibration.alpha_family),
            event_ids=(score_event_id, calibration_event_id), assumptions=(),
            diagnostics={"calibration_block_design_hash": expected_hash,
                         "score_block_design_hash": score_block_design_hash},
            refusal={"reason": "score and calibration block designs differ"})
    normalized = str(loss).upper()
    if normalized == "MSE":
        penalty = calibration.squared_discrepancy_cap
        family = "squared"
    elif normalized == "MAE":
        penalty = calibration.absolute_discrepancy_cap
        family = "absolute"
    elif normalized.startswith("Q"):
        try:
            q = float(normalized[1:]) / 100
        except ValueError as exc:
            raise ValueError("quantile loss must be encoded as Qxx") from exc
        if not 0 < q < 1:
            raise ValueError("pinball level must lie in (0,1)")
        penalty = .5 * float(calibration.absolute_discrepancy_cap)
        family = "absolute_half"
    else:
        raise ValueError("supported losses are MAE, MSE, and Qxx pinball")
    bound = retained * max(0., lcb - float(penalty))
    return CalibratedBarycentricCertificate(
        status="conditional", claim_tier="conditional", loss=str(loss),
        bound=float(bound), lower_score_mean=lcb, penalty_cap=float(penalty),
        retained_fraction=retained, alpha_family=total_alpha,
        score_failure_probability=score_alpha,
        calibration_failure_probability=float(calibration.alpha_family),
        event_ids=(score_event_id, calibration_event_id),
        assumptions=(
            "the score endpoint lower-bounds the mean empirical-optimum block score",
            "the repeated-environment calibration upper-bounds the matching block-design barycentric penalty",
            "score and calibration candidates were frozen before their respective outcomes",
            "the marginal, coupling, metric, shift, and support contracts recorded by calibration hold",
        ),
        diagnostics={
            "penalty_family": family,
            "block_design_hash": expected_hash,
            "score_provenance": score_provenance,
            "alpha_combination": "union bound over distinct score and calibration events",
        })
