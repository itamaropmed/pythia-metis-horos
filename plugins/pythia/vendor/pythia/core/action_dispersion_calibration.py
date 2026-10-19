"""Finite-sample Bayes-action dispersion caps from repeated environments.

The calibration is distribution-free for scalar quantile actions.  It first
builds simultaneous order-statistic intervals for every used unit, then
maximizes the relevant within-block action dispersions over the resulting box.
No predictor is trained and no certification outcome is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

import numpy as np
from scipy.stats import binom

from .declared_modulus import bayes_action_jensen_dispersion


@dataclass(frozen=True)
class RepeatedActionDispersionCalibration:
    status: str
    claim_tier: str
    quantile: float
    alpha: float
    n_environments: int
    n_units: int
    lower_rank: int
    upper_rank: int
    action_variance_cap: float
    action_mad_cap: float
    max_action_deviation_cap: float
    curvature_penalty_cap: float | None
    support: tuple[float, float]
    environment_provenance: str
    support_provenance: str
    assumptions: tuple[str, ...]


def _simultaneous_quantile_ranks(n: int, q: float, alpha: float, m: int) -> tuple[int, int]:
    """Return 1-based ranks with Bonferroni joint coverage over ``m`` units."""
    tail = alpha / (2.0 * m)
    lower = int(binom.ppf(tail, n, q))
    upper = int(binom.ppf(1.0 - tail, n, q)) + 1
    return max(0, lower), min(n + 1, upper)


def _box_dispersion_caps(lo, hi, kappa=None):
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    if len(lo) > 16:
        raise ValueError("exact interval-vertex calibration supports block size at most 16")
    best_var = best_mad = best_max = best_curv = 0.0
    for bits in product((0, 1), repeat=len(lo)):
        a = np.where(np.asarray(bits, bool), hi, lo)
        centre = float(a.mean())
        best_var = max(best_var, float(np.mean((a - centre) ** 2)))
        best_mad = max(best_mad, float(np.mean(np.abs(a - np.median(a)))))
        best_max = max(best_max, float(np.max(np.abs(a - centre))))
        if kappa is not None:
            best_curv = max(
                best_curv,
                bayes_action_jensen_dispersion(a, kappa).quadratic_penalty,
            )
    return best_var, best_mad, best_max, (None if kappa is None else best_curv)


def calibrate_repeated_action_dispersion(
        replicates, blocks, *, quantile: float, alpha: float,
        support: tuple[float, float], environment_provenance: str,
        support_provenance: str, curvature_bounds=None,
        curvature_provenance: str | None = None,
        independence_attested: bool = False) -> RepeatedActionDispersionCalibration:
    """Upper-bound latent quantile-action dispersion on frozen blocks.

    Rows of ``replicates`` must be independent environments measuring the same
    fixed units.  Dependence across units within a row is allowed.  ``support``
    must be a hard bound for every unit outcome and therefore for its quantile.
    The returned caps hold jointly with probability at least ``1-alpha`` under
    the repeated-environment contract.
    """
    y = np.asarray(replicates, dtype=float)
    idx = np.asarray(blocks, dtype=int)
    q = float(quantile)
    error = float(alpha)
    lower_support, upper_support = map(float, support)
    if (y.ndim != 2 or y.shape[0] < 2 or not np.isfinite(y).all()
            or idx.ndim != 2 or idx.shape[0] < 1 or idx.shape[1] < 2
            or idx.size != np.unique(idx).size or idx.min() < 0
            or idx.max() >= y.shape[1]):
        raise ValueError("finite repeated-environment matrix and globally disjoint blocks required")
    if not 0 < q < 1 or not 0 < error < 1:
        raise ValueError("quantile and alpha must lie strictly between zero and one")
    if (not math.isfinite(lower_support) or not math.isfinite(upper_support)
            or lower_support >= upper_support
            or (y < lower_support).any() or (y > upper_support).any()):
        raise ValueError("finite hard support must contain every calibration outcome")
    if not environment_provenance or not support_provenance:
        raise ValueError("environment and support provenance are required")
    if not independence_attested:
        raise ValueError("independence of repeated environment rows must be attested")

    used = idx.reshape(-1)
    lower_rank, upper_rank = _simultaneous_quantile_ranks(
        y.shape[0], q, error, len(used))
    ordered = np.sort(y[:, used], axis=0)
    lower = (np.full(len(used), lower_support) if lower_rank == 0
             else ordered[lower_rank - 1])
    upper = (np.full(len(used), upper_support) if upper_rank == y.shape[0] + 1
             else ordered[upper_rank - 1])
    lower_by_unit = np.full(y.shape[1], np.nan)
    upper_by_unit = np.full(y.shape[1], np.nan)
    lower_by_unit[used], upper_by_unit[used] = lower, upper

    kappa = None
    if curvature_bounds is not None:
        kappa = np.asarray(curvature_bounds, dtype=float).reshape(-1)
        if (len(kappa) != y.shape[1] or not np.isfinite(kappa).all()
                or (kappa < 0).any() or not curvature_provenance):
            raise ValueError("finite aligned curvature bounds and provenance are required")
    block_caps = []
    for block in idx:
        block_caps.append(_box_dispersion_caps(
            lower_by_unit[block], upper_by_unit[block],
            None if kappa is None else kappa[block]))
    variance, mad, maximum = (float(np.mean([row[j] for row in block_caps]))
                              for j in range(3))
    curvature = (None if kappa is None
                 else float(np.mean([row[3] for row in block_caps])))
    return RepeatedActionDispersionCalibration(
        status="conditional", claim_tier="conditional", quantile=q,
        alpha=error, n_environments=int(y.shape[0]), n_units=int(len(used)),
        lower_rank=lower_rank, upper_rank=upper_rank,
        action_variance_cap=variance, action_mad_cap=mad,
        max_action_deviation_cap=maximum,
        curvature_penalty_cap=curvature,
        support=(lower_support, upper_support),
        environment_provenance=str(environment_provenance),
        support_provenance=str(support_provenance),
        assumptions=(
            "environment rows are mutually independent measurements of the same fixed units",
            "arbitrary dependence across units within each environment row is permitted",
            "blocks and quantile level were frozen independently of calibration and certification outcomes",
            "the declared hard support contains every target-law outcome",
            "curvature bounds, when supplied, hold on every action segment inside the simultaneous intervals",
        ),
    )
