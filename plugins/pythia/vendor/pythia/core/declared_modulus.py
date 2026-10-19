"""Conditional sensitivity curves under a declared residual-law modulus.

This module never estimates a modulus from the evaluation outcomes.  It turns
a predeclared feature metric and modulus into the barycentric discrepancy
penalty used by the near-replicate theorem.  The resulting endpoint is
conditional on that declaration and on an already-valid simultaneous lower
endpoint for the block SAA score.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DeclaredModulusProfile:
    status: str
    claim_tier: str
    loss: str
    moduli: tuple[float, ...]
    bounds: tuple[float, ...]
    penalties: tuple[float, ...]
    feature_dispersion_g: float
    feature_dispersion_h: float
    tipping_modulus: float
    metric_provenance: str
    modulus_provenance: str
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class DeclaredCurvaturePoint:
    """One conditional Jensen-gap correction.

    ``action_variance`` is the mean, over certified units, of the squared
    displacement between each unit's Bayes action and its block centre.
    It is a property of the latent conditional laws, not a sample variance of
    the observed outcomes.
    """

    action_variance: float
    curvature_bound: float
    action_mad: float | None
    max_action_deviation: float | None
    curvature_radius: float
    quadratic_penalty: float
    linear_penalty: float | None
    selected_penalty: float
    lower_bound: float


@dataclass(frozen=True)
class DeclaredCurvatureProfile:
    status: str
    claim_tier: str
    loss: str
    points: tuple[DeclaredCurvaturePoint, ...]
    retained_fraction: float
    action_dispersion_provenance: str
    curvature_provenance: str
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class BayesActionJensenDispersion:
    """Oracle/declaration geometry for the sharpest constant-action Taylor bound."""

    quadratic_center: float
    quadratic_penalty: float
    linear_center: float | None
    linear_penalty: float | None
    selected_penalty: float


def bayes_action_jensen_dispersion(
        actions, curvature_bounds, *, action_lipschitz: float | None = None,
        weights=None) -> BayesActionJensenDispersion:
    """Return optimized first- and second-order Bayes-action dispersions.

    This helper does not infer Bayes actions from outcomes.  ``actions`` and
    ``curvature_bounds`` must come from a simulator, a declared structural
    model, or an independent calibration protocol.  The quadratic centre is
    the curvature-weighted mean, which minimizes the valid Taylor penalty.
    The optional first-order centre is a weighted median.
    """
    a = np.asarray(actions, dtype=float).reshape(-1)
    kappa = np.asarray(curvature_bounds, dtype=float).reshape(-1)
    w = np.ones_like(a) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    if (len(a) == 0 or len(kappa) != len(a) or len(w) != len(a)
            or not np.isfinite(a).all() or not np.isfinite(kappa).all()
            or not np.isfinite(w).all() or (kappa < 0).any() or (w < 0).any()
            or w.sum() <= 0):
        raise ValueError("finite aligned actions, nonnegative curvatures, and positive total weight required")
    kw = kappa * w
    if kw.sum() == 0:
        quadratic_center = float(np.average(a, weights=w))
        quadratic = 0.0
    else:
        quadratic_center = float(np.dot(kw, a) / kw.sum())
        quadratic = float(np.dot(w, 0.5 * kappa * (a - quadratic_center) ** 2) / w.sum())
    linear_center = None
    linear = None
    if action_lipschitz is not None:
        lip = float(action_lipschitz)
        if not math.isfinite(lip) or lip < 0:
            raise ValueError("action Lipschitz constant must be finite and nonnegative")
        order = np.argsort(a, kind="stable")
        cumulative = np.cumsum(w[order])
        linear_center = float(a[order[np.searchsorted(cumulative, 0.5 * w.sum(), side="left")]])
        linear = float(lip * np.dot(w, np.abs(a - linear_center)) / w.sum())
    selected = quadratic if linear is None else min(quadratic, linear)
    return BayesActionJensenDispersion(
        quadratic_center=quadratic_center,
        quadratic_penalty=quadratic,
        linear_center=linear_center,
        linear_penalty=linear,
        selected_penalty=selected,
    )


def feature_barycentric_dispersion(features, blocks) -> tuple[float, float]:
    """Return the feature-only ``G`` and ``H`` geometry of disjoint blocks.

    ``G`` is the mean pairwise distance (including diagonal zeros). ``H`` is
    the mean squared per-unit average distance.  The feature scaling is part
    of the declared metric and must be frozen before evaluation outcomes.
    """
    z = np.asarray(features, dtype=float)
    idx = np.asarray(blocks, dtype=int)
    if z.ndim != 2 or len(z) < 2 or not np.isfinite(z).all():
        raise ValueError("finite two-dimensional features required")
    if (idx.ndim != 2 or idx.shape[0] < 2 or idx.shape[1] < 2
            or idx.size != np.unique(idx).size or idx.min() < 0
            or idx.max() >= len(z)):
        raise ValueError("at least two globally disjoint blocks required")
    zb = z[idx]
    distance = np.linalg.norm(zb[:, :, None, :] - zb[:, None, :, :], axis=-1)
    return float(distance.mean()), float(np.mean(distance.mean(axis=2) ** 2))


def _loss_coefficient(loss: str) -> tuple[str, float]:
    normalized = str(loss).upper()
    if normalized == "MAE":
        return "absolute", 1.0
    if normalized == "MSE":
        return "squared", 1.0
    if normalized.startswith("Q"):
        try:
            q = float(normalized[1:]) / 100.0
        except ValueError as exc:
            raise ValueError("quantile loss must be encoded as Qxx") from exc
        if 0 < q < 1:
            return "absolute", 0.5
    raise ValueError("supported losses are MAE, MSE, and Qxx pinball")


def _linear_action_lipschitz(loss: str) -> float | None:
    """Lipschitz constant of scalar loss in its action argument."""
    normalized = str(loss).upper()
    if normalized == "MAE":
        return 1.0
    if normalized == "MSE":
        return None
    if normalized.startswith("Q"):
        try:
            q = float(normalized[1:]) / 100.0
        except ValueError as exc:
            raise ValueError("quantile loss must be encoded as Qxx") from exc
        if 0 < q < 1:
            return max(q, 1.0 - q)
    raise ValueError("supported losses are MAE, MSE, and Qxx pinball")


def declared_action_curvature_profile(
        *, lower_score_mean: float, retained_fraction: float, loss: str,
        declared_action_variances: Iterable[float],
        curvature_bounds: Iterable[float],
        action_dispersion_provenance: str, curvature_provenance: str,
        declared_action_mads: Iterable[float] | None = None,
        max_action_deviations: Iterable[float] | None = None,
        curvature_radius: float = math.inf) -> DeclaredCurvatureProfile:
    """Apply a second-order Jensen-gap penalty to a block-SAA endpoint.

    Let ``a_j*`` be unit ``j``'s scalar Bayes action and ``abar`` its block
    average.  If every unit's *expected* loss has action curvature at most
    ``kappa`` on the segment joining ``a_j*`` and ``abar``, then

    ``mixture Bayes risk - mean unit Bayes risk <= kappa * V / 2``,

    where ``V = mean_j (a_j* - abar)^2``.  For MAE and pinball, a declared
    mean absolute displacement also supplies the first-order Lipschitz bound;
    the smaller of the two valid penalties is used.

    This function estimates neither ``V`` nor ``kappa``.  It compiles a
    simultaneous sensitivity profile from declarations made independently of
    evaluation outcomes.  A finite local curvature radius is accepted only
    when a matching maximum action displacement is supplied at every point.
    """
    lcb = float(lower_score_mean)
    retained = float(retained_fraction)
    variances = tuple(float(v) for v in declared_action_variances)
    kappas = tuple(float(v) for v in curvature_bounds)
    mads = None if declared_action_mads is None else tuple(
        float(v) for v in declared_action_mads)
    maxima = None if max_action_deviations is None else tuple(
        float(v) for v in max_action_deviations)
    radius = float(curvature_radius)
    if not variances or len(kappas) != len(variances):
        raise ValueError("action variance and curvature declarations must be nonempty and aligned")
    if (not math.isfinite(lcb) or lcb < 0 or not math.isfinite(retained)
            or not 0 <= retained <= 1
            or any(not math.isfinite(v) or v < 0 for v in variances + kappas)):
        raise ValueError("finite nonnegative endpoint, variances, and curvatures required")
    if mads is not None and (len(mads) != len(variances)
                             or any(not math.isfinite(v) or v < 0 for v in mads)):
        raise ValueError("action MAD declarations must be finite, nonnegative, and aligned")
    if maxima is not None and (len(maxima) != len(variances)
                               or any(not math.isfinite(v) or v < 0 for v in maxima)):
        raise ValueError("maximum action deviations must be finite, nonnegative, and aligned")
    if math.isnan(radius) or radius <= 0:
        raise ValueError("curvature radius must be positive")
    if math.isfinite(radius):
        if maxima is None:
            raise ValueError("local curvature requires maximum action deviations")
        if any(v > radius for v in maxima):
            raise ValueError("declared action displacement leaves the curvature domain")
    if not action_dispersion_provenance or not curvature_provenance:
        raise ValueError("action-dispersion and curvature provenance are required")

    lipschitz = _linear_action_lipschitz(loss)
    if str(loss).upper() == "MSE" and any(k < 2.0 for k in kappas):
        raise ValueError("squared loss has action curvature 2 under the MSE convention")
    points = []
    for i, (variance, kappa) in enumerate(zip(variances, kappas)):
        quadratic = 0.5 * kappa * variance
        linear = None if mads is None or lipschitz is None else lipschitz * mads[i]
        penalty = quadratic if linear is None else min(quadratic, linear)
        points.append(DeclaredCurvaturePoint(
            action_variance=variance,
            curvature_bound=kappa,
            action_mad=None if mads is None else mads[i],
            max_action_deviation=None if maxima is None else maxima[i],
            curvature_radius=radius,
            quadratic_penalty=quadratic,
            linear_penalty=linear,
            selected_penalty=penalty,
            lower_bound=retained * max(0.0, lcb - penalty),
        ))
    scope = ("globally in the action argument" if math.isinf(radius)
             else "on every segment joining a unit Bayes action to its block centre")
    return DeclaredCurvatureProfile(
        status="sensitivity_only", claim_tier="conditional", loss=str(loss),
        points=tuple(points), retained_fraction=retained,
        action_dispersion_provenance=str(action_dispersion_provenance),
        curvature_provenance=str(curvature_provenance),
        assumptions=(
            "blocks and the score lower endpoint were frozen without evaluation outcomes",
            "declared action dispersion upper-bounds latent within-block Bayes-action drift",
            f"declared curvature upper-bounds each unit expected-loss curvature {scope}",
            "declarations and their provenance are independent of evaluation outcomes",
            "the score endpoint and Jensen-gap correction target the same blocks and loss scaling",
        ),
    )


def declared_feature_modulus_profile(
        *, lower_score_mean: float, retained_fraction: float, loss: str,
        feature_dispersion_g: float, feature_dispersion_h: float,
        declared_moduli: Iterable[float], metric_provenance: str,
        modulus_provenance: str) -> DeclaredModulusProfile:
    """Compile a simultaneous conditional curve over declared moduli.

    For MAE/pinball, the penalty is ``c * Lambda * G`` with ``c=1`` for
    MAE and ``c=1/2`` for every pinball level.  For MSE it is
    ``Lambda_2**2 * H``.  No alpha is spent by evaluating more points on this
    curve: all points transform the same pre-existing score event.
    """
    lcb = float(lower_score_mean)
    retained = float(retained_fraction)
    g, h = float(feature_dispersion_g), float(feature_dispersion_h)
    values = tuple(float(v) for v in declared_moduli)
    if (not math.isfinite(lcb) or lcb < 0 or not math.isfinite(retained)
            or not 0 <= retained <= 1 or not math.isfinite(g) or g < 0
            or not math.isfinite(h) or h < 0 or not values
            or any(not math.isfinite(v) or v < 0 for v in values)):
        raise ValueError("finite nonnegative endpoint, geometry, and moduli required")
    if not metric_provenance or not modulus_provenance:
        raise ValueError("metric and modulus provenance are required")
    geometry, coefficient = _loss_coefficient(loss)
    if geometry == "squared":
        penalties = tuple(v * v * h for v in values)
        tipping = math.sqrt(lcb / h) if h > 0 else math.inf
    else:
        penalties = tuple(coefficient * v * g for v in values)
        tipping = lcb / (coefficient * g) if g > 0 else math.inf
    bounds = tuple(retained * max(0.0, lcb - p) for p in penalties)
    return DeclaredModulusProfile(
        status="sensitivity_only", claim_tier="conditional", loss=str(loss),
        moduli=values, bounds=bounds, penalties=penalties,
        feature_dispersion_g=g, feature_dispersion_h=h,
        tipping_modulus=float(tipping),
        metric_provenance=str(metric_provenance),
        modulus_provenance=str(modulus_provenance),
        assumptions=(
            "blocks are globally disjoint and frozen without evaluation outcomes",
            "the score lower endpoint is simultaneous for every reported candidate",
            "the registered feature metric and scaling were frozen without evaluation outcomes",
            "the declared modulus upper-bounds the relevant within-block residual-law discrepancy",
            "any residual shift was frozen independently of evaluation outcomes",
        ),
    )
