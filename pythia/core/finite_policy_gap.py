"""Loss-relative refinement-gap bounds for finite decision-policy classes.

Unlike a joint-multinomial confidence ball, this route does not pay for the
outcome alphabet.  It uniformly controls empirical risks over every policy
from coarse groups or refined ``(group,state)`` cells to a finite action set.
For bounded loss and iid rows,

    Delta(P) <= Delta(P_hat) + eps_coarse + eps_fine,

where each epsilon is a one-sided Hoeffding/finite-class radius.  Alpha is
allocated between the two policy classes using a deterministic optimization
that depends only on registered class sizes and sample size.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Mapping, Sequence

import math
import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class FinitePolicyGapUpperBound:
    status: str
    claim_tier: str
    loss: str
    gap_cap: float | None
    plugin_gap: float | None
    plugin_coarse_risk: float | None
    plugin_fine_risk: float | None
    alpha: float
    n: int
    loss_range: float | None
    action_count: int
    coarse_cell_count: int
    fine_cell_count: int
    assumptions: tuple[str, ...]
    diagnostics: Mapping[str, object]
    refusal: Mapping[str, str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


FINITE_GAP_LOSS_ROUTER_V1 = {
    "MAE": "finite_policy_hoeffding",
    "MSE": "finite_policy_hoeffding",
    "Q10": "coordinate_empirical_bernstein",
    "Q50": "finite_policy_hoeffding",
    "Q90": "coordinate_empirical_bernstein",
    "W_EXACT_MISS": "finite_policy_hoeffding",
}


def _alpha_allocation(alpha, log_coarse, log_fine, amplitude, n):
    def objective(alpha_coarse):
        alpha_fine = alpha - alpha_coarse
        coarse = amplitude * math.sqrt((log_coarse + math.log(1. / alpha_coarse)) / (2. * n))
        fine = amplitude * math.sqrt((log_fine + math.log(1. / alpha_fine)) / (2. * n))
        return coarse + fine
    result = minimize_scalar(
        objective, bounds=(alpha * 1e-8, alpha * (1. - 1e-8)),
        method="bounded", options={"xatol": max(1e-15, alpha * 1e-12)})
    alpha_coarse = float(result.x)
    alpha_fine = float(alpha - alpha_coarse)
    coarse = amplitude * math.sqrt((log_coarse + math.log(1. / alpha_coarse)) / (2. * n))
    fine = amplitude * math.sqrt((log_fine + math.log(1. / alpha_fine)) / (2. * n))
    return alpha_coarse, alpha_fine, coarse, fine


def finite_policy_gap_resolution(
        *, n: int, g_alphabet_size: int, z_alphabet_size: int,
        action_count: int, loss_range: float, alpha: float = .05) -> dict:
    """Outcome-blind preflight for the finite-policy gap route."""
    sizes = (n, g_alphabet_size, z_alphabet_size, action_count)
    if (any(not isinstance(value, (int, np.integer)) or value < 1 for value in sizes)
            or not 0 < alpha < 1):
        raise ValueError("positive integer sizes and valid alpha required")
    amplitude = float(loss_range)
    if not math.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("positive finite loss range required")
    log_actions = math.log(action_count)
    log_coarse = g_alphabet_size * log_actions
    log_fine = g_alphabet_size * z_alphabet_size * log_actions
    allocation = _alpha_allocation(alpha, log_coarse, log_fine, amplitude, n)
    alpha_coarse, alpha_fine, coarse, fine = allocation
    return {
        "n": int(n), "alpha": float(alpha),
        "g_alphabet_size": int(g_alphabet_size),
        "z_alphabet_size": int(z_alphabet_size),
        "action_count": int(action_count), "loss_range": amplitude,
        "log_coarse_policy_count": float(log_coarse),
        "log_fine_policy_count": float(log_fine),
        "alpha_coarse": alpha_coarse, "alpha_fine": alpha_fine,
        "coarse_radius": float(coarse), "fine_radius": float(fine),
        "gap_uncertainty": float(coarse + fine),
        "formula": "gap_cap=plugin_gap+coarse_radius+fine_radius",
        "outcome_alphabet_required": False, "outcome_blind": True,
    }


def _encode(values, alphabet, label):
    declared = tuple(alphabet)
    if not declared or len(set(declared)) != len(declared):
        raise ValueError(f"{label} alphabet must be nonempty and unique")
    index = {value: j for j, value in enumerate(declared)}
    try:
        return np.asarray([index[value] for value in values], dtype=int), declared
    except (KeyError, TypeError) as exc:
        raise ValueError(f"observed {label} value is outside its declared alphabet") from exc


def finite_policy_gap_upper_bound(
        losses: Sequence[Sequence[float]] | np.ndarray,
        z: Sequence[object], g: Sequence[object], *, loss: str,
        action_labels: Sequence[object], z_alphabet: Sequence[object],
        g_alphabet: Sequence[object], loss_bounds: tuple[float, float],
        alpha: float = .05, registry_frozen_before_evaluation: bool,
        action_class_complete_for_target: bool,
        provenance: Mapping[str, object] | None = None,
        maximum_bayes_risk: float | None = None) -> FinitePolicyGapUpperBound:
    """Upper-bound loss-relative information value over a finite action class."""
    matrix = np.asarray(losses, dtype=float)
    n = int(matrix.shape[0]) if matrix.ndim == 2 else 0
    actions = tuple(action_labels)
    g_declared, z_declared = tuple(g_alphabet), tuple(z_alphabet)
    base = dict(loss=loss, alpha=float(alpha), n=n, action_count=len(actions),
                coarse_cell_count=len(g_declared),
                fine_cell_count=len(g_declared) * len(z_declared))
    if not 0 < alpha < 1 or not loss or not provenance:
        raise ValueError("loss, provenance, and valid alpha are required")
    if matrix.ndim != 2 or n < 1 or len(actions) != matrix.shape[1] or not actions:
        raise ValueError("losses must have one finite action column per action label")
    if len(set(actions)) != len(actions) or not np.isfinite(matrix).all():
        raise ValueError("action labels must be unique and losses finite")
    lo, hi = map(float, loss_bounds)
    if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
        raise ValueError("finite increasing loss bounds required")
    if np.any(matrix < lo - 1e-12) or np.any(matrix > hi + 1e-12):
        raise ValueError("observed losses leave the declared bounds")
    if len(tuple(z)) != n or len(tuple(g)) != n:
        raise ValueError("group/refinement labels must align with loss rows")
    refusal = None
    if not registry_frozen_before_evaluation:
        refusal = {
            "reason": "policy registry was not frozen before evaluation",
            "referral": "freeze actions, cells, loss bounds, and provenance before outcomes",
        }
    elif not action_class_complete_for_target:
        refusal = {
            "reason": "finite action class is not complete for the target Bayes-risk problem",
            "referral": "prove action completeness or label the result as a restricted-class diagnostic",
        }
    if refusal:
        return FinitePolicyGapUpperBound(
            status="refused", claim_tier="refused", gap_cap=None,
            plugin_gap=None, plugin_coarse_risk=None, plugin_fine_risk=None,
            loss_range=hi - lo, assumptions=(),
            diagnostics={"provenance": dict(provenance)}, refusal=refusal, **base)
    try:
        gi, ga = _encode(tuple(g), g_declared, "G")
        zi, za = _encode(tuple(z), z_declared, "Z")
    except ValueError as exc:
        return FinitePolicyGapUpperBound(
            status="refused", claim_tier="refused", gap_cap=None,
            plugin_gap=None, plugin_coarse_risk=None, plugin_fine_risk=None,
            loss_range=hi - lo, assumptions=(),
            diagnostics={"provenance": dict(provenance)},
            refusal={"reason": str(exc),
                     "referral": "correct the registry; never add observed states post hoc"}, **base)

    coarse = 0.
    fine = 0.
    for g_index in range(len(ga)):
        in_g = gi == g_index
        if np.any(in_g):
            coarse += float(np.min(np.sum(matrix[in_g], axis=0)) / n)
        for z_index in range(len(za)):
            in_cell = in_g & (zi == z_index)
            if np.any(in_cell):
                fine += float(np.min(np.sum(matrix[in_cell], axis=0)) / n)
    plugin = float(max(0., coarse - fine))
    plan = finite_policy_gap_resolution(
        n=n, g_alphabet_size=len(ga), z_alphabet_size=len(za),
        action_count=len(actions), loss_range=hi - lo, alpha=alpha)
    cap = plugin + plan["gap_uncertainty"]
    if maximum_bayes_risk is not None:
        maximum = float(maximum_bayes_risk)
        if not math.isfinite(maximum) or maximum < 0:
            raise ValueError("maximum Bayes risk must be finite and nonnegative")
        cap = min(cap, maximum)
    return FinitePolicyGapUpperBound(
        status="conditional", claim_tier="conditional", gap_cap=float(cap),
        plugin_gap=plugin, plugin_coarse_risk=coarse, plugin_fine_risk=fine,
        loss_range=hi - lo,
        assumptions=(
            "rows are iid from one registered population",
            "G and Z have complete finite alphabets frozen before evaluation",
            "the finite action class contains a Bayes action for every supported conditional law",
            "the complete pointwise loss range is bounded by the declared interval",
        ),
        diagnostics={
            **plan, "concentration": "one-sided Hoeffding union over finite policies",
            "alpha_allocation": "outcome-blind numerical optimum",
            "target": "loss-relative value of Z beyond G",
            "provenance": dict(provenance),
            "maximum_bayes_risk_cap": maximum_bayes_risk,
        }, refusal=None, **base)


def finite_coordinate_bernstein_gap_upper_bound(
        losses: Sequence[Sequence[float]] | np.ndarray,
        z: Sequence[object], g: Sequence[object], *, loss: str,
        action_labels: Sequence[object], z_alphabet: Sequence[object],
        g_alphabet: Sequence[object], loss_bounds: tuple[float, float],
        alpha: float = .05, registry_frozen_before_evaluation: bool,
        action_class_complete_for_target: bool,
        provenance: Mapping[str, object] | None = None,
        maximum_bayes_risk: float | None = None) -> FinitePolicyGapUpperBound:
    """Variance-adaptive gap cap from simultaneous cell/action rectangles.

    For every coarse cell/action and refined cell/action, this route treats
    ``1(cell) * shifted_loss(action)`` as one bounded iid variable.  Simultaneous
    Maurer--Pontil empirical-Bernstein intervals imply

    ``R_G <= sum_g min_a U[g,a]`` and
    ``R_GZ >= sum_gz min_a L[g,z,a]``.

    The subtraction therefore upper-bounds the information gap.  Loss is
    shifted by its declared lower endpoint before building coordinates; this
    leaves the gap invariant and gives coordinate range equal to loss range.
    """
    base_result = finite_policy_gap_upper_bound(
        losses, z, g, loss=loss, action_labels=action_labels,
        z_alphabet=z_alphabet, g_alphabet=g_alphabet,
        loss_bounds=loss_bounds, alpha=alpha,
        registry_frozen_before_evaluation=registry_frozen_before_evaluation,
        action_class_complete_for_target=action_class_complete_for_target,
        provenance=provenance, maximum_bayes_risk=maximum_bayes_risk)
    if base_result.status != "conditional":
        return base_result
    matrix = np.asarray(losses, dtype=float)
    n, action_count = matrix.shape
    lo, hi = map(float, loss_bounds)
    amplitude = hi - lo
    shifted = matrix - lo
    gi, ga = _encode(tuple(g), g_alphabet, "G")
    zi, za = _encode(tuple(z), z_alphabet, "Z")
    n_events = len(ga) * action_count + len(ga) * len(za) * action_count
    log_term = math.log(2. * n_events / alpha)

    def interval(values):
        estimate = float(np.mean(values))
        variance = float(np.var(values, ddof=1))
        radius = (math.sqrt(2. * variance * log_term / n)
                  + 7. * amplitude * log_term / (3. * (n - 1.)))
        return max(0., estimate - radius), min(amplitude, estimate + radius), radius

    coarse_upper = 0.
    fine_lower = 0.
    radii = []
    for g_index in range(len(ga)):
        coarse_coordinates = []
        for action in range(action_count):
            values = (gi == g_index) * shifted[:, action]
            lower, upper, radius = interval(values)
            coarse_coordinates.append((lower, upper))
            radii.append(radius)
        coarse_upper += min(upper for _, upper in coarse_coordinates)
        for z_index in range(len(za)):
            fine_coordinates = []
            mask = (gi == g_index) & (zi == z_index)
            for action in range(action_count):
                values = mask * shifted[:, action]
                lower, upper, radius = interval(values)
                fine_coordinates.append((lower, upper))
                radii.append(radius)
            fine_lower += min(lower for lower, _ in fine_coordinates)
    cap = float(max(0., coarse_upper - fine_lower))
    if maximum_bayes_risk is not None:
        cap = min(cap, float(maximum_bayes_risk))
    diagnostics = dict(base_result.diagnostics)
    diagnostics.update({
        "concentration": "simultaneous Maurer-Pontil empirical-Bernstein cell/action rectangles",
        "coordinate_event_count": int(n_events),
        "coordinate_log_term": float(log_term),
        "coarse_risk_upper": float(coarse_upper),
        "fine_risk_lower": float(fine_lower),
        "median_coordinate_radius": float(np.median(radii)),
        "maximum_coordinate_radius": float(np.max(radii)),
        "loss_shift": lo,
        "selection": "no posthoc route selection; this endpoint uses its full declared alpha",
    })
    return FinitePolicyGapUpperBound(
        status="conditional", claim_tier="conditional", loss=loss,
        gap_cap=cap, plugin_gap=base_result.plugin_gap,
        plugin_coarse_risk=base_result.plugin_coarse_risk,
        plugin_fine_risk=base_result.plugin_fine_risk,
        alpha=float(alpha), n=n, loss_range=amplitude,
        action_count=action_count, coarse_cell_count=len(ga),
        fine_cell_count=len(ga) * len(za),
        assumptions=base_result.assumptions + (
            "all cell/action empirical-Bernstein intervals share the recorded union-bound event",
        ),
        diagnostics=diagnostics, refusal=None)


def finite_loss_routed_gap_upper_bound(
        losses: Sequence[Sequence[float]] | np.ndarray,
        z: Sequence[object], g: Sequence[object], *, loss: str,
        action_labels: Sequence[object], z_alphabet: Sequence[object],
        g_alphabet: Sequence[object], loss_bounds: tuple[float, float],
        alpha: float = .05, registry_frozen_before_evaluation: bool,
        action_class_complete_for_target: bool,
        provenance: Mapping[str, object] | None = None,
        maximum_bayes_risk: float | None = None,
        router_version: str = "finite-gap-v1") -> FinitePolicyGapUpperBound:
    """Apply the frozen, confirmation-audited loss router.

    V1 supports only the exact registered loss names in
    ``FINITE_GAP_LOSS_ROUTER_V1``.  It refuses to extrapolate the empirical
    Q10/Q90 finding to arbitrary quantile levels.
    """
    if router_version != "finite-gap-v1":
        raise ValueError("unsupported finite-gap router version")
    key = loss.strip().upper()
    if key not in FINITE_GAP_LOSS_ROUTER_V1:
        raise ValueError("loss is not registered in finite-gap-v1; choose and audit a route explicitly")
    route = FINITE_GAP_LOSS_ROUTER_V1[key]
    kwargs = dict(
        losses=losses, z=z, g=g, loss=loss, action_labels=action_labels,
        z_alphabet=z_alphabet, g_alphabet=g_alphabet,
        loss_bounds=loss_bounds, alpha=alpha,
        registry_frozen_before_evaluation=registry_frozen_before_evaluation,
        action_class_complete_for_target=action_class_complete_for_target,
        provenance=provenance, maximum_bayes_risk=maximum_bayes_risk)
    if route == "coordinate_empirical_bernstein":
        result = finite_coordinate_bernstein_gap_upper_bound(**kwargs)
    else:
        result = finite_policy_gap_upper_bound(**kwargs)
    diagnostics = dict(result.diagnostics)
    diagnostics.update({
        "router_version": router_version,
        "selected_route": route,
        "route_selected_from_current_outcomes": False,
        "router_scope": tuple(FINITE_GAP_LOSS_ROUTER_V1),
    })
    return replace(result, diagnostics=diagnostics)
