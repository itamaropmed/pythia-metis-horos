"""Fail-closed transfer of a group-reference bound to richer information.

This implements classical decision-risk/stability inequalities. It does not
estimate or validate the supplied information-gap or conditional-law cap.
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict


def transfer_bound(group_result: dict, *, bound_key: str = "tuple_bound",
                   risk_gap_cap: float | None = None,
                   mean_w1_cap: float | None = None,
                   cap_provenance: str | None = None) -> dict:
    """Lower-bound full-information risk only under a declared valid gap cap.

    risk_gap_cap upper-bounds group-reference original Bayes risk minus
    full audited original Bayes risk (loss units). Alternatively mean_w1_cap
    bounds the audited mean distance from each clipped full conditional law
    to its clipped group-reference law. All caps need external justification.
    """
    if risk_gap_cap is None and mean_w1_cap is None:
        return {"status": "refused", "bound": None,
                "reason": "no independently justified raw-information gap cap supplied",
                "target": "full audited information Bayes risk"}
    if risk_gap_cap is not None and mean_w1_cap is not None:
        raise ValueError("supply a risk-gap cap OR a mean-W1 cap, not both")
    if cap_provenance is None or not cap_provenance.strip():
        return {"status": "refused", "bound": None,
                "reason": "gap-cap provenance required; provenance does not itself prove validity"}
    base = float(group_result[bound_key])
    cap = float(risk_gap_cap if risk_gap_cap is not None else mean_w1_cap)
    if not np.isfinite([base, cap]).all() or base < 0 or cap < 0:
        raise ValueError("finite nonnegative bound and cap required")
    multiplier = 1.0
    if mean_w1_cap is not None:
        loss = group_result["loss"]
        if loss == "MAE":
            multiplier = 1.0
        elif loss == "Q90":
            multiplier = 0.9
        elif loss == "MSE":
            lo, hi = map(float, group_result["outcome_clip"])
            if not np.isfinite([lo, hi]).all() or lo >= hi:
                raise ValueError("finite ordered MSE outcome clip required")
            multiplier = 2 * (hi - lo)
        else:
            return {"status": "refused", "bound": None,
                    "reason": "this loss has no implemented W1 transfer"}
    return {"status": "conditional", "claim_tier": "conditional",
            "bound": float(max(0., base - multiplier * cap)),
            "group_reference_bound": base, "cap": cap,
            "cap_kind": "risk_gap" if risk_gap_cap is not None else "clipped_mean_w1",
            "loss_multiplier": multiplier, "cap_breakpoint": base / multiplier,
            "cap_provenance": cap_provenance,
            "target": "full audited original-outcome Bayes risk",
            "cap_validated_by_this_function": False,
            "assumptions": group_result.get("assumptions", []) + [
                "supplied cap truly upper-bounds the declared full-information gap",
                "any statistical cap uncertainty is separately accounted for",
            ]}


def sensitivity_profile(group_result: dict, grid: np.ndarray, *,
                        cap_kind: str, bound_key: str = "tuple_bound") -> dict:
    caps = np.asarray(grid, float).ravel()
    if (not len(caps) or not np.isfinite(caps).all() or np.any(caps < 0)
            or np.any(np.diff(caps) < 0) or cap_kind not in ("risk_gap", "mean_w1")):
        raise ValueError("ordered finite nonnegative grid and supported cap kind required")
    rows = [transfer_bound(group_result, bound_key=bound_key,
                           cap_provenance="sensitivity only: cap not validated",
                           **{f"{cap_kind}_cap": float(cap)}) for cap in caps]
    if any(row["status"] == "refused" for row in rows):
        return {"status": "refused", "reason": "unsupported transfer loss",
                "bounds": None}
    return {"status": "sensitivity_only", "cap_kind": cap_kind,
            "grid": caps.tolist(), "bounds": [row["bound"] for row in rows],
            "breakpoint": rows[0]["cap_breakpoint"],
            "warning": "the curve does not identify or validate a cap"}


def maximum_clipped_bayes_risk(loss: str, clip: tuple[float, float]) -> float:
    """Distribution-free maximum Bayes risk on a bounded interval."""
    lo, hi = map(float, clip)
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError("finite ordered clip required")
    width = hi - lo
    if loss == "MAE":
        return width / 2  # midpoint action
    if loss == "MSE":
        return width ** 2 / 4  # Popoviciu / midpoint action
    if loss.startswith("Q"):
        q = float(loss[1:]) / 100
        if not 0 < q < 1:
            raise ValueError("valid quantile required")
        return q * (1 - q) * width  # minimax fixed action q*hi+(1-q)*lo
    if loss.startswith("W"):
        h = float(loss[1:])
        if h <= 0:
            raise ValueError("positive window required")
        return 0. if 2 * h >= width else 1.
    raise ValueError("unsupported loss")


def clipped_loss_range(loss: str, clip: tuple[float, float]) -> float:
    """Range of a location loss when outcomes/actions lie in the clip."""
    lo, hi = map(float, clip)
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError("finite ordered clip required")
    width = hi - lo
    if loss == "MAE":
        return width
    if loss == "MSE":
        return width ** 2
    if loss.startswith("Q"):
        q = float(loss[1:]) / 100
        if not 0 < q < 1:
            raise ValueError("valid quantile required")
        return max(q, 1 - q) * width
    if loss.startswith("W"):
        h = float(loss[1:])
        if h <= 0:
            raise ValueError("positive window required")
        return 1.
    raise ValueError("unsupported loss")


def clipped_loss_lipschitz(loss: str, clip: tuple[float, float]) -> float | None:
    """Outcome-Lipschitz constant; window loss deliberately returns None."""
    lo, hi = map(float, clip)
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError("finite ordered clip required")
    width = hi - lo
    if loss == "MAE":
        return 1.
    if loss == "MSE":
        return 2 * width
    if loss.startswith("Q"):
        q = float(loss[1:]) / 100
        if not 0 < q < 1:
            raise ValueError("valid quantile required")
        return max(q, 1 - q)
    if loss.startswith("W"):
        h = float(loss[1:])
        if h <= 0:
            raise ValueError("positive window required")
        return None
    raise ValueError("unsupported loss")


def registered_conditional_entropy(coarse_labels, refined_labels,
                                   weights=None) -> float:
    """Exact H(refined|coarse) for a registered finite target design.

    Labels and optional weights describe the target design itself, not a sample
    used to infer a future covariate distribution. Natural logarithms are used.
    """
    coarse = list(coarse_labels)
    refined = list(refined_labels)
    if len(coarse) != len(refined) or not coarse:
        raise ValueError("nonempty aligned labels required")
    if weights is None:
        w = np.full(len(coarse), 1 / len(coarse))
    else:
        w = np.asarray(weights, float).ravel()
        if len(w) != len(coarse) or not np.isfinite(w).all() or np.any(w < 0):
            raise ValueError("finite nonnegative aligned weights required")
        if w.sum() <= 0:
            raise ValueError("positive total weight required")
        w = w / w.sum()
    joint = defaultdict(float)
    marginal = defaultdict(float)
    for g, z, wi in zip(coarse, refined, w):
        try:
            joint[(g, z)] += float(wi)
            marginal[g] += float(wi)
        except TypeError as exc:
            raise ValueError("labels must be hashable registered values") from exc
    entropy = 0.
    for (g, _), mass in joint.items():
        if mass > 0:
            entropy -= mass * np.log(mass / marginal[g])
    return float(max(0., entropy))


def information_refinement_profile(
        group_result: dict, information_grid_nats: np.ndarray, *,
        bound_key: str = "lower", representation_provenance: str | None = None,
        pre_outcome_registered: bool = False,
        t1_constant: float | None = None) -> dict:
    """Transfer a coarse lower bound through a conditional-information cap.

    If I(Y;Z|G) <= J, bounded-loss Pinsker gives a value-of-information cap
    range(loss)*sqrt(J/2). A supplied conditional T1 constant c optionally
    adds the sharper Lipschitz cap L*sqrt(2*c*J). H(Z|G) is always a valid J
    for a finite pre-outcome refinement Z, but this routine does not infer J.
    """
    if not pre_outcome_registered:
        return {"status": "refused", "bound": None,
                "reason": "refinement is not registered as pre-outcome",
                "target": "clipped richer-information Bayes risk"}
    if representation_provenance is None or not representation_provenance.strip():
        return {"status": "refused", "bound": None,
                "reason": "representation/information-cap provenance required",
                "target": "clipped richer-information Bayes risk"}
    grid = np.asarray(information_grid_nats, float).ravel()
    if (not len(grid) or not np.isfinite(grid).all() or np.any(grid < 0)
            or np.any(np.diff(grid) < 0)):
        raise ValueError("ordered finite nonnegative information grid required")
    if "outcome_clip" not in group_result or "loss" not in group_result:
        raise ValueError("group result needs loss and outcome clip")
    base = float(group_result[bound_key])
    if not np.isfinite(base) or base < 0:
        raise ValueError("finite nonnegative group lower bound required")
    loss = group_result["loss"]
    clip = tuple(group_result["outcome_clip"])
    risk_max = maximum_clipped_bayes_risk(loss, clip)
    loss_range = clipped_loss_range(loss, clip)
    pinsker = loss_range * np.minimum(1., np.sqrt(grid / 2))
    caps = np.minimum(risk_max, pinsker)
    routes = np.full(len(grid), "bounded_loss_pinsker", dtype=object)
    if t1_constant is not None:
        c = float(t1_constant)
        if not np.isfinite(c) or c < 0:
            raise ValueError("finite nonnegative T1 constant required")
        lipschitz = clipped_loss_lipschitz(loss, clip)
        if lipschitz is None:
            return {"status": "refused", "bound": None,
                    "reason": "T1 refinement is unsupported for discontinuous window loss",
                    "target": "clipped richer-information Bayes risk"}
        t1_caps = lipschitz * np.sqrt(2 * c * grid)
        take_t1 = t1_caps < caps
        caps = np.minimum(caps, t1_caps)
        routes[take_t1] = "conditional_T1"
    bounds = np.maximum(0., base - caps)
    return {
        "status": "sensitivity_only", "claim_tier": "conditional",
        "loss": loss, "information_nats": grid.tolist(),
        "bounds": bounds.tolist(), "gap_caps": caps.tolist(),
        "routes": routes.tolist(), "group_reference_bound": base,
        "maximum_clipped_bayes_risk": float(risk_max),
        "loss_range": float(loss_range),
        "t1_constant": None if t1_constant is None else float(t1_constant),
        "representation_provenance": representation_provenance,
        "target": "clipped richer-information Bayes risk",
        "assumptions": [
            "the supplied value upper-bounds I(Y; refinement | coarse) in nats",
            "the refinement was fixed and available before the outcome",
            "the coarse lower confidence event and its original assumptions hold",
        ] + ([] if t1_constant is None else [
            "every relevant coarse conditional outcome law satisfies T1(c) for the supplied c",
        ]),
        "warning": "the profile does not estimate mutual information or validate a T1 constant",
    }


def localized_refinement_profile(group_result: dict, epsilon_grid: np.ndarray,
                                 *, bound_key: str = "lower") -> dict:
    """Sensitivity when hidden refinement can help only on epsilon target mass.

    Assumption: outside a predeclared exceptional set of target mass <=epsilon,
    the group representation is Bayes-sufficient for this loss. Inside it,
    richer information may reduce risk arbitrarily. This function does not
    identify the exceptional set or certify the supplied mass bound.
    """
    eps = np.asarray(epsilon_grid, float).ravel()
    if (not len(eps) or not np.isfinite(eps).all() or np.any(eps < 0)
            or np.any(eps > 1) or np.any(np.diff(eps) < 0)):
        raise ValueError("ordered epsilon grid in [0,1] required")
    if "outcome_clip" not in group_result or "loss" not in group_result:
        raise ValueError("group result needs loss and outcome clip")
    base = float(group_result[bound_key])
    maximum = maximum_clipped_bayes_risk(
        group_result["loss"], tuple(group_result["outcome_clip"]))
    bounds = np.maximum(0., base - eps * maximum)
    breakpoint = 0. if maximum == 0 else min(1., base / maximum)
    return {"status": "sensitivity_only", "claim_tier": "conditional",
            "loss": group_result["loss"], "epsilon": eps.tolist(),
            "bounds": bounds.tolist(), "group_reference_bound": base,
            "maximum_clipped_bayes_risk": float(maximum),
            "epsilon_breakpoint": float(breakpoint),
            "target": "clipped full-information Bayes risk",
            "assumptions": [
                "outside an exceptional target set of mass <=epsilon, the group representation is Bayes-sufficient for this loss",
                "exceptional-set membership/mass is fixed or justified independently of outcomes",
            ],
            "warning": "the curve neither discovers nor validates the exceptional set"}
