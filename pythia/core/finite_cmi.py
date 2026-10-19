"""Finite-alphabet upper confidence bounds on conditional information.

Pythia needs an *upper* bound on the value of a representation refinement.
Conditional-independence tests and variational mutual-information lower bounds
cannot provide it.  This module supplies a conservative finite-alphabet route:

1. Weissman's multinomial L1 inequality gives one confidence ball for the
   complete joint law of ``(Y,Z,G)``;
2. total variation contracts under marginalisation;
3. the sharp Fannes--Audenaert entropy modulus controls the four entropy terms
   in ``I(Y;Z|G)``;
4. Pinsker converts the CMI upper endpoint into a bounded-loss Bayes-risk gap.

The route is intentionally refused for undeclared/continuous alphabets.  A
coarsened outcome cannot upper-bound information present in the original
outcome without an additional approximation theorem.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import math
import numpy as np


@dataclass(frozen=True)
class FiniteCMIUpperBound:
    status: str
    claim_tier: str
    upper_cmi: float | None
    plugin_cmi: float | None
    alpha: float
    n: int
    alphabet_sizes: Mapping[str, int]
    l1_radius: float | None
    tv_radius: float | None
    assumptions: tuple[str, ...]
    diagnostics: Mapping[str, object]
    refusal: Mapping[str, str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FiniteDecisionGapUpperBound:
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
    tv_radius: float | None
    assumptions: tuple[str, ...]
    diagnostics: Mapping[str, object]
    refusal: Mapping[str, str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0]
    return float(-np.sum(positive * np.log(positive)))


def _entropy_modulus(tv: float, alphabet_size: int) -> float:
    """Sharp classical Fannes--Audenaert modulus in natural units."""
    if alphabet_size <= 1 or tv <= 0:
        return 0.
    maximum = math.log(alphabet_size)
    threshold = 1. - 1. / alphabet_size
    if tv >= threshold:
        return maximum
    binary_entropy = (-tv * math.log(tv)
                      - (1. - tv) * math.log1p(-tv))
    return float(min(maximum, tv * math.log(alphabet_size - 1.) + binary_entropy))


def finite_decision_gap_resolution(
        *, n: int, y_alphabet_size: int, z_alphabet_size: int,
        g_alphabet_size: int, loss_range: float, alpha: float = .05) -> dict:
    """Outcome-blind preflight resolution for the direct finite-gap route.

    The returned ``gap_uncertainty`` is the additive term ``2*A*delta``, equal
    to ``A*epsilon_l1``.  Metis can use it to reject a route before inspecting
    any outcome counts or empirical gap.
    """
    sizes = (n, y_alphabet_size, z_alphabet_size, g_alphabet_size)
    if (any(not isinstance(value, (int, np.integer)) or value < 1 for value in sizes)
            or not 0 < alpha < 1):
        raise ValueError("positive integer sample/alphabet sizes and valid alpha required")
    amplitude = float(loss_range)
    if not math.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("positive finite loss range required")
    joint_size = int(y_alphabet_size * z_alphabet_size * g_alphabet_size)
    l1 = min(2., math.sqrt(2. * (joint_size * math.log(2.)
                                + math.log(1. / alpha)) / n))
    return {
        "n": int(n), "alpha": float(alpha), "joint_alphabet_size": joint_size,
        "l1_radius": float(l1), "tv_radius": float(l1 / 2.),
        "loss_range": amplitude, "gap_uncertainty": float(amplitude * l1),
        "saturated": bool(l1 >= 2. - 1e-15),
        "formula": "gap_cap=plugin_gap+loss_range*l1_radius",
        "outcome_blind": True,
    }


def _encode(values, alphabet, label):
    declared = tuple(alphabet)
    if not declared or len(set(declared)) != len(declared):
        raise ValueError(f"{label} alphabet must be nonempty and unique")
    index = {value: position for position, value in enumerate(declared)}
    try:
        encoded = np.asarray([index[value] for value in values], dtype=int)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"observed {label} value is outside its declared alphabet") from exc
    return encoded, declared


def finite_cmi_upper_bound(
        y: Sequence[object], z: Sequence[object], g: Sequence[object], *,
        y_alphabet: Sequence[object], z_alphabet: Sequence[object],
        g_alphabet: Sequence[object], alpha: float = .05,
        alphabet_frozen_before_evaluation: bool,
        provenance: Mapping[str, object] | None = None) -> FiniteCMIUpperBound:
    """Upper-bound ``I(Y;Z|G)`` in nats from iid finite-alphabet samples."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if not provenance:
        raise ValueError("alphabet and sample provenance are required")
    y_values, z_values, g_values = tuple(y), tuple(z), tuple(g)
    n = len(y_values)
    sizes = {
        "Y": len(tuple(y_alphabet)), "Z": len(tuple(z_alphabet)),
        "G": len(tuple(g_alphabet)),
    }
    if n == 0 or len(z_values) != n or len(g_values) != n:
        raise ValueError("aligned nonempty Y, Z, and G samples are required")
    if not alphabet_frozen_before_evaluation:
        return FiniteCMIUpperBound(
            status="refused", claim_tier="refused", upper_cmi=None,
            plugin_cmi=None, alpha=float(alpha), n=n, alphabet_sizes=sizes,
            l1_radius=None, tv_radius=None, assumptions=(),
            diagnostics={"provenance": dict(provenance)},
            refusal={
                "reason": "finite alphabets were not frozen before evaluation",
                "referral": "declare complete alphabets, including unseen values, before inspecting outcomes",
            })
    try:
        yi, ya = _encode(y_values, y_alphabet, "Y")
        zi, za = _encode(z_values, z_alphabet, "Z")
        gi, ga = _encode(g_values, g_alphabet, "G")
    except ValueError as exc:
        return FiniteCMIUpperBound(
            status="refused", claim_tier="refused", upper_cmi=None,
            plugin_cmi=None, alpha=float(alpha), n=n, alphabet_sizes=sizes,
            l1_radius=None, tv_radius=None, assumptions=(),
            diagnostics={"provenance": dict(provenance)},
            refusal={"reason": str(exc),
                     "referral": "correct the registry; never silently add observed categories"})

    dy, dz, dg = len(ya), len(za), len(ga)
    joint_size = dy * dz * dg
    flat = (yi * dz + zi) * dg + gi
    joint = np.bincount(flat, minlength=joint_size).reshape(dy, dz, dg) / n
    p_yg = np.sum(joint, axis=1)
    p_zg = np.sum(joint, axis=0)
    p_g = np.sum(joint, axis=(0, 1))
    plugin = max(0., _entropy(p_yg) + _entropy(p_zg)
                 - _entropy(p_g) - _entropy(joint.ravel()))

    # Weissman et al.: P(||P_hat-P||_1 >= eps) <= 2^K exp(-n eps^2/2).
    # The slightly looser 2^K prefactor avoids numerical special cases and is
    # still a valid one-event confidence ball.
    l1 = min(2., math.sqrt(2. * (joint_size * math.log(2.)
                                + math.log(1. / alpha)) / n))
    tv = l1 / 2.
    moduli = {
        "H(Y,G)": _entropy_modulus(tv, dy * dg),
        "H(Z,G)": _entropy_modulus(tv, dz * dg),
        "H(G)": _entropy_modulus(tv, dg),
        "H(Y,Z,G)": _entropy_modulus(tv, joint_size),
    }
    deterministic_cap = math.log(min(dy, dz))
    upper = float(min(deterministic_cap, plugin + sum(moduli.values())))
    return FiniteCMIUpperBound(
        status="conditional", claim_tier="conditional", upper_cmi=upper,
        plugin_cmi=float(plugin), alpha=float(alpha), n=n,
        alphabet_sizes={"Y": dy, "Z": dz, "G": dg, "joint": joint_size},
        l1_radius=float(l1), tv_radius=float(tv),
        assumptions=(
            "samples are iid from one registered joint law of (Y,Z,G)",
            "the listed finite alphabets are complete and were frozen before evaluation",
            "Z is available at the registered decision time and G is a deterministic coarsening of (G,Z)",
            "no outcome discretisation or category discovery was performed on evaluation data",
        ),
        diagnostics={
            "concentration": "Weissman multinomial L1 deviation bound",
            "continuity": "Fannes-Audenaert entropy modulus",
            "entropy_units": "nats",
            "entropy_moduli": moduli,
            "deterministic_cmi_cap": deterministic_cap,
            "provenance": dict(provenance),
            "common_failure_event": True,
        })


def bounded_loss_gap_from_cmi(
        result: FiniteCMIUpperBound, *, loss: str, loss_range: float,
        maximum_bayes_risk: float | None = None) -> dict:
    """Convert a CMI upper endpoint into a refinement-risk upper endpoint."""
    if result.status != "conditional" or result.upper_cmi is None:
        return {
            "status": "refused", "claim_tier": "refused", "loss": loss,
            "gap_cap": None, "alpha": result.alpha,
            "reason": "a completed finite-alphabet CMI upper bound is required",
        }
    amplitude = float(loss_range)
    if not loss or not math.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("loss and a positive finite loss range are required")
    cap = amplitude * math.sqrt(result.upper_cmi / 2.)
    if maximum_bayes_risk is not None:
        maximum = float(maximum_bayes_risk)
        if not math.isfinite(maximum) or maximum < 0:
            raise ValueError("maximum Bayes risk must be finite and nonnegative")
        cap = min(cap, maximum)
    return {
        "status": "conditional", "claim_tier": result.claim_tier,
        "loss": loss, "gap_cap": float(cap), "upper_cmi": result.upper_cmi,
        "alpha": result.alpha,
        "assumptions": result.assumptions + (
            "the pointwise loss range is bounded by the declared amplitude",
            "Pinsker/Jensen decision-risk transfer applies to the registered refinement",
        ),
        "diagnostics": {
            "formula": "gap <= loss_range*sqrt(I(Y;Z|G)/2)",
            "loss_range": amplitude,
            "maximum_bayes_risk_cap": maximum_bayes_risk,
            "cmi": result.to_dict(),
        },
    }


def finite_decision_gap_upper_bound(
        y: Sequence[object], z: Sequence[object], g: Sequence[object], *,
        y_alphabet: Sequence[object], z_alphabet: Sequence[object],
        g_alphabet: Sequence[object], loss: str,
        loss_matrix: Sequence[Sequence[float]] | np.ndarray,
        alpha: float = .05, alphabet_frozen_before_evaluation: bool,
        provenance: Mapping[str, object] | None = None,
        maximum_bayes_risk: float | None = None) -> FiniteDecisionGapUpperBound:
    """Directly upper-bound the loss-relative value of finite refinement.

    On the same Weissman event as :func:`finite_cmi_upper_bound`, total
    variation between the empirical and population joint laws is at most
    ``delta``.  A Bayes-risk functional for any fixed representation and a
    loss of range ``A`` is ``A``-Lipschitz in TV.  Hence

        gap(P) <= gap(P_hat) + 2 A delta.

    This avoids the generally loose CMI-to-Pinsker conversion and makes the
    representation gap explicitly loss-relative.
    """
    cmi = finite_cmi_upper_bound(
        y, z, g, y_alphabet=y_alphabet, z_alphabet=z_alphabet,
        g_alphabet=g_alphabet, alpha=alpha,
        alphabet_frozen_before_evaluation=alphabet_frozen_before_evaluation,
        provenance=provenance)
    if cmi.status != "conditional":
        return FiniteDecisionGapUpperBound(
            status="refused", claim_tier="refused", loss=loss, gap_cap=None,
            plugin_gap=None, plugin_coarse_risk=None, plugin_fine_risk=None,
            alpha=float(alpha), n=cmi.n, loss_range=None, tv_radius=None,
            assumptions=(), diagnostics={"cmi_validation": cmi.to_dict()},
            refusal=cmi.refusal)
    if not loss:
        raise ValueError("loss name is required")
    matrix = np.asarray(loss_matrix, dtype=float)
    dy = cmi.alphabet_sizes["Y"]
    if (matrix.ndim != 2 or matrix.shape[0] != dy or matrix.shape[1] < 1
            or not np.isfinite(matrix).all()):
        raise ValueError("loss matrix must be finite with one row per declared outcome")
    amplitude = float(np.max(matrix) - np.min(matrix))
    if amplitude <= 0:
        raise ValueError("loss matrix must have positive range")

    yi, ya = _encode(tuple(y), y_alphabet, "Y")
    zi, za = _encode(tuple(z), z_alphabet, "Z")
    gi, ga = _encode(tuple(g), g_alphabet, "G")
    dz, dg = len(za), len(ga)
    flat = (yi * dz + zi) * dg + gi
    joint = np.bincount(flat, minlength=dy * dz * dg).reshape(dy, dz, dg) / cmi.n
    coarse_cost = np.einsum("yzg,ya->ga", joint, matrix)
    fine_cost = np.einsum("yzg,ya->gza", joint, matrix)
    coarse = float(np.sum(np.min(coarse_cost, axis=1)))
    fine = float(np.sum(np.min(fine_cost, axis=2)))
    plugin_gap = float(max(0., coarse - fine))
    cap = plugin_gap + 2. * amplitude * float(cmi.tv_radius)
    if maximum_bayes_risk is not None:
        maximum = float(maximum_bayes_risk)
        if not math.isfinite(maximum) or maximum < 0:
            raise ValueError("maximum Bayes risk must be finite and nonnegative")
        cap = min(cap, maximum)
    return FiniteDecisionGapUpperBound(
        status="conditional", claim_tier="conditional", loss=loss,
        gap_cap=float(cap), plugin_gap=plugin_gap,
        plugin_coarse_risk=coarse, plugin_fine_risk=fine,
        alpha=float(alpha), n=cmi.n, loss_range=amplitude,
        tv_radius=float(cmi.tv_radius),
        assumptions=cmi.assumptions + (
            "the supplied finite action/loss matrix is the registered decision problem",
            "Bayes risks are computed over every action column in that matrix",
        ),
        diagnostics={
            "formula": "gap(P)<=gap(P_hat)+2*loss_range*TV(P,P_hat)",
            "concentration": "one shared Weissman joint-law event",
            "information_object": "loss-relative decision information",
            "generic_cmi_not_required": True,
            "maximum_bayes_risk_cap": maximum_bayes_risk,
            "joint_l1_radius": cmi.l1_radius,
            "provenance": dict(provenance or {}),
        })
