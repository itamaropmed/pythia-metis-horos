"""Original-scale SAA lower endpoints under declared moment bounds.

This module deliberately solves a narrow problem.  Let ``S_b`` be independent,
nonnegative empirical-optimum (SAA) scores from predeclared, disjoint blocks.
For each block, SAA optimism gives ``E[S_b] <= R_b^*``.  If externally justified
variance caps ``v_b >= Var(S_b)`` are available, one-sided Cantelli yields

    mean_b R_b^* >= mean_b S_b
                        - sqrt((1-alpha)/alpha * sum_b(v_b) / B**2)

with probability at least ``1-alpha``.  Unlike a clipped concentration route,
the endpoint is on the original outcome/loss scale.  The price is an explicit
moment assumption; a cap estimated from the same evaluation scores is refused.

The helpers below derive conservative score-variance caps for location losses
from moments around a fixed anchor.  They never fit a predictor or choose an
anchor from the evaluation outcomes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import math
import numpy as np


@dataclass(frozen=True)
class MomentSAAResult:
    status: str
    claim_tier: str
    loss: str
    lower: float | None
    observed_mean: float | None
    radius: float | None
    alpha: float
    n_scores: int
    total_variance_bound: float | None
    outcome_scale: str
    cap_source: str
    assumptions: tuple[str, ...]
    diagnostics: Mapping[str, object]
    refusal: Mapping[str, str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _refusal(*, loss: str, alpha: float, n_scores: int, outcome_scale: str,
             cap_source: str, reason: str, referral: str,
             diagnostics: Mapping[str, object] | None = None) -> MomentSAAResult:
    return MomentSAAResult(
        status="refused", claim_tier="refused", loss=loss, lower=None,
        observed_mean=None, radius=None, alpha=float(alpha), n_scores=n_scores,
        total_variance_bound=None, outcome_scale=outcome_scale,
        cap_source=cap_source, assumptions=(), diagnostics=diagnostics or {},
        refusal={"reason": reason, "referral": referral})


def cantelli_saa_lower_bound(
        scores: Sequence[float] | np.ndarray, *, loss: str,
        score_variance_bounds: float | Sequence[float] | np.ndarray,
        alpha: float = .05, cap_source: str,
        cap_frozen_before_evaluation: bool,
        outcome_scale: str = "original loss units") -> MomentSAAResult:
    """Lower-bound average Bayes risk from independent SAA block scores.

    A scalar variance bound is broadcast to every score.  A vector supplies a
    separate cap for each independent score.  ``cap_source`` is recorded, not
    verified; callers remain responsible for the scientific truth of the cap.
    The function refuses caps learned from the evaluation outcomes because the
    fixed-cap Cantelli proof does not cover that reuse.
    """
    values = np.asarray(scores, dtype=float)
    n = int(values.size) if values.ndim == 1 else 0
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if not loss or not outcome_scale:
        raise ValueError("loss and outcome scale are required")
    if not cap_source:
        raise ValueError("a nonempty variance-cap source is required")
    if values.ndim != 1 or n == 0 or not np.all(np.isfinite(values)):
        return _refusal(
            loss=loss, alpha=alpha, n_scores=n, outcome_scale=outcome_scale,
            cap_source=cap_source, reason="scores must be a finite nonempty vector",
            referral="supply one nonnegative score per predeclared independent block")
    if np.any(values < 0):
        return _refusal(
            loss=loss, alpha=alpha, n_scores=n, outcome_scale=outcome_scale,
            cap_source=cap_source, reason="SAA loss scores must be nonnegative",
            referral="audit the score construction and loss orientation")
    if not cap_frozen_before_evaluation:
        return _refusal(
            loss=loss, alpha=alpha, n_scores=n, outcome_scale=outcome_scale,
            cap_source=cap_source,
            reason="variance cap was not frozen independently of evaluation outcomes",
            referral="declare an analytic, elicited, prospective, or independent-sample cap")

    caps = np.asarray(score_variance_bounds, dtype=float)
    if caps.ndim == 0:
        caps = np.full(n, float(caps))
    if (caps.ndim != 1 or caps.size != n or not np.all(np.isfinite(caps))
            or np.any(caps < 0)):
        return _refusal(
            loss=loss, alpha=alpha, n_scores=n, outcome_scale=outcome_scale,
            cap_source=cap_source,
            reason="variance caps must be finite, nonnegative, and aligned with scores",
            referral="supply one valid cap per independent score or one uniform scalar cap")

    total_variance = float(np.sum(caps))
    radius = math.sqrt((1. - alpha) / alpha * total_variance / (n * n))
    observed = float(np.mean(values))
    lower = float(max(0., observed - radius))
    return MomentSAAResult(
        status="conditional", claim_tier="conditional", loss=loss,
        lower=lower, observed_mean=observed, radius=radius, alpha=float(alpha),
        n_scores=n, total_variance_bound=total_variance,
        outcome_scale=outcome_scale, cap_source=cap_source,
        assumptions=(
            "score blocks are mutually independent",
            "each score is a nonnegative empirical optimum whose expectation does not exceed its block Bayes risk",
            "the declared cap bounds each score variance and was frozen independently of evaluation outcomes",
            "the reported target is the unweighted average of the block Bayes risks",
        ),
        diagnostics={
            "inequality": "one-sided Cantelli",
            "original_scale": True,
            "clipping_used": False,
            "cap_is_assumed_not_estimated": True,
            "formula": "max(0, mean(scores)-sqrt((1-alpha)/alpha*sum(v_b)/B^2))",
        })


def catoni_second_moment_saa_lower_bound(
        scores: Sequence[float] | np.ndarray, *, loss: str,
        score_second_moment_bounds: float | Sequence[float] | np.ndarray,
        alpha: float = .05, cap_source: str,
        cap_frozen_before_evaluation: bool, eta: float | None = None,
        outcome_scale: str = "original loss units") -> MomentSAAResult:
    """Catoni-style lower endpoint for an average of heterogeneous means.

    This version does **not** assume iid scores or a common mean.  For
    independent nonnegative ``X_i`` and declared ``m2_i >= E[X_i^2]``, set

        psi(x) = log(1 + x + x^2/2).

    Independence and ``1+a <= exp(a)`` imply

        E exp(sum psi(eta X_i) - eta sum E[X_i]
              - eta^2/2 sum m2_i) <= 1.

    Markov's inequality then gives the returned lower endpoint.  The default
    ``eta`` minimises the two deterministic penalty terms and is fixed solely
    by the declared moments and alpha.  It is therefore not selected from the
    evaluation scores.
    """
    values = np.asarray(scores, dtype=float)
    n = int(values.size) if values.ndim == 1 else 0
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if not loss or not outcome_scale:
        raise ValueError("loss and outcome scale are required")
    if not cap_source:
        raise ValueError("a nonempty second-moment-cap source is required")
    if values.ndim != 1 or n == 0 or not np.all(np.isfinite(values)):
        return _refusal(
            loss=loss, alpha=alpha, n_scores=n, outcome_scale=outcome_scale,
            cap_source=cap_source, reason="scores must be a finite nonempty vector",
            referral="supply one nonnegative score per predeclared independent block")
    if np.any(values < 0):
        return _refusal(
            loss=loss, alpha=alpha, n_scores=n, outcome_scale=outcome_scale,
            cap_source=cap_source, reason="SAA loss scores must be nonnegative",
            referral="audit the score construction and loss orientation")
    if not cap_frozen_before_evaluation:
        return _refusal(
            loss=loss, alpha=alpha, n_scores=n, outcome_scale=outcome_scale,
            cap_source=cap_source,
            reason="second-moment cap was not frozen independently of evaluation outcomes",
            referral="declare an analytic, elicited, prospective, or independent-sample cap")

    caps = np.asarray(score_second_moment_bounds, dtype=float)
    if caps.ndim == 0:
        caps = np.full(n, float(caps))
    if (caps.ndim != 1 or caps.size != n or not np.all(np.isfinite(caps))
            or np.any(caps < 0)):
        return _refusal(
            loss=loss, alpha=alpha, n_scores=n, outcome_scale=outcome_scale,
            cap_source=cap_source,
            reason="second-moment caps must be finite, nonnegative, and aligned with scores",
            referral="supply one valid cap per independent score or one uniform scalar cap")
    total_m2 = float(np.sum(caps))
    if total_m2 == 0:
        if np.any(values > 0):
            return _refusal(
                loss=loss, alpha=alpha, n_scores=n, outcome_scale=outcome_scale,
                cap_source=cap_source,
                reason="positive scores contradict a declared zero second-moment cap",
                referral="correct the cap or score provenance")
        chosen_eta = 0.
        lower = observed = 0.
        effective_shortfall = 0.
    else:
        chosen_eta = (math.sqrt(2. * math.log(1. / alpha) / total_m2)
                      if eta is None else float(eta))
        if not math.isfinite(chosen_eta) or chosen_eta <= 0:
            raise ValueError("eta must be finite and positive")
        z = chosen_eta * values
        # Stable log(1 + z + z^2/2), including z=0 and very large finite z.
        log_z = np.full_like(z, -np.inf)
        positive = z > 0
        log_z[positive] = np.log(z[positive])
        psi = np.logaddexp(np.logaddexp(0., log_z), math.log(.5) + 2. * log_z)
        raw = ((float(np.sum(psi)) - .5 * chosen_eta ** 2 * total_m2
                - math.log(1. / alpha)) / (chosen_eta * n))
        lower = float(max(0., raw))
        observed = float(np.mean(values))
        effective_shortfall = float(observed - lower)

    return MomentSAAResult(
        status="conditional", claim_tier="conditional", loss=loss,
        lower=lower, observed_mean=observed, radius=effective_shortfall,
        alpha=float(alpha), n_scores=n, total_variance_bound=total_m2,
        outcome_scale=outcome_scale, cap_source=cap_source,
        assumptions=(
            "score blocks are mutually independent but need not be identically distributed or share a common mean",
            "each score is a nonnegative empirical optimum whose expectation does not exceed its block Bayes risk",
            "the declared cap bounds each score second moment and was frozen independently of evaluation outcomes",
            "the reported target is the unweighted average of the block Bayes risks",
        ),
        diagnostics={
            "inequality": "Catoni polynomial-soft-truncation e-value plus Markov",
            "psi": "log(1+x+x^2/2)",
            "eta": chosen_eta,
            "eta_source": "declared-moment optimum" if eta is None else "externally supplied",
            "original_scale": True,
            "clipping_used": False,
            "cap_type": "second_moment",
            "cap_is_assumed_not_estimated": True,
            "common_mean_required": False,
        })


def location_saa_score_variance_cap(
        *, loss: str, block_size: int, first_absolute_moment: float,
        second_moment: float, fourth_moment: float | None = None,
        quantile: float | None = None) -> dict:
    """Derive a score-variance cap from moments about a fixed action.

    The declared moments bound ``E|Y-a0|``, ``E[(Y-a0)^2]``, and, for MSE,
    ``E[(Y-a0)^4]`` uniformly within a block.  Outcomes within a score block
    must be independent.  The result bounds the second moment of the empirical
    optimum, hence also its variance.
    """
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    m1, m2 = float(first_absolute_moment), float(second_moment)
    if (not math.isfinite(m1) or not math.isfinite(m2) or m1 < 0 or m2 < 0
            or m2 + 1e-12 < m1 * m1):
        raise ValueError("moments must be finite, nonnegative, and satisfy m2 >= m1^2")

    key = loss.strip().lower().replace("_", "")
    q = quantile
    if key.startswith("q") and len(key) > 1:
        try:
            q = float(key[1:]) / 100.
            key = "pinball"
        except ValueError:
            pass
    if key in {"quantile", "pinball"}:
        if q is None or not 0 < float(q) < 1:
            raise ValueError("pinball loss requires a quantile strictly between zero and one")
        coefficient = max(float(q), 1. - float(q))
        second_score_moment = coefficient ** 2 * (
            m2 / block_size + (block_size - 1.) / block_size * m1 * m1)
        normalized_loss = f"Q{100 * float(q):g}"
        moments_used = ("first_absolute_moment", "second_moment")
    elif key in {"mae", "absolute", "absoluteerror"}:
        coefficient = 1.
        second_score_moment = (
            m2 / block_size + (block_size - 1.) / block_size * m1 * m1)
        normalized_loss = "MAE"
        moments_used = ("first_absolute_moment", "second_moment")
    elif key in {"mse", "squared", "squarederror"}:
        if fourth_moment is None:
            raise ValueError("MSE requires a declared finite fourth moment")
        m4 = float(fourth_moment)
        if not math.isfinite(m4) or m4 < 0 or m4 + 1e-12 < m2 * m2:
            raise ValueError("fourth moment must be finite and satisfy m4 >= m2^2")
        coefficient = 1.
        second_score_moment = (
            m4 / block_size + (block_size - 1.) / block_size * m2 * m2)
        normalized_loss = "MSE"
        moments_used = ("second_moment", "fourth_moment")
    else:
        raise ValueError("supported losses are MAE, MSE, and pinball/quantile")

    return {
        "loss": normalized_loss,
        "block_size": block_size,
        "variance_bound": float(second_score_moment),
        "second_score_moment_bound": float(second_score_moment),
        "loss_lipschitz_coefficient": coefficient,
        "moments_used": moments_used,
        "assumptions": (
            "moments are uniform bounds about one fixed action chosen without evaluation outcomes",
            "outcomes within the SAA score block are independent",
            "the empirical action minimizes the declared loss over an action class containing the fixed action",
        ),
        "derivation": "Var(S)<=E[S^2]<=E[(mean loss(Y,a0))^2]",
    }
