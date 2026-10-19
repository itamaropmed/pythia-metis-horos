"""Model-free conditional permutation test for finite information refinements."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import math
import numpy as np

from .finite_cell_saa import _cell_score


@dataclass(frozen=True)
class FiniteRefinementTest:
    status: str
    claim_tier: str
    loss: str
    n: int
    coarse_cells: int
    refined_cells: int
    coarse_saa_score: float | None
    refined_saa_score: float | None
    observed_gain: float | None
    permutations: int
    p_value: float | None
    null_mean: float | None
    null_quantiles: Mapping[str, float]
    seed: int
    provenance: Mapping[str, object]
    assumptions: tuple[str, ...]
    refusal: Mapping[str, str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _joint_labels(coarse, refinement):
    return np.asarray([f"{left}\x1f{right}" for left, right
                       in zip(coarse, refinement)])


def _saa(labels, outcomes, loss, quantile, window):
    unique, inverse = np.unique(labels, return_inverse=True)
    pieces = []
    for cell in range(len(unique)):
        values = outcomes[inverse == cell]
        pieces.append(len(values) * _cell_score(values, loss, quantile, window))
    return math.fsum(pieces) / len(outcomes)


def finite_refinement_permutation_test(
        coarse_labels, refinement_labels, outcomes, *, loss: str,
        permutations: int = 999, seed: int = 0,
        quantile: float | None = None, window: float | None = None,
        registry_frozen_before_outcomes: bool,
        outcome_blind_attested: bool,
        conditional_exchangeability_attested: bool,
        exploratory_exchangeability_override: bool = False,
        provenance: Mapping[str, object]) -> FiniteRefinementTest:
    """Test whether a finite refinement adds decision-relevant information.

    The statistic is the empirical SAA reduction from coarse cells to joint
    ``(coarse, refinement)`` cells.  Refinement labels are permuted within
    coarse cells.  Under ``Y independent of Z given G`` and conditional
    exchangeability of the registered assignments, the plus-one Monte Carlo
    p-value is finite-sample valid for any loss statistic.

    This is a conditional-independence test, not a lower confidence bound on
    the population information value.
    """
    coarse = np.asarray(coarse_labels).astype(str).reshape(-1)
    refinement = np.asarray(refinement_labels).astype(str).reshape(-1)
    y = np.asarray(outcomes, dtype=float).reshape(-1)
    selected = str(loss).lower()
    n = len(y)
    if (n < 2 or len(coarse) != n or len(refinement) != n
            or not np.isfinite(y).all()):
        raise ValueError("aligned finite labels and outcomes are required")
    if selected not in {"mae", "pinball", "window_miss"}:
        raise ValueError("loss must be mae, pinball, or window_miss")
    if not isinstance(permutations, (int, np.integer)) or permutations < 1:
        raise ValueError("permutations must be a positive integer")
    if selected == "pinball" and (
            quantile is None or not math.isfinite(quantile) or not 0 < quantile < 1):
        raise ValueError("pinball requires a quantile strictly between zero and one")
    if selected == "window_miss" and (
            window is None or not math.isfinite(window) or window <= 0):
        raise ValueError("window miss requires a finite positive half-window")
    if not provenance:
        raise ValueError("provenance is required")
    base = dict(
        loss=selected, n=n, coarse_cells=int(np.unique(coarse).size),
        refined_cells=int(np.unique(_joint_labels(coarse, refinement)).size),
        permutations=int(permutations), seed=int(seed), provenance=dict(provenance))
    missing = []
    if not registry_frozen_before_outcomes:
        missing.append("registry frozen before outcomes")
    if not outcome_blind_attested:
        missing.append("outcome-blind refinement")
    if (not conditional_exchangeability_attested
            and not exploratory_exchangeability_override):
        missing.append("conditional exchangeability")
    if missing:
        return FiniteRefinementTest(
            status="refused", claim_tier="refused", coarse_saa_score=None,
            refined_saa_score=None, observed_gain=None, p_value=None,
            null_mean=None, null_quantiles={}, assumptions=(),
            refusal={
                "reason": "missing required contract: " + ", ".join(missing),
                "referral": "freeze and independently audit the within-stratum permutation law",
            }, **base)

    coarse_score = _saa(coarse, y, selected, quantile, window)
    refined_score = _saa(
        _joint_labels(coarse, refinement), y, selected, quantile, window)
    observed = float(max(0., coarse_score - refined_score))
    groups = [np.flatnonzero(coarse == value) for value in np.unique(coarse)]
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, float)
    for draw in range(permutations):
        permuted = refinement.copy()
        for index in groups:
            permuted[index] = refinement[rng.permutation(index)]
        score = _saa(_joint_labels(coarse, permuted), y, selected, quantile, window)
        null[draw] = max(0., coarse_score - score)
    exceed = int(np.sum(null >= observed - 1e-12))
    p_value = float((1 + exceed) / (permutations + 1))
    q = np.quantile(null, [.5, .9, .95, .99])
    claim_tier = ("conditional" if conditional_exchangeability_attested
                  else "exploratory")
    exchangeability_assumption = (
        "rows are exchangeable within every frozen coarse cell under the null"
        if conditional_exchangeability_attested else
        "conditional exchangeability is not attested; the Monte Carlo tail area is diagnostic, not a valid p-value"
    )
    return FiniteRefinementTest(
        status="complete", claim_tier=claim_tier,
        coarse_saa_score=float(coarse_score), refined_saa_score=float(refined_score),
        observed_gain=observed, p_value=p_value, null_mean=float(np.mean(null)),
        null_quantiles={name: float(value) for name, value in
                        zip(("q50", "q90", "q95", "q99"), q)},
        assumptions=(
            exchangeability_assumption,
            "the refinement registry and statistic were frozen before outcomes",
            "refinement labels were constructed without outcomes or post-outcome information",
            "the null is conditional independence of outcome and refinement given coarse label",
            "the p-value tests signal and is not a confidence bound on population Bayes-risk gain",
        ), refusal=None, **base)
