"""Lower confidence bounds on information value using a frozen policy."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import math
import numpy as np

from .concentration import bounded_upper_mean
from .finite_cell_saa import FiniteCellSAAResult, finite_cell_saa_lower_bound


UPPER_MEAN_METHODS = ("Hoeffding", "empirical_Bernstein", "Bentkus", "dKelly")


@dataclass(frozen=True)
class FixedPolicyInformationValue:
    status: str
    claim_tier: str
    loss: str
    value_lower_bound: float | None
    raw_value_endpoint: float | None
    coarse_risk_lower: float | None
    policy_risk_estimate: float | None
    policy_risk_upper: float | None
    selected_upper_method: str | None
    upper_candidates: tuple[Mapping[str, float | str], ...]
    alpha_family: float
    alpha_coarse: float | None
    alpha_per_upper_method: float | None
    n: int
    loss_range_bound: float
    coarse_result: FiniteCellSAAResult | None
    assumptions: tuple[str, ...]
    diagnostics: Mapping[str, object]
    refusal: Mapping[str, str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def fixed_policy_information_value_lower_bound(
        coarse_labels, outcomes, entities, policy_losses, *, loss: str,
        loss_range_bound: float, outcome_range_bound: float | None = None,
        quantile: float | None = None, window: float | None = None,
        alpha: float = .05,
        upper_methods: Sequence[str] = UPPER_MEAN_METHODS,
        representation_provenance: str, support_provenance: str,
        policy_provenance: str,
        iid_rows_attested: bool, outcome_blind_coarse_attested: bool,
        policy_frozen_before_evaluation: bool,
        policy_refinement_measurable_attested: bool,
        policy_preoutcome_attested: bool,
        cell_outcome_range_bounds: Mapping[str, float] | None = None,
        cell_support_provenance: str | None = None,
        exploratory_sampling_override: bool = False,
        ) -> FixedPolicyInformationValue:
    """Lower-bound ``R*(coarse) - R*(refined)`` without fitting a model.

    Any policy measurable from the refined information has risk at least the
    refined Bayes risk. Therefore, on a simultaneous event,
    ``R*(coarse) - R*(refined) >= L_coarse - U_policy``.
    """
    y = np.asarray(outcomes, dtype=float).reshape(-1)
    losses = np.asarray(policy_losses, dtype=float).reshape(-1)
    n = len(y)
    cap = float(loss_range_bound)
    methods = tuple(dict.fromkeys(str(method) for method in upper_methods))
    if (n < 2 or len(losses) != n or not np.isfinite(losses).all()
            or not math.isfinite(cap) or cap <= 0
            or np.any(losses < -1e-12) or np.any(losses > cap + 1e-12)):
        raise ValueError("aligned finite policy losses in [0, loss_range_bound] required")
    if not 0 < alpha < 1 or not methods:
        raise ValueError("valid family alpha and at least one upper method required")
    unknown = tuple(method for method in methods if method not in UPPER_MEAN_METHODS)
    if unknown:
        raise ValueError(f"unsupported upper-mean methods: {unknown}")
    if not policy_provenance:
        raise ValueError("policy provenance is required")
    missing = []
    if not policy_frozen_before_evaluation:
        missing.append("policy frozen before evaluation")
    if not policy_refinement_measurable_attested:
        missing.append("policy measurable from the refined representation")
    if not policy_preoutcome_attested:
        missing.append("pre-outcome policy inputs")
    if missing:
        return FixedPolicyInformationValue(
            status="refused", claim_tier="refused", loss=str(loss).lower(),
            value_lower_bound=None, raw_value_endpoint=None,
            coarse_risk_lower=None, policy_risk_estimate=None,
            policy_risk_upper=None, selected_upper_method=None,
            upper_candidates=(), alpha_family=float(alpha), alpha_coarse=None,
            alpha_per_upper_method=None, n=n, loss_range_bound=cap,
            coarse_result=None, assumptions=(),
            diagnostics={"policy_provenance": policy_provenance},
            refusal={
                "reason": "missing required policy contract: " + ", ".join(missing),
                "referral": "supply an independently frozen refined-information policy",
            })

    alpha_coarse = float(alpha / 2)
    alpha_upper = float(alpha / (2 * len(methods)))
    coarse = finite_cell_saa_lower_bound(
        coarse_labels, y, entities, loss=loss,
        outcome_range_bound=outcome_range_bound, quantile=quantile,
        window=window, alpha=alpha_coarse,
        representation_provenance=representation_provenance,
        support_provenance=support_provenance,
        iid_rows_attested=iid_rows_attested,
        outcome_blind_attested=outcome_blind_coarse_attested,
        cell_outcome_range_bounds=(None if cell_outcome_range_bounds is None
                                   else dict(cell_outcome_range_bounds)),
        cell_support_provenance=cell_support_provenance,
        exploratory_sampling_override=exploratory_sampling_override)
    if coarse.status != "complete":
        return FixedPolicyInformationValue(
            status="refused", claim_tier="refused", loss=str(loss).lower(),
            value_lower_bound=None, raw_value_endpoint=None,
            coarse_risk_lower=None, policy_risk_estimate=None,
            policy_risk_upper=None, selected_upper_method=None,
            upper_candidates=(), alpha_family=float(alpha),
            alpha_coarse=alpha_coarse, alpha_per_upper_method=alpha_upper,
            n=n, loss_range_bound=cap, coarse_result=coarse, assumptions=(),
            diagnostics={"policy_provenance": policy_provenance},
            refusal={"reason": coarse.reason or "coarse endpoint refused",
                     "referral": "repair the coarse-risk contract"})

    candidates = tuple({
        "method": method,
        "upper": bounded_upper_mean(losses, cap, alpha_upper, method),
    } for method in methods)
    selected = min(candidates, key=lambda row: float(row["upper"]))
    upper = float(selected["upper"])
    raw = float(coarse.bound - upper)
    tier = "conditional" if coarse.claim_tier == "conditional" else "exploratory"
    assumptions = coarse.assumptions + (
        "the supplied policy was frozen independently of evaluation outcomes",
        "the policy is measurable using only the declared refined information at decision time",
        "policy losses use the same population, outcome, action space and loss as the coarse endpoint",
        "all upper-mean routes share a Bonferroni event and are selected only after simultaneous construction",
        "the endpoint lower-bounds information value but need not be tight when the supplied policy is suboptimal",
    )
    return FixedPolicyInformationValue(
        status="complete", claim_tier=tier, loss=str(loss).lower(),
        value_lower_bound=float(max(0., raw)), raw_value_endpoint=raw,
        coarse_risk_lower=float(coarse.bound),
        policy_risk_estimate=float(np.mean(losses)), policy_risk_upper=upper,
        selected_upper_method=str(selected["method"]),
        upper_candidates=candidates, alpha_family=float(alpha),
        alpha_coarse=alpha_coarse, alpha_per_upper_method=alpha_upper,
        n=n, loss_range_bound=cap, coarse_result=coarse,
        assumptions=assumptions,
        diagnostics={
            "policy_provenance": policy_provenance,
            "alpha_spent": float(alpha_coarse + len(methods) * alpha_upper),
            "model_training_performed": False,
            "identity": "R_coarse-R_refined >= L_coarse-U_frozen_policy",
            "upper_method_selection": "minimum over simultaneous registered endpoints",
        }, refusal=None)
