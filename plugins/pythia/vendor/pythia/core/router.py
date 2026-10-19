"""S1 -- the loss router.

Everything downstream branches here, so the branch is made explicit and recorded
in the certificate rather than being implied by which code path ran.

Three destinations:

  bregman   the decomposition is an identity; no envelope, no synthetic library,
            no shape ratio. Still assumes fold-error orthogonality (A-fold).
  guarded   absolute / pinball loss; needs the measured envelope q(rho, tau).
  refuse    the method does not apply. Returns a referral, not a number.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from .losses import Loss, get_loss

REFERRALS = {
    "zero_one": "Bayes-error estimators: Ishida et al. (ICLR 2023); Fukunaga-Hummels; "
                "Devroye-Wagner kNN; Tumer-Ghosh.",
    "accuracy": "same as 0-1 loss -- see the Bayes-error estimation literature.",
    "auc": "not pointwise decomposable: there is no conditional Bayes act to bound. "
           "Bound a pointwise surrogate (logloss) instead and report AUC separately.",
    "f1": "not pointwise decomposable; threshold-dependent. Bound logloss instead.",
    "ndcg": "ranking metric, not pointwise decomposable. Bound a pointwise surrogate.",
}


@dataclass(frozen=True)
class Route:
    loss: str
    regime: str            # 'bregman' | 'guarded' | 'refuse'
    bayes_act: str
    guard_required: bool
    assumptions: tuple[str, ...]
    referral: str | None
    reason: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["assumptions"] = list(self.assumptions)
        return d


def route(loss_name: str) -> Route:
    loss: Loss = get_loss(loss_name)

    if loss.family == "bregman":
        return Route(
            loss=loss.name,
            regime="bregman",
            bayes_act=loss.bayes_act,
            guard_required=False,
            assumptions=("A-fold: fold-error orthogonality",),
            referral=None,
            reason=(
                "Bregman divergence: Risk(f) = R* + E[D(m, f)] is an identity, so the "
                "excess risk is exactly the divergence between the Bayes act and the "
                "predictor. No quadrature guard is needed (q = 1)."
            ),
        )

    if loss.family == "quantile":
        return Route(
            loss=loss.name,
            regime="guarded",
            bayes_act=loss.bayes_act,
            guard_required=True,
            assumptions=(
                "A-quad: the quadrature law, guarded by the measured envelope",
                "A-indep: residual independent of model error",
                "A-fold: fold-error exchangeability",
            ),
            referral=None,
            reason=(
                "Absolute / pinball loss does not decompose additively, so the bound "
                "needs the measured envelope q(rho, tau) and inherits its calibration."
            ),
        )

    return Route(
        loss=loss.name,
        regime="refuse",
        bayes_act=loss.bayes_act,
        guard_required=False,
        assumptions=(),
        referral=REFERRALS.get(loss.name, "no applicable method in this system."),
        reason=(
            "This loss is not certifiable by fold disagreement: it is either not "
            "pointwise decomposable, or its Bayes act is the conditional mode, for "
            "which a mature and better-suited literature already exists."
        ),
    )


def describe(loss_name: str) -> str:
    r = route(loss_name)
    lines = [f"loss      : {r.loss}", f"regime    : {r.regime}", f"Bayes act : {r.bayes_act}"]
    if r.regime == "refuse":
        lines += [f"REFUSED   : {r.reason}", f"use instead: {r.referral}"]
    else:
        lines += [f"guard     : {'required' if r.guard_required else 'none (exact)'}"]
        lines += ["assumes   : " + a for a in r.assumptions]
        lines += [f"why       : {r.reason}"]
    return "\n".join(lines)
