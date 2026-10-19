"""Loss registry.

Every loss the system supports is declared here once, with the three facts the
rest of the pipeline needs:

  * which **family** it belongs to -- this is what S1 routes on;
  * what its **Bayes act** is (mean / median / quantile / distribution / mode);
  * for Bregman losses, the **divergence** ``D(u, v)``, which is what makes the
    bound exact rather than calibrated.

Nothing in this module knows about models, folds or bounds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

EPS = 1e-12

# --------------------------------------------------------------------------- #
#  divergences
# --------------------------------------------------------------------------- #


def d_squared(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bregman divergence of phi(u) = u**2."""
    return (u - v) ** 2


def d_logloss(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Binary KL -- the Bregman divergence of negative entropy.

    With ``u`` in {0, 1} this reduces to the usual log loss.
    """
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, EPS, 1.0 - EPS)
    out = np.zeros_like(v, dtype=float)
    m = u > 0
    out[m] += u[m] * np.log(u[m] / v[m])
    m = u < 1
    out[m] += (1 - u[m]) * np.log((1 - u[m]) / (1 - v[m]))
    return out


def d_poisson(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Poisson deviance / 2 -- Bregman divergence of phi(u) = u log u - u."""
    u = np.maximum(u, 0.0)
    v = np.maximum(v, EPS)
    out = v - u
    m = u > 0
    out[m] += u[m] * np.log(u[m] / v[m])
    return out


def d_gamma(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Gamma deviance / 2 -- Bregman divergence of phi(u) = -log u."""
    u = np.maximum(u, EPS)
    v = np.maximum(v, EPS)
    return u / v - np.log(u / v) - 1.0


# --------------------------------------------------------------------------- #
#  pointwise losses (used when the target is the observation, not a mean)
# --------------------------------------------------------------------------- #


def _pinball(alpha: float) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def f(y: np.ndarray, a: np.ndarray) -> np.ndarray:
        r = y - a
        return np.maximum(alpha * r, (alpha - 1.0) * r)

    return f


# --------------------------------------------------------------------------- #
#  the registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Loss:
    name: str
    family: str                    # 'bregman' | 'quantile' | 'unsupported'
    bayes_act: str                 # 'mean' | 'median' | 'quantile' | 'mode'
    pointwise: Callable            # (y, a) -> elementwise loss
    divergence: Callable | None    # D(u, v), Bregman family only
    monotone_map: Callable | None  # applied to the risk AFTER bounding, e.g. sqrt for RMSE
    note: str = ""

    # ---- risk helpers -----------------------------------------------------
    def risk(self, y: np.ndarray, a: np.ndarray) -> float:
        return float(np.mean(self.pointwise(np.asarray(y, float), np.asarray(a, float))))

    def zero_info(self, y: np.ndarray) -> float:
        """Risk of the best constant predictor -- the reference every number is read against."""
        y = np.asarray(y, float)
        if self.bayes_act == "mean":
            a = y.mean()
        elif self.bayes_act == "median":
            a = np.median(y)
        else:                                   # quantile
            a = np.quantile(y, self.alpha or 0.5)
        return float(np.mean(self.pointwise(y, np.full_like(y, a))))

    @property
    def alpha(self) -> float | None:
        if self.name.startswith("pinball"):
            return float(self.name.split("@")[1])
        return None

    @property
    def is_bregman(self) -> bool:
        return self.family == "bregman"


_REGISTRY: dict[str, Loss] = {}


def _reg(loss: Loss) -> Loss:
    _REGISTRY[loss.name] = loss
    return loss


_reg(Loss("mse", "bregman", "mean", d_squared, d_squared, None,
          "exact: q = 1, no envelope"))
_reg(Loss("rmse", "bregman", "mean", d_squared, d_squared, np.sqrt,
          "bound MSE, then take the square root (monotone)"))
_reg(Loss("logloss", "bregman", "mean", d_logloss, d_logloss, None,
          "binary cross-entropy; R* is the conditional entropy H(Y|X)"))
_reg(Loss("cross_entropy", "bregman", "mean", d_logloss, d_logloss, None,
          "alias of logloss"))
_reg(Loss("poisson", "bregman", "mean", d_poisson, d_poisson, None, ""))
_reg(Loss("gamma", "bregman", "mean", d_gamma, d_gamma, None, ""))

_reg(Loss("mae", "quantile", "median", lambda y, a: np.abs(y - a), None, None,
          "guarded branch: needs the measured envelope q(rho, tau)"))
for _a in (0.1, 0.25, 0.5, 0.75, 0.9):
    _reg(Loss(f"pinball@{_a}", "quantile", "quantile", _pinball(_a), None, None,
              "guarded branch, envelope recalibrated per alpha"))

for _n in ("accuracy", "zero_one", "auc", "f1", "ndcg"):
    _reg(Loss(_n, "unsupported", "mode",
              lambda y, a: (np.asarray(y) != np.asarray(a)).astype(float), None, None,
              "not certifiable by this method -- see pythia.core.router for the referral"))


def get_loss(name: str) -> Loss:
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown loss {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def available() -> list[str]:
    return sorted(_REGISTRY)
