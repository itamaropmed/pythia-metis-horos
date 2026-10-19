"""Synthetic designs whose irreducible risk R* is known by construction.

These are the ground truth the Bregman branch is validated against. Every design
is a pure function of its integer seed, so the whole suite regenerates
bit-identically -- which is what makes the certification gate re-runnable.

Regression designs
    Y = m(X) + sigma(X) * eps,   Var(eps) = 1   ->   R*_mse = E[sigma(X)^2]

Classification designs
    Y ~ Bernoulli(p(X))                          ->   R*_logloss = E[H(p(X))]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
#  mean functions
# --------------------------------------------------------------------------- #


def _m_linear(X, w):
    return X @ w


def _m_nonlinear(X, w):
    return np.sin(1.5 * X[:, 0]) * 3.0 + 0.8 * X[:, 1] ** 2 + X @ w * 0.5


def _m_interaction(X, w):
    return 2.0 * X[:, 0] * X[:, 1] + 1.5 * np.tanh(X[:, 2]) + X @ w * 0.3


def _m_step(X, w):
    return 3.0 * (X[:, 0] > 0.3) - 2.0 * (X[:, 1] < -0.5) + X @ w * 0.4


MEANS = {"linear": _m_linear, "nonlinear": _m_nonlinear,
         "interaction": _m_interaction, "step": _m_step}

# --------------------------------------------------------------------------- #
#  noise laws, all standardised to unit variance
# --------------------------------------------------------------------------- #


def _noise(rng, law: str, n: int) -> np.ndarray:
    if law == "gauss":
        e = rng.standard_normal(n)
    elif law == "t5":
        e = rng.standard_t(5, n)
        e /= np.sqrt(5 / 3)
    elif law == "laplace":
        e = rng.laplace(size=n)
        e /= np.sqrt(2.0)
    elif law == "mix":                      # two-component scale mixture
        z = rng.random(n) < 0.15
        e = np.where(z, rng.standard_normal(n) * 3.0, rng.standard_normal(n))
        e /= np.sqrt(0.15 * 9 + 0.85)
    else:
        raise ValueError(law)
    return e - e.mean()


# --------------------------------------------------------------------------- #
#  designs
# --------------------------------------------------------------------------- #


@dataclass
class Design:
    design_id: str
    task: str                # 'regression' | 'classification'
    X: np.ndarray
    y: np.ndarray
    r_star: float
    meta: dict


def regression_design(seed: int) -> Design:
    rng = np.random.default_rng(seed)
    n = int(rng.choice([800, 1500, 3000, 6000]))
    d = int(rng.choice([4, 8, 15]))
    shape = str(rng.choice(list(MEANS)))
    law = str(rng.choice(["gauss", "t5", "laplace", "mix"]))
    hetero = bool(rng.random() < 0.5)
    snr = float(rng.choice([0.5, 1.0, 2.0, 4.0]))

    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d) / np.sqrt(d)
    m = MEANS[shape](X, w)
    m = (m - m.mean()) / (m.std() + 1e-12)

    if hetero:
        s = 0.5 + 1.2 * np.abs(X[:, 0])
        s = s / np.sqrt(np.mean(s ** 2))
    else:
        s = np.ones(n)
    sigma = s / np.sqrt(snr)

    y = m + sigma * _noise(rng, law, n)
    r_star = float(np.mean(sigma ** 2))

    return Design(
        design_id=f"reg-{seed:04d}",
        task="regression",
        X=X, y=y, r_star=r_star,
        meta=dict(n=n, d=d, shape=shape, law=law, hetero=hetero, snr=snr),
    )


def classification_design(seed: int) -> Design:
    rng = np.random.default_rng(10_000 + seed)
    n = int(rng.choice([1500, 3000, 6000]))
    d = int(rng.choice([4, 8, 15]))
    shape = str(rng.choice(list(MEANS)))
    strength = float(rng.choice([0.6, 1.2, 2.5]))

    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d) / np.sqrt(d)
    g = MEANS[shape](X, w)
    g = (g - g.mean()) / (g.std() + 1e-12) * strength
    p = 1.0 / (1.0 + np.exp(-g))
    y = (rng.random(n) < p).astype(float)

    q = np.clip(p, 1e-12, 1 - 1e-12)
    r_star = float(np.mean(-(q * np.log(q) + (1 - q) * np.log(1 - q))))

    return Design(
        design_id=f"clf-{seed:04d}",
        task="classification",
        X=X, y=y, r_star=r_star,
        meta=dict(n=n, d=d, shape=shape, strength=strength, base_rate=float(y.mean())),
    )


def suite(n_reg: int = 60, n_clf: int = 40) -> list[Design]:
    """The M0 certification suite: deterministic given the counts."""
    return ([regression_design(s) for s in range(n_reg)]
            + [classification_design(s) for s in range(n_clf)])
