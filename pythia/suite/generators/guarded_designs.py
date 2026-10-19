"""Deterministic MAE designs with known conditional-median Bayes risk."""
from __future__ import annotations

from dataclasses import dataclass
from math import gamma, pi, sqrt

import numpy as np


@dataclass(frozen=True)
class GuardedDesign:
    design_id: str
    y: np.ndarray
    predictions: np.ndarray
    r_star: float
    meta: dict


def _standard_noise(rng: np.random.Generator, law: str, n: int) -> tuple[np.ndarray, float]:
    if law == "gauss":
        return rng.normal(size=n), sqrt(2 / pi)
    if law == "laplace":
        return rng.laplace(scale=1 / sqrt(2), size=n), 1 / sqrt(2)
    if law in ("t5", "t3"):
        nu = int(law[1:])
        scale = sqrt((nu - 2) / nu)
        e = rng.standard_t(nu, size=n) * scale
        eabs = (2 * sqrt(nu) * gamma((nu + 1) / 2)
                / ((nu - 1) * sqrt(pi) * gamma(nu / 2))) * scale
        return e, eabs
    if law == "mix":
        wide = rng.random(n) < 0.15
        denom = sqrt(0.85 + 0.15 * 9)
        e = np.where(wide, rng.normal(scale=3, size=n), rng.normal(size=n)) / denom
        eabs = sqrt(2 / pi) * (0.85 + 0.15 * 3) / denom
        return e, eabs
    raise ValueError(law)


def guarded_design(seed: int, shared_bias: float = 0.0) -> GuardedDesign:
    rng = np.random.default_rng(50_000 + seed)
    n = int(rng.choice([800, 1500, 3000, 6000]))
    d = int(rng.choice([4, 15, 100]))
    shape = str(rng.choice(["linear", "interaction", "step"]))
    law = str(rng.choice(["gauss", "laplace", "t5", "t3", "mix"]))
    X = rng.normal(size=(n, d))
    if shape == "linear":
        m = X[:, :min(d, 5)].sum(axis=1) / sqrt(min(d, 5))
    elif shape == "interaction":
        m = 2 * X[:, 0] * X[:, 1] + np.tanh(X[:, 2])
    else:
        m = 2 * (X[:, 0] > 0) - 1.5 * (X[:, 1] < -0.5)
    hetero = bool(rng.integers(0, 2))
    sigma = 0.5 + 1.2 * np.abs(X[:, 0]) if hetero else np.ones(n)
    sigma = sigma / np.sqrt(np.mean(sigma ** 2))
    noise, eabs = _standard_noise(rng, law, n)
    y = m + sigma * noise
    r_star = float(np.mean(sigma) * eabs)
    shared = rng.normal(scale=0.35, size=n)
    predictors = []
    for scale in (0.12, 0.2, 0.3, 0.45, 0.6, 0.8):
        individual = rng.normal(scale=scale, size=n)
        predictors.append(m + shared_bias * shared + individual)
    return GuardedDesign(
        f"mae-{seed:04d}", y, np.asarray(predictors), r_star,
        {"n": n, "d": d, "shape": shape, "law": law,
         "heteroskedastic": hetero, "shared_bias": shared_bias},
    )

