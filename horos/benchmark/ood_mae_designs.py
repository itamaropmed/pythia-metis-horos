"""Ten frozen out-of-distribution MAE designs with analytic Bayes risk."""
from __future__ import annotations

import numpy as np


OOD_DESIGNS = (
    "beta_gaussian", "gap_laplace", "oscillatory_uniform",
    "vshape_gaussian", "radial_laplace", "max3_uniform",
    "sparse20_gaussian", "hetero_gaussian", "hetero_laplace",
    "skew_exponential",
)


def ood_conditional_median(name: str, x: np.ndarray) -> np.ndarray:
    """Analytic median, kept separate for generator truth/guard audits."""
    x = np.asarray(x, float)
    if name == "beta_gaussian":
        return 3 * x[:, 0]
    if name == "gap_laplace":
        return 2 * x[:, 0]
    if name == "oscillatory_uniform":
        return np.sin(6 * np.pi * x[:, 0])
    if name == "vshape_gaussian":
        return 2 * np.abs(x[:, 0] - .5)
    if name == "radial_laplace":
        return np.sqrt(np.sum((x - .5) ** 2, axis=1))
    if name == "max3_uniform":
        return np.max(x, axis=1)
    if name == "sparse20_gaussian":
        return 2 * x[:, 0]
    if name == "hetero_gaussian":
        return 2 * x[:, 0]
    if name == "hetero_laplace":
        return x[:, 0] ** 2
    if name == "skew_exponential":
        return x[:, 0]
    raise ValueError(name)


def ood_mae_design(rng: np.random.Generator, name: str, n: int):
    """Return x, y, valid median Lipschitz cap, conditional MAE array."""
    normal_factor = np.sqrt(2 / np.pi)
    if name == "beta_gaussian":
        x = rng.beta(.4, 3., size=(n, 1))
        center = 3 * x[:, 0]
        noise = rng.normal(0, .3, n)
        return x, center + noise, 3., np.full(n, .3 * normal_factor)
    if name == "gap_laplace":
        left = rng.uniform(0, .3, n)
        right = rng.uniform(.7, 1., n)
        x = np.where(rng.random(n) < .5, left, right)[:, None]
        center = 2 * x[:, 0]
        return x, center + rng.laplace(0, .2, n), 2., np.full(n, .2)
    if name == "oscillatory_uniform":
        x = rng.uniform(size=(n, 1))
        center = np.sin(6 * np.pi * x[:, 0])
        return x, center + rng.uniform(-.4, .4, n), 6 * np.pi, np.full(n, .2)
    if name == "vshape_gaussian":
        x = rng.uniform(size=(n, 1))
        center = 2 * np.abs(x[:, 0] - .5)
        return x, center + rng.normal(0, .25, n), 2., np.full(n, .25 * normal_factor)
    if name == "radial_laplace":
        x = rng.uniform(size=(n, 2))
        center = np.sqrt(np.sum((x - .5) ** 2, axis=1))
        return x, center + rng.laplace(0, .15, n), 2., np.full(n, .15)
    if name == "max3_uniform":
        x = rng.uniform(size=(n, 3))
        center = np.max(x, axis=1)
        return x, center + rng.uniform(-.5, .5, n), 3., np.full(n, .25)
    if name == "sparse20_gaussian":
        x = rng.uniform(size=(n, 20))
        center = 2 * x[:, 0]
        return x, center + rng.normal(0, .3, n), 40., np.full(n, .3 * normal_factor)
    if name == "hetero_gaussian":
        x = rng.uniform(size=(n, 1))
        scale = .1 + .6 * x[:, 0]
        return x, 2 * x[:, 0] + rng.normal(0, scale), 2., scale * normal_factor
    if name == "hetero_laplace":
        x = rng.uniform(size=(n, 1))
        scale = .05 + .5 * x[:, 0]
        return x, x[:, 0] ** 2 + rng.laplace(0, scale), 2., scale
    if name == "skew_exponential":
        x = rng.uniform(size=(n, 1))
        center = x[:, 0]
        noise = .2 * (rng.exponential(size=n) - np.log(2))
        return x, center + noise, 1., np.full(n, .2 * np.log(2))
    raise ValueError(name)
