"""Second predeclared ten-design challenge; analytic conditional Bayes MAE.

This registry is separate from the first ten designs. It is frozen in source
before running the comparison and deliberately includes difficult regimes.
"""
from __future__ import annotations

import numpy as np


OOD_DESIGNS_SECOND = (
    "rare_wide_outliers", "two_scale_laplace", "central_atom",
    "symmetric_bimodal", "skew_scale_shift", "sparse12_uniform",
    "correlated4_laplace", "corner_beta", "hetero_atom",
    "nonlinear_cusp",
)


def ood_mae_design_second(rng: np.random.Generator, name: str, n: int):
    """Return x in [0,1], outcomes, valid median-L, exact risk at each x.

    L uses mean L1 over all coordinates. All noise distributions are symmetric
    about zero except centered exponential; their median is exactly zero.
    """
    if name == "rare_wide_outliers":
        x = rng.uniform(size=(n, 1))
        wide = rng.random(n) < .02
        noise = np.where(wide, rng.uniform(-3, 3, n), rng.uniform(-.2, .2, n))
        return x, x[:, 0] + noise, 1., np.full(n, .98 * .1 + .02 * 1.5)
    if name == "two_scale_laplace":
        x = rng.uniform(size=(n, 1))
        scale = np.where(rng.random(n) < .5, .05, .5)
        return x, .5 * x[:, 0] + rng.laplace(0, scale), .5, np.full(n, .275)
    if name == "central_atom":
        x = rng.uniform(size=(n, 1))
        u = rng.random(n)
        noise = np.where(u < .6, 0., np.where(u < .8, -1., 1.))
        return x, 2 * x[:, 0] + noise, 2., np.full(n, .4)
    if name == "symmetric_bimodal":
        x = rng.uniform(size=(n, 1))
        sign = np.where(rng.random(n) < .5, -1., 1.)
        noise = sign * (.2 + rng.uniform(0, .1, n))
        return x, x[:, 0] + noise, 1., np.full(n, .25)
    if name == "skew_scale_shift":
        x = rng.uniform(size=(n, 1))
        scale = .05 + .35 * x[:, 0]
        noise = scale * (rng.exponential(size=n) - np.log(2))
        return x, x[:, 0] + noise, 1., scale * np.log(2)
    if name == "sparse12_uniform":
        x = rng.uniform(size=(n, 12))
        noise = rng.uniform(-.4, .4, n)
        return x, 2 * x[:, 0] + noise, 24., np.full(n, .2)
    if name == "correlated4_laplace":
        base = rng.uniform(size=n)
        x = np.column_stack((base, np.clip(base + rng.normal(0, .02, n), 0, 1),
                             rng.uniform(size=n), rng.uniform(size=n)))
        return x, x[:, 0] + x[:, 1] + rng.laplace(0, .2, n), 4., np.full(n, .2)
    if name == "corner_beta":
        x = rng.beta(.3, .3, size=(n, 1))
        noise = rng.uniform(-.6, .6, n)
        return x, 2 * x[:, 0] + noise, 2., np.full(n, .3)
    if name == "hetero_atom":
        x = rng.uniform(size=(n, 1))
        scale = .1 + .8 * x[:, 0]
        u = rng.random(n)
        noise = np.where(u < .5, 0., np.where(u < .75, -scale, scale))
        return x, x[:, 0] + noise, 1., .5 * scale
    if name == "nonlinear_cusp":
        x = rng.uniform(size=(n, 1))
        center = 3 * np.abs(x[:, 0] - .35)
        scale = .05 + .25 * x[:, 0]
        return x, center + rng.laplace(0, scale), 3., scale
    raise ValueError(name)
