"""Research-only variance-stabilized SIMEX estimator candidate.

For Gaussian homogeneous residuals, expected MAE/pinball tuple statistics
scale as the square root of total variance, while added SIMEX variance is
linear in 1+lambda. Squaring the statistic before extrapolation is exact in
that idealized case. It is not a one-sided bound or a universal correction.
"""
from __future__ import annotations

import numpy as np

import peras5 as P


def _estimate_from_rows(rows: dict, *, square_losses: set[str]) -> dict:
    out = {}
    for loss in P.LOSSES:
        config = P.CFG54[loss]
        a = np.asarray(rows[loss], float)
        a = a[np.isin(a[:, 0], config["ks"])]
        if P.RIDGE.get("guard"):
            keep = []
            for k in np.unique(a[:, 0]):
                ak = a[a[:, 0] == k]
                cc = config["w"] * ak[:, 2] + (1 - config["w"]) * ak[:, 3]
                g0 = cc[ak[:, 1] == 0].mean()
                g1 = cc[ak[:, 1] == 1].mean()
                if g0 > 0 and (g1 / g0 - 1) < P.RIDGE["guard"]:
                    keep.append(k)
            if len(keep) >= (2 if config["spread"] == "rho4" else 1):
                a = a[np.isin(a[:, 0], keep)]
        c = config["w"] * a[:, 2] + (1 - config["w"]) * a[:, 3]
        X = np.column_stack(
            [np.ones(len(a)), 1 + a[:, 1]]
            + ([a[:, 4]] if config["spread"] == "rho4" else [])
        )
        if loss in square_losses:
            intercept = np.linalg.lstsq(X, c ** 2, rcond=None)[0][0]
            out[loss] = float(np.sqrt(max(intercept, 0.0)))
        else:
            intercept = np.linalg.lstsq(X, c, rcond=None)[0][0]
            out[loss] = float(max(intercept, 0.0))
    out["RMSE"] = float(np.sqrt(out["MSE"]))
    return out


def estimate(d: dict, *, seed: int = 0, lams=(0.0, 0.5, 1.0, 2.0),
             m: int = 60, nsim: int = 2) -> dict:
    """Return frozen-linear and square-SIMEX estimates from identical draws."""
    old = dict(P.RIDGE)
    P.RIDGE.update(relative=0.01, floor=0.01, guard=0.3)
    try:
        rng = np.random.default_rng(seed)
        cols = P.screen_forward(d, rng, tol=0.01)
        dd = dict(d)
        dd["tab"] = d["tab"][:, cols]
        Z = P.represent(dd)
        y = d["y"]
        _, nb = P.neighbours(Z, m)
        rows = {loss: [] for loss in P.LOSSES}
        for k in (3, 4, 6, 8, 12, 16):
            T, Ctr, rho2, _ = P.detrend_full(Z, y, nb, k, m)
            r4 = float((rho2 ** 2).mean())
            for lam in lams:
                for _ in range(nsim if lam > 0 else 1):
                    Tl = P.simex_draw(T, Ctr, lam, rng)
                    for loss in P.LOSSES:
                        r, rt = P.tuple_stats(Tl, loss)
                        rows[loss].append((k, lam, r.mean(), rt.mean(), r4))
        linear = _estimate_from_rows(rows, square_losses=set())
        squared = _estimate_from_rows(rows, square_losses={"MAE", "Q90"})
        return {"linear": linear, "squared": squared,
                "selected_columns": cols}
    finally:
        P.RIDGE.clear()
        P.RIDGE.update(old)
