"""Independent outcome-mechanism stress suite for frozen Peras v5.7.

The ten recipes below are not in the v5 development registry.  Clinical note
sentences and their cached, outcome-blind concept readings are reused so every
comparator receives exactly the same text information.  Each conditional law
is location-scale, giving an exact Bayes target through gold.pointwise.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

import data5
import gold
import methods as M
import peras5 as P
from core import LOSSES


NAMES = {
    1: "linear Gaussian",
    2: "Friedman-1 Gaussian",
    3: "checkerboard Laplace",
    4: "smooth heteroscedastic Gaussian",
    5: "sparse 30-D Student-t5",
    6: "text interaction Laplace",
    7: "Friedman-1 skew lognormal",
    8: "text-dependent mixture",
    9: "nonlinear Student-t3",
    10: "discrete replicate Gamma/normal",
}


def make(design: int, n: int, seed: int):
    base = 5 if design == 5 else 10 if design == 10 else 1
    d = data5.make(base, n, seed + 450_000 + design * 101)
    rng = np.random.default_rng(seed + 790_000 + design * 701)
    x = d["tab"][:, 1:]
    u = norm.cdf(x)
    c, r, k = d["concepts"].T
    b = d["tab"][:, 0]
    fam = np.full(n, "normal", dtype=object)
    if design == 1:
        mu = 100 + 0.40 * b + 12 * x[:, 0] - 9 * x[:, 1]
        s = np.full(n, 19.0)
    elif design == 2:
        mu = 100 + 8 * np.sin(np.pi * u[:, 0] * u[:, 1]) + 25 * (u[:, 2] - 0.5) ** 2 + 10 * u[:, 3] + 5 * u[:, 4]
        s = np.full(n, 22.0)
    elif design == 3:
        mu = 90 + 0.4 * b + 34 * ((x[:, 0] > 0) ^ (x[:, 1] > 0)) + 15 * (x[:, 2] > 0)
        s = np.full(n, 17.0)
        fam[:] = "laplace"
    elif design == 4:
        mu = 110 + 0.3 * b + 20 * np.sin(2 * x[:, 0]) + 8 * x[:, 1] * x[:, 2]
        s = 8 + 32 / (1 + np.exp(-2 * x[:, 3]))
    elif design == 5:
        mu = 120 + 0.25 * b + 16 * x[:, 0] + 14 * x[:, 1] * x[:, 2] + 10 * np.sin(x[:, 3])
        s = 12 + 9 * (x[:, 4] > 0)
        fam[:] = "t5"
    elif design == 6:
        mu = 85 + 0.45 * b + 28 * c * r + 18 * k + 7 * x[:, 0]
        s = 13 + 5 * c
        fam[:] = "laplace"
    elif design == 7:
        mu = 90 + 0.25 * b + 18 * np.sin(np.pi * u[:, 0] * u[:, 1]) + 12 * u[:, 2]
        s = 14 + 16 * u[:, 3]
        fam[:] = "lognorm"
    elif design == 8:
        mu = 95 + 0.30 * b + 22 * c + 12 * x[:, 0]
        s = np.full(n, 18.0)
        fam[:] = np.where(k == 1, "mix35", "mix05")
    elif design == 9:
        mu = 105 + 0.25 * b + 26 * np.sin(x[:, 0] * x[:, 1]) + 12 * np.maximum(x[:, 2], 0) ** 2
        s = 12 + 10 * (x[:, 3] > 0)
        fam[:] = "t3"
    elif design == 10:
        code = d["code"]
        mu = 70 + 3.2 * code + 13 * c
        s = 9 + (code % 5) * 4.0
        fam[:] = np.where(code % 2 == 0, "gamma2", "normal")
    else:
        raise ValueError(design)
    z = np.empty(n)
    for f in np.unique(fam):
        mask = fam == f
        z[mask] = gold.sample(f, int(mask.sum()), rng)
    d.update(y=mu + s * z, mu=mu, s=s, fam=np.asarray(fam, dtype=str))
    return d


def run(design: int, seed: int, n: int):
    started = time.time()
    d = make(design, n, seed)
    Gp = gold.pointwise(d["fam"], d["s"])
    G = {loss: float(Gp[loss].mean()) for loss in LOSSES}
    E, B, cols = P.peras57(d, seed=seed)
    R = np.column_stack([d["tab"], d["llm"]])
    R = (R - R.mean(axis=0)) / (R.std(axis=0) + 1e-8)
    knn = {str(k): M.knn_loo(R, d["y"], k=k) for k in (5, 20, 50)}
    rice = M.rice_1nn(R, d["y"])
    gamma = M.gamma_test(R, d["y"])
    pred = M.crossfit(R, d["y"], K=3, seed=seed, TF=None)
    oof = M.oof_risk(d["y"], pred)
    lc = M.cortes_curve(R, d["y"], None, seed, pred, K=3)
    out = dict(design=design, name=NAMES[design], seed=seed, n=n, gold=G,
               peras=E, bound=B, knn=knn, rice=rice, gamma=gamma,
               gbm_oof=oof, learning_curve=lc, selected_columns=cols,
               seconds=round(time.time() - started, 2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("design", type=int, nargs="?")
    ap.add_argument("seed", type=int, nargs="?")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.out is not None and a.out.exists():
        ap.error(f"output already exists: {a.out}; choose a new file")
    jobs = [(design, 27001 + rep) for rep in range(a.reps) for design in range(1, 11)] if a.all else [(a.design, a.seed)]
    for design, seed in jobs:
        row = run(design, seed, a.n)
        if a.out is not None:
            with a.out.open("a") as handle:
                handle.write(json.dumps(row, default=float) + "\n")
        print(json.dumps({"design": design, "seed": seed, "seconds": row["seconds"],
                          "MAE_gold": row["gold"]["MAE"], "MAE_peras": row["peras"]["MAE"]}), flush=True)
