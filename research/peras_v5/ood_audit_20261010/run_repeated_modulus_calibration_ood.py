"""Frozen repeated-environment modulus audit.

Registration: ten affine-quantile mechanisms; 80 fixed units; disjoint blocks
of four after sorting the registered scalar feature; 500 independent common-U
environment rows; clip [-5,5]; alpha .05 jointly across absolute/squared
penalties and Hoeffding, empirical Bernstein, and Bentkus; seeds
133001--133100.  Population barycentric costs are analytic and are diagnostics
only.  No evaluation outcome or fitted predictor enters the protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from peras import calibrate_repeated_environment_discrepancy


def mechanism(design: int):
    x = np.linspace(-1, 1, 80)
    note = (np.arange(80) % 3 == 0).astype(float)
    if design == 1:
        a, s = 1.8*x, .8 + .2*x
    elif design == 2:
        a, s = 2*np.sin(2*x), .8 + .3*x*x
    elif design == 3:
        a, s = np.where(x > 0, 1.4, -1.4), .7 + .4*(x > 0)
    elif design == 4:
        a, s = 1.5*x, .5 + 1.2*np.abs(x)
    elif design == 5:
        a, s = 1.8*np.sign(x)*np.sqrt(np.abs(x)), 1.0 + .2*note
    elif design == 6:
        a, s = 1.2*x + .8*note, .7 + .6*note
    elif design == 7:
        a, s = 2.2*x**3, .6 + .8*(x > .5)
    elif design == 8:
        a, s = 1.5*np.sin(4*x) + .4*note, .8 + .3*np.cos(x)**2
    elif design == 9:
        a, s = np.where(np.abs(x) < .2, 0., 2*x), .6 + .9*(np.abs(x) < .2)
    elif design == 10:
        a, s = 1.6*x + .3*np.sin(9*x), .7 + .5*(np.arange(80) % 5 == 0)
    else:
        raise ValueError("design must be 1--10")
    features = np.column_stack([x, note])
    blocks = np.arange(80).reshape(-1, 4)
    return np.asarray(a), np.asarray(s), features, blocks


def absolute_affine_integral(a, b):
    left, right = a - b/2, a + b/2
    no_cross = left * right >= 0
    answer = np.empty_like(a, dtype=float)
    answer[no_cross] = np.abs(a[no_cross])
    crossing = ~no_cross
    answer[crossing] = ((left[crossing]**2 + right[crossing]**2)
                        / (2*np.abs(b[crossing])))
    answer[(np.abs(b) < 1e-15)] = np.abs(a[(np.abs(b) < 1e-15)])
    return answer


def true_costs(location, scale, blocks):
    a = location[blocks]
    b = scale[blocks]
    da = a - a.mean(axis=1, keepdims=True)
    db = b - b.mean(axis=1, keepdims=True)
    absolute = float(np.mean(absolute_affine_integral(da, db)))
    squared = float(np.mean(da*da + db*db/12))
    return absolute, squared


def run(design: int, seed: int):
    location, scale, features, blocks = mechanism(design)
    rng = np.random.default_rng(seed)
    u = rng.random((500, 1))
    replicates = location + scale*(u-.5)
    truth_abs, truth_sq = true_costs(location, scale, blocks)
    result = calibrate_repeated_environment_discrepancy(
        replicates, blocks, features, clip=(-5, 5), alpha_family=.05,
        marginal_contract="target_unit_conditional",
        coupling_provenance="registered common-U target coupling",
        metric_provenance="registered Euclidean [x,note] metric",
        portfolio_frozen_before_replicates=True)
    return {
        "design": design, "seed": seed,
        "true_absolute": truth_abs, "true_squared": truth_sq,
        "absolute_cap": result.absolute_discrepancy_cap,
        "squared_cap": result.squared_discrepancy_cap,
        "absolute_effective_modulus_cap": result.effective_absolute_modulus_cap,
        "squared_effective_modulus_cap": result.effective_squared_modulus_cap,
        "G": result.feature_dispersion_g, "H": result.feature_dispersion_h,
        "absolute_candidates": result.candidates["absolute"],
        "squared_candidates": result.candidates["squared"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists")
    with args.out.open("w") as handle:
        for design in range(1, 11):
            for seed in range(133001, 133101):
                row = run(design, seed)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
        print(args.out)
