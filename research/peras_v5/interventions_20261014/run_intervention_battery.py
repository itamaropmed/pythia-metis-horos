"""Frozen run registered in INTERVENTION_BATTERY_REGISTRATION.md."""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.stats import norm, t

from peras import finite_cell_saa_lower_bound


MECHANISMS = (
    "gaussian", "uniform", "triangular", "t3", "lognormal", "bimodal",
    "heteroscedastic", "high_snr", "low_snr", "discrete_mixture",
)
TRANSFORMS = (
    "baseline", "bijective", "nuisance_canonical", "nuisance_naive",
    "family_garble", "within_family_reassign", "global_misalign",
    "token_shuffle_bow", "literal_target", "unary_target",
)


def _laws(name: str, grid_size: int = 4096):
    u = (np.arange(grid_size) + .5) / grid_size
    z = np.arange(20)
    x = (z - 9.5) / 9.5
    location = 1.2 * x + .5 * np.sin(2 * np.pi * x)
    scale = np.ones(20)
    if name == "gaussian":
        noise = np.clip(norm.ppf(u), -3, 3)
    elif name == "uniform":
        noise = 2 * u - 1
    elif name == "triangular":
        noise = np.where(u < .5, np.sqrt(2 * u) - 1,
                         1 - np.sqrt(2 * (1 - u)))
    elif name == "t3":
        noise = np.clip(t.ppf(u, 3) / np.sqrt(3), -4, 4)
    elif name == "lognormal":
        raw = np.exp(norm.ppf(u))
        noise = np.clip((raw - np.exp(.5)) / np.sqrt((np.e - 1) * np.e), -2, 5)
    elif name == "bimodal":
        half = np.where(u < .5, norm.ppf(2 * u), norm.ppf(2 * u - 1))
        noise = np.clip(half + np.where(u < .5, -1.2, 1.2), -4, 4)
    elif name == "heteroscedastic":
        noise = np.clip(norm.ppf(u), -3, 3)
        scale = .25 + 1.5 * np.abs(x)
    elif name == "high_snr":
        noise = np.clip(norm.ppf(u), -3, 3)
        scale[:] = .15
    elif name == "low_snr":
        noise = np.clip(norm.ppf(u), -3, 3)
        scale[:] = 2.0
    elif name == "discrete_mixture":
        noise = np.select([u < .15, u < .85], [-2.0, 0.0], default=2.0)
    else:
        raise ValueError(name)
    return location[:, None] + scale[:, None] * noise[None, :]


def _sample(name: str, cells, rng):
    n = len(cells)
    u = rng.random(n)
    laws = _laws(name)
    index = np.minimum((u * laws.shape[1]).astype(int), laws.shape[1] - 1)
    return laws[cells, index]


def _risk(values):
    return float(np.mean(np.abs(values - np.median(values))))


def _gold_tables(name):
    laws = _laws(name)
    by_cell = np.array([_risk(row) for row in laws])
    by_family = np.array([_risk(laws[4 * f:4 * f + 4].reshape(-1)) for f in range(5)])
    return laws, by_cell, by_family, _risk(laws.reshape(-1))


def _labels(transform, cells, outcomes, rng):
    n = len(cells)
    if transform in {"baseline", "nuisance_canonical", "token_shuffle_bow"}:
        return cells.astype(str)
    if transform == "bijective":
        return np.array([f"cipher-{37 * int(v) + 11}" for v in cells])
    if transform == "nuisance_naive":
        nuisance = rng.integers(0, 64, n)
        return np.array([f"cell-{c}|random-{r}" for c, r in zip(cells, nuisance)])
    if transform == "family_garble":
        return (cells // 4).astype(str)
    if transform == "within_family_reassign":
        reassigned = 4 * (cells // 4) + rng.integers(0, 4, n)
        return reassigned.astype(str)
    if transform == "global_misalign":
        return rng.integers(0, 20, n).astype(str)
    if transform == "literal_target":
        return np.array([f"actual-duration={value:.12g}" for value in outcomes])
    if transform == "unary_target":
        rank = np.argsort(np.argsort(outcomes, kind="stable"), kind="stable")
        return np.array(["X" * (int(v) + 1) for v in rank])
    raise ValueError(transform)


def _gold(transform, labels, cells, by_cell, by_family, global_risk):
    if transform in {"baseline", "bijective", "nuisance_canonical",
                     "nuisance_naive", "token_shuffle_bow"}:
        return float(np.mean(by_cell[cells]))
    if transform in {"family_garble", "within_family_reassign"}:
        return float(np.mean(by_family[cells // 4]))
    if transform == "global_misalign":
        return float(global_risk)
    return 0.0


def _external_estimate(train_labels, train_y, test_labels, test_y):
    fallback = float(np.median(train_y))
    actions = {label: float(np.median(train_y[train_labels == label]))
               for label in np.unique(train_labels)}
    prediction = np.array([actions.get(label, fallback) for label in test_labels])
    return float(np.mean(np.abs(test_y - prediction)))


def run_one(name, seed):
    rng = np.random.default_rng(seed)
    cells = rng.integers(0, 20, 2048)
    outcomes = _sample(name, cells, rng)
    train = np.arange(2048) < 1024
    _, by_cell, by_family, global_risk = _gold_tables(name)
    support = _laws(name)
    outcome_range = float(support.max() - support.min())
    rows = []
    for j, transform in enumerate(TRANSFORMS):
        transform_rng = np.random.default_rng(seed * 101 + j)
        labels = _labels(transform, cells, outcomes, transform_rng)
        gold = _gold(transform, labels[~train], cells[~train], by_cell, by_family, global_risk)
        target_derived = transform in {"literal_target", "unary_target"}
        result = finite_cell_saa_lower_bound(
            labels[~train], outcomes[~train], np.flatnonzero(~train), loss="mae",
            outcome_range_bound=outcome_range, alpha=.05,
            representation_provenance=f"synthetic-{transform}-v1",
            support_provenance=f"analytic-{name}-support-v1",
            iid_rows_attested=True, outcome_blind_attested=not target_derived,
        )
        estimate = (0.0 if target_derived else _external_estimate(
            labels[train], outcomes[train], labels[~train], outcomes[~train]))
        rows.append({
            "mechanism": name, "seed": seed, "transform": transform,
            "gold": gold, "bound": result.bound, "status": result.status,
            "estimate": estimate, "n_cells": result.n_cells,
            "singleton_fraction": result.singleton_fraction,
            "effective_cell_size": result.effective_cell_size,
            "violation": False if result.bound is None else result.bound > gold + 1e-12,
            "target_derived": target_derived,
        })
    return rows


if __name__ == "__main__":
    output = sys.argv[1]
    with open(output, "w", encoding="utf-8") as handle:
        for mechanism in MECHANISMS:
            for run_seed in range(61001, 61051):
                for row in run_one(mechanism, run_seed):
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(mechanism, flush=True)
