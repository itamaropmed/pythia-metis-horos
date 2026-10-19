"""Guarded MAE certificate ported from the frozen Mayo analysis.

The calibrated 2-D envelope is empirical and must never be described as proved.
The Qcap and sqrt(2) rungs are reported separately so claim strength is visible.
"""
from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from pythia.contracts import GuardSpec

SQRT2 = float(np.sqrt(2.0))


def _sha256(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode())
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def load_guard(directory: str | Path, version: str = "mayo-2d-v1") -> GuardSpec:
    directory = Path(directory)
    bound_path = directory / "env2.npy"
    estimator_path = directory / "med2.npy"
    axes_path = directory / "env2_axes.npy"
    paths = [bound_path, estimator_path, axes_path]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"guard artifact is incomplete: {missing}")
    axes = np.load(axes_path)
    library_path = directory / "lib2.json"
    hash_paths = paths + ([library_path] if library_path.exists() else [])
    return GuardSpec(
        version=version,
        kind="measured_envelope_2d",
        bound_grid=np.load(bound_path),
        estimator_grid=np.load(estimator_path),
        rho_domain=(float(axes[0]), float(axes[1])),
        tau_domain=(float(axes[3]), float(axes[4])),
        calibration_population="5,952 shape/scale combinations; frozen legacy library",
        source_hash=_sha256(hash_paths),
        seed_registry="data/derived/mayo/lib2.json",
    )


def q_cap(tau: float) -> float:
    return float(1.0 / np.sqrt((1.0 - tau) ** 2 + tau ** 2))


def _grid_value(grid: np.ndarray, value: float, domain: tuple[float, float]) -> int:
    lo, hi = domain
    return int(np.clip(round((value - lo) / (hi - lo) * (grid - 1)), 0, grid - 1))


def guard_value(spec: GuardSpec, rho: float, tau: float, *, estimator: bool = False) -> float:
    if not (spec.rho_domain[0] <= rho <= spec.rho_domain[1]):
        raise ValueError(f"rho={rho:.6g} outside guard domain {spec.rho_domain}")
    if not (spec.tau_domain[0] <= tau <= spec.tau_domain[1]):
        raise ValueError(f"tau={tau:.6g} outside guard domain {spec.tau_domain}")
    grid = spec.estimator_grid if estimator else spec.bound_grid
    i = _grid_value(grid.shape[0], rho, spec.rho_domain)
    j = _grid_value(grid.shape[1], tau, spec.tau_domain)
    value = float(grid[i, j])
    if not np.isfinite(value):
        raise ValueError("guard grid has no finite value at requested location")
    return value


def mean_absolute_deviation(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(np.mean(np.abs(x - np.median(x))))


def _cell(y: np.ndarray, p: np.ndarray, guard: GuardSpec) -> dict[str, Any]:
    losses = np.mean(np.abs(y[None, :] - p), axis=1)
    best = int(np.argmin(losses))
    L = float(losses[best])
    residual = y - p[best]
    residual_mad = mean_absolute_deviation(residual)
    residual_sd = float(np.std(residual))
    rho = residual_mad / residual_sd if residual_sd > 0 else 1.0
    disagreement = np.array([
        mean_absolute_deviation(p[i] - p[j])
        for i, j in combinations(range(p.shape[0]), 2)
    ])
    m_plus = float(np.max(disagreement))
    m_bar = float(np.mean(disagreement))
    tau = (m_plus / SQRT2) / residual_mad if residual_mad > 0 else 0.0
    q2 = guard_value(guard, rho, tau)
    q50 = guard_value(guard, rho, tau, estimator=True)
    qcap = q_cap(tau)

    def bound(q: float) -> float:
        return float(np.sqrt(max((residual_mad / q) ** 2 - m_plus ** 2, 0.0)))

    estimator = float(np.sqrt(max((L / q50) ** 2 - (m_bar / SQRT2) ** 2, 0.0)))
    return {
        "n": int(y.size), "best_predictor_index": best, "L": L,
        "residual_mad": residual_mad, "rho": float(rho), "tau": float(tau),
        "m_plus": m_plus, "m_bar": m_bar, "q_calibrated": q2,
        "q_estimator": q50, "q_cap": qcap,
        "rungs": {
            "unguarded_exploratory": bound(1.0),
            "calibrated_2d": bound(q2),
            "proved_qcap": bound(qcap),
            "worst_case_sqrt2": bound(SQRT2),
        },
        "rung_claim_tiers": {
            "unguarded_exploratory": "exploratory",
            "calibrated_2d": "calibrated",
            "proved_qcap": "conditional",
            "worst_case_sqrt2": "conditional",
        },
        "estimator": estimator,
    }


def guarded_mae(y: np.ndarray, p: np.ndarray, guard: GuardSpec,
                strata: np.ndarray | None = None, min_n: int = 30) -> dict[str, Any]:
    y = np.asarray(y, float).ravel()
    p = np.atleast_2d(np.asarray(p, float))
    if p.shape[1] != y.size or p.shape[0] < 2:
        raise ValueError("MAE certification needs at least two predictors aligned with y")
    global_result = _cell(y, p, guard)
    result: dict[str, Any] = {"global": global_result, "stratified": None}
    if strata is not None:
        strata = np.asarray(strata)
        if strata.size != y.size:
            raise ValueError("strata does not match y")
        cells = []
        for value in np.unique(strata):
            mask = strata == value
            if int(mask.sum()) < min_n:
                raise ValueError(f"stratum {value!r} has fewer than {min_n} rows")
            cell = _cell(y[mask], p[:, mask], guard)
            cell["value"] = value.item() if isinstance(value, np.generic) else value
            cell["weight"] = float(mask.mean())
            cells.append(cell)
        rung_names = tuple(global_result["rungs"])
        result["stratified"] = {
            "n_cells": len(cells),
            "rungs": {
                name: float(sum(c["weight"] * c["rungs"][name] for c in cells))
                for name in rung_names
            },
            "estimator": float(sum(c["weight"] * c["estimator"] for c in cells)),
            "cells": cells,
        }
    return result


def mae_shared_bias_sensitivity(
    result: dict[str, Any],
    common_bias_mad_caps: np.ndarray,
    *,
    use_stratified: bool = True,
) -> dict[str, Any]:
    """Downgrade MAE floor rungs for predictor bias shared by the pool.

    If a base rung lower-bounds ``MAD(epsilon-b)`` and the common component
    satisfies ``MAD(b) <= C``, subadditivity of MAD gives
    ``R* >= max(base-C, 0)``. This does not estimate C from disagreement; it
    makes the otherwise invisible shared-bias assumption explicit.
    """
    caps = np.asarray(common_bias_mad_caps, float).ravel()
    if caps.size == 0 or np.any(~np.isfinite(caps)) or np.any(caps < 0):
        raise ValueError("common-bias MAD caps must be finite and nonnegative")
    if np.any(np.diff(caps) < 0):
        raise ValueError("common-bias MAD caps must be sorted")
    selected = result.get("stratified") if use_stratified else None
    selected = selected or result["global"]
    rungs = selected["rungs"]
    tier = {
        "unguarded_exploratory": "exploratory",
        "calibrated_2d": "calibrated",
        "proved_qcap": "conditional",
        "worst_case_sqrt2": "conditional",
    }
    curves = {
        name: [
            {"common_bias_mad_cap": float(cap), "bound": float(max(value - cap, 0.0))}
            for cap in caps
        ]
        for name, value in rungs.items()
    }
    return {
        "status": "complete", "curve": curves, "rung_claim_tiers": tier,
        "collapse_caps": {name: float(value) for name, value in rungs.items()},
        "assumptions": (
            "each base rung is valid for the residual after removing the declared common component",
            "within the reported population/strata, common-bias MAD is no greater than the supplied cap",
            "MAD subadditivity; the cap is external and is not identified from model disagreement",
        ),
    }


def cluster_bootstrap_mae(y: np.ndarray, p: np.ndarray, groups: np.ndarray,
                          guard: GuardSpec, strata: np.ndarray | None = None,
                          reps: int = 400, seed: int = 0,
                          alpha: float = 0.05) -> dict[str, Any]:
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    unique = np.unique(groups)
    indices = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    point = guarded_mae(y, p, guard, strata)
    use_strat = strata is not None
    draws = {"bound": [], "estimator": [], "L": []}
    for _ in range(reps):
        selected = rng.choice(unique, size=unique.size, replace=True)
        ix = np.concatenate([indices[g] for g in selected])
        s = None if strata is None else np.asarray(strata)[ix]
        value = guarded_mae(y[ix], p[:, ix], guard, s)
        primary = value["stratified"] if use_strat else value["global"]
        draws["bound"].append(primary["rungs"]["calibrated_2d"])
        draws["estimator"].append(value["global"]["estimator"])
        draws["L"].append(value["global"]["L"])
    out = {}
    for key, values in draws.items():
        values = np.asarray(values)
        point_value = (point["stratified"]["rungs"]["calibrated_2d"]
                       if key == "bound" and use_strat else
                       point["global"]["estimator"] if key == "estimator" else
                       point["global"]["L"])
        out[key] = {
            "point": float(point_value),
            "lo": float(np.quantile(values, alpha / 2)),
            "hi": float(np.quantile(values, 1 - alpha / 2)),
            "q05": float(np.quantile(values, 0.05)),
            "sampling": "entity-clustered percentile bootstrap",
        }
    return out
