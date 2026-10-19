"""Adversarial shared-bias validation for the MAE sensitivity correction."""
from __future__ import annotations

from math import pi, sqrt

import numpy as np

from pythia.contracts import GuardSpec
from pythia.core.guarded import guarded_mae, mae_shared_bias_sensitivity
from .generators.guarded_designs import guarded_design
from .guarded_runner import _wilson_lower


def run_mae_shared_bias_suite(
    guard: GuardSpec,
    *,
    n_designs: int = 500,
    seed_offset: int = 30_000,
    shared_bias_levels: tuple[float, ...] = (0.0, 3.0, 4.0, 5.0, 7.0, 10.0),
) -> dict:
    rows = []
    factor_rows = []
    for level in shared_bias_levels:
        population_mad_cap = abs(level) * 0.35 * sqrt(2.0 / pi)
        for seed in range(seed_offset, seed_offset + n_designs):
            design = guarded_design(seed, shared_bias=level)
            try:
                base = guarded_mae(design.y, design.predictions, guard)
                sensitivity = mae_shared_bias_sensitivity(
                    base, np.asarray([population_mad_cap]), use_stratified=False,
                )
                factor_sensitivity = mae_shared_bias_sensitivity(
                    base,
                    np.asarray([0.0, 0.5, 1.0, 1.5]) * population_mad_cap,
                    use_stratified=False,
                )
            except ValueError as exc:
                rows.append({"level": level, "refused": True, "reason": str(exc)})
                continue
            for rung in ("calibrated_2d", "proved_qcap", "worst_case_sqrt2"):
                raw = base["global"]["rungs"][rung]
                adjusted = sensitivity["curve"][rung][0]["bound"]
                rows.append({
                    "level": level, "rung": rung, "refused": False,
                    "raw_valid": bool(raw <= design.r_star),
                    "adjusted_valid": bool(adjusted <= design.r_star),
                    "raw_bound": raw, "adjusted_bound": adjusted,
                    "r_star": design.r_star, "common_bias_mad_cap": population_mad_cap,
                })
                for factor, point in zip(
                    (0.0, 0.5, 1.0, 1.5), factor_sensitivity["curve"][rung]
                ):
                    factor_rows.append({
                        "level": level, "rung": rung, "cap_factor": factor,
                        "valid": bool(point["bound"] <= design.r_star),
                    })
    evaluated = [row for row in rows if not row["refused"]]
    by_rung = {}
    for rung in ("calibrated_2d", "proved_qcap", "worst_case_sqrt2"):
        selected = [row for row in evaluated if row["rung"] == rung]
        raw = sum(row["raw_valid"] for row in selected)
        adjusted = sum(row["adjusted_valid"] for row in selected)
        by_rung[rung] = {
            "total": len(selected), "raw_valid": raw,
            "adjusted_valid": adjusted,
            "adjusted_wilson_lower_95": _wilson_lower(adjusted, len(selected)),
        }
    by_factor = {}
    for factor in (0.0, 0.5, 1.0, 1.5):
        selected = [row for row in factor_rows if row["cap_factor"] == factor]
        valid = sum(row["valid"] for row in selected)
        by_factor[str(factor)] = {
            "valid": valid, "total": len(selected),
            "validity_rate": valid / len(selected),
        }
    return {
        "status": "exploratory_validation", "n_designs_per_level": n_designs,
        "levels": list(shared_bias_levels), "by_rung": by_rung,
        "by_cap_factor": by_factor,
        "claim": "known shared-bias MAD simulation; validates sensitivity algebra, not cap estimation",
    }
