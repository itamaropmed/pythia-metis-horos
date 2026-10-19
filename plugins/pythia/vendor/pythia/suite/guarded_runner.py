"""Held-out synthetic gate for the frozen MAE guard."""
from __future__ import annotations

import numpy as np

from pythia.contracts import GuardSpec
from pythia.core.guarded import guarded_mae
from .generators.guarded_designs import guarded_design


def _wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total == 0:
        return 0.0
    p = successes / total
    den = 1 + z * z / total
    center = p + z * z / (2 * total)
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return float((center - radius) / den)


def run_guarded_suite(guard: GuardSpec, n_designs: int = 500,
                      seed_offset: int = 10_000,
                      shared_bias: float = 0.0) -> dict:
    rows = []
    for seed in range(seed_offset, seed_offset + n_designs):
        design = guarded_design(seed, shared_bias=shared_bias)
        try:
            result = guarded_mae(design.y, design.predictions, guard)["global"]
            bound = result["rungs"]["calibrated_2d"]
            valid = bool(bound <= design.r_star)
            row = {"design_id": design.design_id, "r_star": design.r_star,
                   "bound": bound, "valid": valid, "refused": False,
                   "tightness": bound / design.r_star, "meta": design.meta}
        except ValueError as exc:
            row = {"design_id": design.design_id, "valid": False, "refused": True,
                   "reason": str(exc), "meta": design.meta}
        rows.append(row)
    evaluated = [row for row in rows if not row["refused"]]
    valid = sum(row["valid"] for row in evaluated)
    lower = _wilson_lower(valid, len(evaluated))
    return {
        "summary": {
            "n_designs": n_designs, "evaluated": len(evaluated),
            "refused": n_designs - len(evaluated), "valid": valid,
            "validity_rate": valid / len(evaluated) if evaluated else 0.0,
            "wilson_lower_95": lower,
            "release_gate_passed": bool(evaluated and lower >= 0.95),
            "shared_bias": shared_bias, "seed_offset": seed_offset,
        },
        "failures": [row for row in rows if not row["valid"]],
    }

