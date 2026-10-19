"""Variable-size patient simulations for the bounded-range dependence rung."""
from __future__ import annotations

import numpy as np

from pythia.core.bregman import bregman_bound
from pythia.core.dependence_bounds import (
    catoni_cluster_pairwise_bound,
    empirical_bernstein_patient_pairwise_bound,
)

from .dependence_comparison import _wilson_lower


def _square_variance_uniform_sum(half_widths: tuple[float, ...]) -> float:
    variances = np.asarray([width * width / 3.0 for width in half_widths])
    fourth = sum(width ** 4 / 5.0 for width in half_widths)
    fourth += 6.0 * sum(
        variances[i] * variances[j]
        for i in range(len(variances)) for j in range(i + 1, len(variances))
    )
    return float(fourth - np.sum(variances) ** 2)


def run_patient_range_comparison(*, seed: int = 20260914, repetitions: int = 200) -> dict:
    """Compare valid rungs on iid patients with variable case counts."""
    patient_counts = (500, 2_000, 10_000)
    common_bounds = (0.0, 0.2, 0.5)
    noise_bound = 0.7
    idiosyncratic_bounds = (0.15, 0.25, 0.35)
    r_star = noise_bound ** 2 / 3.0
    seeds = np.random.SeedSequence(seed).spawn(
        len(patient_counts) * len(common_bounds) * repetitions
    )
    rows = []
    cursor = 0
    for n_patients in patient_counts:
        for common_bound in common_bounds:
            common_variance = common_bound ** 2 / 3.0
            error_variances = np.asarray([
                common_variance + width ** 2 / 3.0 for width in idiosyncratic_bounds
            ])
            lambda_cap = max(
                common_variance / np.sqrt(error_variances[i] * error_variances[j])
                if common_variance else 0.0
                for i in range(3) for j in range(i + 1, 3)
            )
            error_cap = noise_bound + common_bound + max(idiosyncratic_bounds)
            difference_cap = sum(sorted(idiosyncratic_bounds)[-2:])
            risk_variance_cap = max(
                _square_variance_uniform_sum((noise_bound, common_bound, width))
                for width in idiosyncratic_bounds
            ) * 1.001
            disagreement_variance_cap = max(
                _square_variance_uniform_sum((idiosyncratic_bounds[i], idiosyncratic_bounds[j]))
                for i in range(3) for j in range(i + 1, 3)
            ) * 1.001
            for repetition in range(repetitions):
                rng = np.random.default_rng(seeds[cursor]); cursor += 1
                sizes = rng.integers(1, 5, n_patients)
                groups = np.repeat(np.arange(n_patients), sizes)
                signal = rng.uniform(-1.0, 1.0, groups.size)
                noise = rng.uniform(-noise_bound, noise_bound, groups.size)
                patient_common = rng.uniform(-common_bound, common_bound, n_patients)
                common = patient_common[groups]
                errors = np.vstack([
                    common + rng.uniform(-width, width, groups.size)
                    for width in idiosyncratic_bounds
                ])
                target = signal + noise
                predictions = signal[None, :] + errors
                kwargs = dict(
                    y=target, predictions=predictions, groups=groups,
                    lambda_max=lambda_cap, absolute_error_bound=error_cap,
                    prediction_difference_bound=difference_cap,
                )
                eb = empirical_bernstein_patient_pairwise_bound(**kwargs)
                hf = empirical_bernstein_patient_pairwise_bound(
                    **kwargs, concentration="hoeffding"
                )
                catoni = catoni_cluster_pairwise_bound(
                    target, predictions, groups, lambda_cap,
                    risk_variance_cap, disagreement_variance_cap,
                )
                legacy = bregman_bound(target, predictions, "mse")
                active = next(
                    item for item in eb["pairs"]
                    if tuple(item["pair"]) == tuple(eb["active_pair"])
                )
                rows.append({
                    "n_patients": n_patients, "common_bound": common_bound,
                    "r_star": r_star, "patient_eb_sharp": eb["bound"],
                    "patient_eb_conservative": active["conservative_bound"],
                    "patient_hoeffding": hf["bound"], "catoni_oracle": catoni["bound"],
                    "legacy": legacy["bound"],
                })

    def summarize(selected: list[dict]) -> dict:
        result = {}
        for method in (
            "patient_eb_sharp", "patient_eb_conservative", "patient_hoeffding",
            "catoni_oracle", "legacy",
        ):
            values = np.asarray([row[method] for row in selected])
            valid = values <= r_star + 1e-12
            result[method] = {
                "valid": int(valid.sum()), "total": int(values.size),
                "coverage": float(valid.mean()),
                "wilson_lower_95": _wilson_lower(int(valid.sum()), int(values.size)),
                "median_usable_tightness": float(np.median(np.where(valid, values / r_star, 0.0))),
            }
        sharp = np.asarray([row["patient_eb_sharp"] for row in selected])
        conservative = np.asarray([row["patient_eb_conservative"] for row in selected])
        hoeffding = np.asarray([row["patient_hoeffding"] for row in selected])
        result["sharp_geometry_gain"] = {
            "strict_wins": int(np.sum(sharp > conservative + 1e-12)),
            "median_gain": float(np.median(sharp - conservative)),
        }
        result["empirical_bernstein_gain"] = {
            "strict_wins": int(np.sum(sharp > hoeffding + 1e-12)),
            "median_gain": float(np.median(sharp - hoeffding)),
        }
        return result

    return {
        "status": "exploratory_validation",
        "claim": "bounded iid-patient simulation with variable case counts and known caps",
        "pythia_trained_models": False,
        "seed": seed, "repetitions_per_cell": repetitions,
        "n_runs": len(rows), "known_bayes_risk": r_star,
        "summary": summarize(rows),
        "by_patient_count": {
            str(n): summarize([row for row in rows if row["n_patients"] == n])
            for n in patient_counts
        },
    }
