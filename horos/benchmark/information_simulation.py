"""Known-risk tabular/text-channel simulations outside the Pythia boundary.

No estimator is fitted here. Analytic Bayes acts and deliberately controlled
approximation errors create frozen prediction matrices that are then handed to
Pythia exactly as an external model pipeline would hand over OOF predictions.
"""
from __future__ import annotations

import hashlib

import numpy as np

from pythia.core.bregman import bregman_bound
from pythia.core.dependence import dependence_profile
from pythia.information_order import RiskInterval, close_information_order, information_value_interval


def _sha256_lines(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _predictor_pool(
    bayes_act: np.ndarray,
    rng: np.random.Generator,
    *,
    common_scale: float,
) -> tuple[np.ndarray, float]:
    common = rng.normal(scale=common_scale, size=bayes_act.size)
    errors = np.vstack([
        common + rng.normal(scale=scale, size=bayes_act.size)
        for scale in (0.15, 0.25, 0.40)
    ])
    correlations = np.corrcoef(errors)
    cap = max(correlations[i, j] for i in range(3) for j in range(i + 1, 3))
    return bayes_act[None, :] + errors, float(np.clip(max(cap, 0.0) + 1e-8, 0.0, 1.0 - 1e-8))


def run_information_simulation(
    *, seed: int = 20260911, n: int = 40_000,
) -> dict:
    """Exercise a tabular -> noisy-note -> intact-note information chain.

    ``Y = X1 + beta*X2 + epsilon``. The tabular representation observes X1.
    A note contains X2 unless a randomized erasure channel masks it. An intact
    note always contains X2. For squared loss, all three Bayes risks are known:

      tabular: sigma^2 + beta^2
      erased note: sigma^2 + erasure_probability*beta^2
      intact note: sigma^2
    """
    rows: list[dict] = []
    settings = [
        (0.5, 0.3, 0.25, 0.0),
        (0.5, 0.3, 0.50, 0.25),
        (1.0, 0.7, 0.25, 0.50),
        (1.0, 0.7, 0.75, 0.80),
        (2.0, 1.0, 0.50, 1.20),
    ]
    for design_index, (beta, sigma, erase_probability, common_scale) in enumerate(settings):
        rng = np.random.default_rng(seed + design_index)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        epsilon = rng.normal(scale=sigma, size=n)
        erased = rng.random(n) < erase_probability
        notes = [
            "modifier [MASKED]" if missing else f"modifier {value:+.8f}"
            for value, missing in zip(x2, erased)
        ]
        target = x1 + beta * x2 + epsilon
        acts = {
            "intact_note": x1 + beta * x2,
            "erased_note": x1 + beta * np.where(erased, 0.0, x2),
            "tabular": x1,
        }
        population_risks = {
            "intact_note": sigma ** 2,
            "erased_note": sigma ** 2 + erase_probability * beta ** 2,
            "tabular": sigma ** 2 + beta ** 2,
        }
        intervals = {
            name: RiskInterval(risk, risk, risk, "proved", simultaneous=True, source="analytic Gaussian design")
            for name, risk in population_risks.items()
        }
        closure = close_information_order(
            intervals,
            [("intact_note", "erased_note"), ("erased_note", "tabular")],
        )
        information = information_value_interval(
            closure, richer="intact_note", poorer="tabular"
        )
        for name, act in acts.items():
            predictions, cap = _predictor_pool(act, rng, common_scale=common_scale)
            raw = bregman_bound(target, predictions, "mse")
            profile = dependence_profile(target, predictions, np.asarray([cap]))
            adjusted = float(profile["curve"][0]["bound"])
            oracle_population = float(population_risks[name])
            oracle_empirical = float(np.mean((target - act) ** 2))
            rows.append({
                "design": design_index,
                "representation": name,
                "beta": beta,
                "sigma": sigma,
                "erasure_probability": erase_probability,
                "common_scale": common_scale,
                "known_bayes_risk": oracle_population,
                "empirical_oracle_risk": oracle_empirical,
                "raw_orthogonality_bound": float(raw["bound"]),
                "dependence_bound": adjusted,
                "lambda_cap": cap,
                "raw_population_valid": bool(raw["bound"] <= oracle_population),
                "dependence_population_valid": bool(adjusted <= oracle_population),
                "dependence_empirical_valid": bool(adjusted <= oracle_empirical + 1e-8),
                "dependence_tightness": adjusted / oracle_population,
                "notes_hash": _sha256_lines(notes),
            })
    raw_valid = sum(row["raw_population_valid"] for row in rows)
    dep_valid = sum(row["dependence_population_valid"] for row in rows)
    dep_empirical_valid = sum(row["dependence_empirical_valid"] for row in rows)
    return {
        "status": "exploratory_validation",
        "claim": "known-risk simulation; dependence profile remains a theorem candidate",
        "pythia_trained_models": False,
        "seed": seed,
        "n_per_design": n,
        "n_designs": len(settings),
        "n_representation_cases": len(rows),
        "information_chain": ["intact_note", "erased_note", "tabular"],
        "analytic_information_value": float(information["estimate"]),
        "summary": {
            "raw_orthogonality_population_valid": raw_valid,
            "dependence_population_valid": dep_valid,
            "dependence_empirical_valid": dep_empirical_valid,
            "median_dependence_tightness": float(np.median([row["dependence_tightness"] for row in rows])),
        },
        "rows": rows,
    }
