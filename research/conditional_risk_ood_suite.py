"""Ten-family known-truth red team for case-level conditional-risk estimation.

Predictors are frozen analytic functions; only the second-order risk forecasters
are fitted. Development/test are split by patient and time, and the test outcomes
are read only after every method has emitted predictions. Noise laws are finite
normal mixtures, so conditional MAE and W30 probabilities are computed exactly.

This is an estimator benchmark, not a validity proof for Metis' bounded-risk
endpoints. In particular, Gaussian-mixture outcomes have unbounded support.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit
from scipy.stats import norm, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from horos.models.case_risk import fit_model_risk_forecaster


DESIGNS = (
    "homoskedastic_normal", "nonlinear_heteroskedastic", "interaction_bias",
    "contaminated_heavy_tail", "skewed_noise", "rare_high_risk",
    "conditional_mixture", "note_concept_risk", "patient_shared_noise",
    "covariate_shift",
)


def _normal_abs(mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    z = np.abs(mu) / np.maximum(sd, 1e-12)
    return sd * np.sqrt(2 / np.pi) * np.exp(-0.5 * z**2) + np.abs(mu) * (
        1 - 2 * norm.cdf(-z)
    )


def _mixture_truth(bias, scale, weights, means, sds, threshold=30.):
    mae = np.zeros_like(scale, dtype=float)
    miss = np.zeros_like(scale, dtype=float)
    for weight, mean, sd in zip(weights, means, sds):
        residual_mean = scale * mean - bias
        residual_sd = scale * sd
        mae += weight * _normal_abs(residual_mean, residual_sd)
        miss += weight * (
            norm.cdf((-threshold - residual_mean) / residual_sd)
            + norm.sf((threshold - residual_mean) / residual_sd)
        )
    return mae, miss


def generate(design: str, seed: int, *, patients: int = 1400,
             noise_shift: float = 1.) -> dict:
    if design not in DESIGNS or patients < 300 or noise_shift <= 0:
        raise ValueError("unknown design, too few patients, or invalid noise_shift")
    rng = np.random.default_rng(seed)
    visits = 2
    entity = np.repeat(np.arange(patients), visits)
    n = len(entity)
    patient_x = rng.normal(size=(patients, 5))
    x = patient_x[entity] + rng.normal(scale=.15, size=(n, 5))
    # Note-like concepts are observed before the target and depend only on table
    # values plus independent extraction noise. They are numeric encodings of
    # synthetic phrases, not outcome-trained features.
    note_p = expit(np.column_stack([
        1.2 * x[:, 0] - .4 * x[:, 2], -x[:, 1], .7 * x[:, 3] + .3 * x[:, 0],
        x[:, 4] - .25 * x[:, 0],
    ]))
    notes = (rng.random(note_p.shape) < note_p).astype(float)
    features = np.column_stack([x, notes])

    mu = 100 + 10 * x[:, 0] + 6 * x[:, 1] * x[:, 2]
    scale = np.full(n, 10.)
    weights, means, sds = np.array([1.]), np.array([0.]), np.array([1.])
    if design == "nonlinear_heteroskedastic":
        scale = 4 + 20 * expit(1.3 * x[:, 0] - .8 * x[:, 1] + .3 * x[:, 2] ** 2)
    elif design == "interaction_bias":
        scale = 5 + 5 * np.abs(x[:, 0] * x[:, 1])
    elif design == "contaminated_heavy_tail":
        weights, means, sds = np.array([.96, .04]), np.array([0., 0.]), np.array([1., 6.])
        scale = 8 + 5 * expit(x[:, 0])
    elif design == "skewed_noise":
        # The mixture mean is zero: .8*(-.5) + .2*(2) = 0.
        weights, means, sds = np.array([.8, .2]), np.array([-.5, 2.]), np.array([.65, 1.15])
        scale = 7 + 8 * expit(x[:, 1])
    elif design == "rare_high_risk":
        scale = 5 + 32 * (x[:, 0] > 2.0)
    elif design == "conditional_mixture":
        weights = np.column_stack([expit(1.5 * x[:, 2]), expit(-1.5 * x[:, 2])])
        means, sds = np.array([-.8, .8]), np.array([.65, 1.4])
        scale = 8 + 7 * (x[:, 1] > 0)
    elif design == "note_concept_risk":
        scale = 4 + 22 * notes[:, 0] + 10 * (notes[:, 1] * notes[:, 2])
    elif design == "patient_shared_noise":
        shared_sd = 5 + 6 * np.abs(x[:, 0])
        row_sd = 5 + 4 * expit(x[:, 1])
        scale = np.sqrt(shared_sd**2 + row_sd**2)
    elif design == "covariate_shift":
        # Applied below only to the late test block; the conditional law stays fixed.
        pass

    # Frozen predictor 1 is the conditional mean. Predictor 2 has a deliberately
    # nonlinear subgroup bias, making its conditional error risk distinct.
    pred_a = mu.copy()
    pred_b = mu + 9 * np.tanh(x[:, 0] * x[:, 1]) + 4 * notes[:, 0]
    order = np.arange(patients)  # generated in calendar order
    split = int(.70 * patients)
    train_patients, test_patients = order[:split], order[split:]
    train = np.flatnonzero(np.isin(entity, train_patients))
    test = np.flatnonzero(np.isin(entity, test_patients))
    times = np.repeat(np.arange(patients), visits)
    if design == "covariate_shift":
        x[test, 0] += 1.25
        # Recompute deterministic pre-outcome concepts on shifted table values.
        note_p_test = expit(np.column_stack([
            1.2 * x[test, 0] - .4 * x[test, 2], -x[test, 1],
            .7 * x[test, 3] + .3 * x[test, 0], x[test, 4] - .25 * x[test, 0],
        ]))
        notes[test] = (rng.random(note_p_test.shape) < note_p_test).astype(float)
        features[test] = np.column_stack([x[test], notes[test]])
        mu[test] = 100 + 10 * x[test, 0] + 6 * x[test, 1] * x[test, 2]
        pred_a[test] = mu[test]
        pred_b[test] = mu[test] + 9 * np.tanh(x[test, 0] * x[test, 1]) + 4 * notes[test, 0]
    scale = np.asarray(scale, dtype=float)
    scale[test] *= noise_shift
    if np.ndim(weights) == 1:
        weights = np.broadcast_to(weights, (n, len(weights)))
    else:
        weights = weights.copy()
    component = np.array([
        rng.choice(len(means), p=row / row.sum()) for row in weights
    ])
    if design == "patient_shared_noise":
        # A latent patient-level shock is shared over both visits, with a
        # row-level shock added independently. Conditional on row features,
        # their sum is normal with the exact `scale` used for the oracle risk.
        shared_sd = 5 + 6 * np.abs(x[:, 0])
        row_sd = 5 + 4 * expit(x[:, 1])
        shared_sd[test] *= noise_shift
        row_sd[test] *= noise_shift
        patient_shock = rng.normal(size=patients)
        noise = shared_sd * patient_shock[entity] + row_sd * rng.normal(size=n)
    else:
        noise = scale * (means[component] + sds[component] * rng.normal(size=n))
    y = mu + noise
    truths, miss_truths, realized = {}, {}, {}
    for name, prediction in (("frozen_model_a", pred_a), ("frozen_model_b", pred_b)):
        bias = prediction - mu
        mae, miss = _mixture_truth(bias, scale, weights.T, means, sds)
        truths[name], miss_truths[name] = mae, miss
        realized[name] = np.abs(y - prediction)
    return {
        "entity": entity, "times": times, "x": x, "notes": notes,
        "features": features, "mu": mu, "scale": scale, "y": y,
        "predictions": {"frozen_model_a": pred_a, "frozen_model_b": pred_b},
        "realized_error": realized, "risk_truth": truths, "miss_truth": miss_truths,
        "train": train, "test": test, "train_patients": train_patients,
        "test_patients": test_patients,
    }


def _fit_direct(x_train, losses, x_test, seed, *, loss="squared_error"):
    model = HistGradientBoostingRegressor(
        loss=loss, max_iter=160, learning_rate=.06,
        max_leaf_nodes=15, min_samples_leaf=35, l2_regularization=3.,
        random_state=seed,
    )
    model.fit(x_train, losses)
    return np.maximum(model.predict(x_test), 0.)


def _metrics(pred, truth, observed, miss_prob, miss_truth, actual_miss):
    rho = (spearmanr(pred, truth).statistic
           if np.ptp(pred) > 1e-12 and np.ptp(truth) > 1e-12 else 0.)
    return {
        "conditional_mae_rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "conditional_mae_bias": float(np.mean(pred - truth)),
        "risk_truth_spearman": float(0. if not np.isfinite(rho) else rho),
        "w30_probability_rmse": float(np.sqrt(np.mean((miss_prob - miss_truth) ** 2))),
        "w30_brier": float(np.mean((miss_prob - actual_miss) ** 2)),
        "realized_mae_top_decile": float(observed[pred >= np.quantile(pred, .9)].mean()),
    }


def run_one(design: str, seed: int, *, patients: int = 1400,
            noise_shift: float = 1.) -> dict:
    data = generate(design, seed, patients=patients, noise_shift=noise_shift)
    entity, times = data["entity"], data["times"]
    tr, te = data["train"], data["test"]
    if np.intersect1d(entity[tr], entity[te]).size:
        raise RuntimeError("patient leakage across development/test")
    if times[tr].max() >= times[te].min():
        raise RuntimeError("temporal leakage across development/test")
    rows = {}
    for model_i, name in enumerate(data["predictions"]):
        errors = data["realized_error"][name]
        miss_train = (errors[tr] > 30.).astype(float)
        global_risk = np.full(len(te), errors[tr].mean())
        global_miss = np.full(len(te), miss_train.mean())
        direct = _fit_direct(data["features"][tr], errors[tr], data["features"][te], seed + model_i)
        direct_miss = np.clip(
            _fit_direct(data["features"][tr], miss_train, data["features"][te], seed + model_i + 100),
            0., 1.,
        )
        horos = fit_model_risk_forecaster(
            data["features"][tr], errors[tr], entity[tr], model_name=name,
            tolerance=30., folds=4, seed=seed + model_i * 37, times=times[tr],
        )
        risk_horos, prob_horos = horos.predict(data["features"][te])
        # First access to held-out outcomes: evaluation only, after all forecasts
        # and development-only model/calibration choices have been frozen.
        actual_miss = (errors[te] > 30.).astype(float)
        truth, miss_truth = data["risk_truth"][name][te], data["miss_truth"][name][te]
        observed = errors[te]
        rows[name] = {
            "global_mean": _metrics(global_risk, truth, observed, global_miss,
                                     miss_truth, actual_miss),
            "direct_loss_regression": _metrics(direct, truth, observed, direct_miss,
                                                miss_truth, actual_miss),
            "horos_temporal_crossfit": _metrics(risk_horos, truth, observed, prob_horos,
                                                 miss_truth, actual_miss),
            "oracle_conditional_truth": _metrics(truth, truth, observed, miss_truth,
                                                  miss_truth, actual_miss),
        }
    return {
        "design": design, "seed": seed, "patients": patients,
        "train_patients": len(data["train_patients"]),
        "test_patients": len(data["test_patients"]),
        "patient_overlap": 0, "temporal_order_valid": True,
        "noise_shift": noise_shift, "models": rows,
    }


def run_suite(replicates: int = 2, first_seed: int = 20261025,
              patients: int = 1400, noise_shift: float = 1., *,
              progress: bool = False) -> dict:
    if replicates < 2:
        raise ValueError("at least two replicates are required for variability estimates")
    runs = []
    for design in DESIGNS:
        for i in range(replicates):
            runs.append(run_one(design, first_seed + i, patients=patients,
                                noise_shift=noise_shift))
        if progress:
            print(f"completed {design} ({len(runs)}/{len(DESIGNS) * replicates} runs)",
                  file=sys.stderr, flush=True)
    summary = {}
    methods = ("global_mean", "direct_loss_regression", "horos_temporal_crossfit")
    for design in DESIGNS:
        design_runs = [run for run in runs if run["design"] == design]
        model_summaries = {}
        for model_name in ("frozen_model_a", "frozen_model_b"):
            means = {
                method: float(np.mean([run["models"][model_name][method]["conditional_mae_rmse"]
                                       for run in design_runs])) for method in methods
            }
            winner = min(means, key=means.get)
            model_summaries[model_name] = {
                method: {
                    "mean_conditional_mae_rmse": means[method],
                    "sd_across_replicates": float(np.std([
                        run["models"][model_name][method]["conditional_mae_rmse"]
                        for run in design_runs], ddof=1)),
                    "mean_conditional_mae_bias": float(np.mean([
                        run["models"][model_name][method]["conditional_mae_bias"]
                        for run in design_runs])),
                    "mean_w30_probability_rmse": float(np.mean([
                        run["models"][model_name][method]["w30_probability_rmse"]
                        for run in design_runs])),
                } for method in methods
            }
            model_summaries[model_name]["winner"] = winner
        summary[design] = model_summaries
    wins = {method: 0 for method in methods}
    for design_result in summary.values():
        for model_name in ("frozen_model_a", "frozen_model_b"):
            wins[design_result[model_name]["winner"]] += 1
    return {
        "status": "synthetic_known_truth_development_benchmark_not_clinical_validation",
        "dgp_count": len(DESIGNS), "replicates_per_dgp": replicates,
        "patients_per_replicate": patients, "noise_shift": noise_shift,
        "designs": list(DESIGNS), "strict_wins_across_design_model_cells": wins,
        "summary": summary, "runs": runs,
        "leakage_controls": {
            "patient_disjoint": all(r["patient_overlap"] == 0 for r in runs),
            "strict_temporal_holdout": all(r["temporal_order_valid"] for r in runs),
            "test_outcomes_used_for_fit_or_selection": False,
            "test_outcomes_used_for_metrics_only": True,
        },
        "limitations": [
            "synthetic laws are known normal mixtures and do not establish Mayo transportability",
            "results are risk-estimation comparisons, not coverage validation for an upper bound",
            "the sample of DGP families is finite and selected by the developers",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--first-seed", type=int, default=20261025)
    parser.add_argument("--patients", type=int, default=1400)
    parser.add_argument("--noise-shift", type=float, default=1.)
    parser.add_argument("--output", type=Path,
                        help="optional path for the full synthetic result JSON")
    args = parser.parse_args()
    result = run_suite(args.replicates, args.first_seed, args.patients,
                       args.noise_shift, progress=True)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(payload)
