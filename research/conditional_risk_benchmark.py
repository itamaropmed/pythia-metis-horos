"""Known-truth benchmark of Metis case-risk scores vs conditional-loss regression.

Synthetic only. The base predictors are frozen, so the experiment isolates the
second-order task: estimate each predictor's conditional expected absolute
error from outcome-blind features. Patients, not rows, define train/test split.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import norm, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from horos.models.case_risk import fit_model_risk_forecaster
from metis.case_risk import fit_case_risk_map


def _normal_abs_mean(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    z = np.abs(mu) / sigma
    return sigma * np.sqrt(2. / np.pi) * np.exp(-0.5 * z**2) + np.abs(mu) * (
        1. - 2. * norm.cdf(-z)
    )


def _normal_miss_prob(mu: np.ndarray, sigma: np.ndarray, threshold: float) -> np.ndarray:
    return norm.cdf((-threshold - mu) / sigma) + norm.sf((threshold - mu) / sigma)


def _metrics(risk: np.ndarray, truth: np.ndarray, observed: np.ndarray) -> dict:
    rho = 0. if np.ptp(risk) <= 1e-12 else float(spearmanr(risk, observed).statistic)
    decile = np.quantile(risk, .9)
    return {
        "conditional_mae_rmse": float(mean_squared_error(truth, risk) ** .5),
        "conditional_risk_mean_bias": float(np.mean(risk - truth)),
        "risk_vs_realized_error_spearman": 0. if not np.isfinite(rho) else rho,
        "top_decile_realized_mae": float(observed[risk >= decile].mean()),
        "overall_realized_mae": float(observed.mean()),
    }


def run(seed: int = 20261024, test_shift: float = 0.,
        test_noise_multiplier: float = 1.) -> dict:
    rng = np.random.default_rng(seed)
    n_patients, visits = 2500, 2
    n = n_patients * visits
    patient = np.repeat(np.arange(n_patients), visits)
    x_patient = rng.normal(size=(n_patients, 3))
    x = x_patient[patient] + rng.normal(scale=.12, size=(n, 3))
    noise = rng.normal(size=n)

    # Entire patients are assigned to either development or untouched test.
    order = rng.permutation(n_patients)
    train_patients, test_patients = order[:1750], order[1750:]
    train = np.flatnonzero(np.isin(patient, train_patients))
    test = np.flatnonzero(np.isin(patient, test_patients))
    if np.intersect1d(patient[train], patient[test]).size:
        raise RuntimeError("patient leakage in synthetic split")
    if not np.isfinite(test_shift):
        raise ValueError("test_shift must be finite")
    if not np.isfinite(test_noise_multiplier) or test_noise_multiplier <= 0:
        raise ValueError("test_noise_multiplier must be finite and positive")
    # Controlled covariate-shift attack: only test-patient x1 is shifted; the
    # outcome law remains the same conditional function, with known truth.
    x[test, 0] += float(test_shift)
    x1, x2, x3 = x.T
    sigma = 4. + 18. / (1. + np.exp(-(1.25 * x1 + .35 * x2)))
    sigma_for_outcome = sigma.copy()
    sigma_for_outcome[test] *= float(test_noise_multiplier)
    mean_y = 100. + 12. * x2
    prediction_a = mean_y
    prediction_b = mean_y + 12. * x3
    y = mean_y + noise * sigma_for_outcome
    pred_mean_error = {"arm_a": prediction_a - mean_y,
                       "arm_b": prediction_b - mean_y}
    truth_risk = {name: _normal_abs_mean(mu, sigma_for_outcome)
                  for name, mu in pred_mean_error.items()}
    truth_miss = {name: _normal_miss_prob(mu, sigma_for_outcome, 30.)
                  for name, mu in pred_mean_error.items()}
    observed_error = {"arm_a": np.abs(y - prediction_a),
                      "arm_b": np.abs(y - prediction_b)}
    observed_miss = {name: (value > 30.).astype(float)
                     for name, value in observed_error.items()}

    # Strong generic baseline from the conditional-risk paper's regression view:
    # regress observed loss on the available sample features, per frozen model.
    conditional = {}
    conditional_miss = {}
    metis = {}
    learned_miss = {}
    for name in observed_error:
        regressor = HistGradientBoostingRegressor(
            max_iter=180, learning_rate=.06, max_leaf_nodes=15,
            min_samples_leaf=35, l2_regularization=2., random_state=seed,
        )
        regressor.fit(x[train], observed_error[name][train])
        conditional[name] = np.maximum(regressor.predict(x[test]), 0.)
        miss_regressor = HistGradientBoostingRegressor(
            max_iter=180, learning_rate=.06, max_leaf_nodes=15,
            min_samples_leaf=35, l2_regularization=2., random_state=seed + 1,
        )
        miss_regressor.fit(x[train], observed_miss[name][train])
        conditional_miss[name] = np.clip(miss_regressor.predict(x[test]), 0., 1.)

    # Current Horos forecaster: patient-grouped cross-fitted candidate selection,
    # then calibration on development OOF losses; test labels remain untouched.
    learned = {}
    for i, name in enumerate(observed_error):
        forecaster = fit_model_risk_forecaster(
            x[train], observed_error[name][train], patient[train],
            model_name=name, tolerance=30., folds=5, seed=seed + i * 31,
        )
        learned[name], learned_miss[name] = forecaster.predict(x[test])

    # Current Metis map: uses model predictions, disagreement, and a predeclared
    # coarse x1 stratum, but does not receive the full covariate vector.
    predictions = {name: value[train] for name, value in
                   (("arm_a", prediction_a), ("arm_b", prediction_b))}
    strata = np.where(x1 < 0., "low_x1", "high_x1")
    risk_map = fit_case_risk_map(
        y[train], predictions, strata=strata[train], entity_ids=patient[train],
        tolerance=30., error_cap=200., min_entities=50,
        prediction_bins=5, disagreement_bins=3,
    )
    map_risk = {name: np.empty(len(test)) for name in predictions}
    map_probability = {name: np.empty(len(test)) for name in predictions}
    for j, idx in enumerate(test):
        card = risk_map.score(
            f"synthetic-{idx}",
            {"arm_a": float(prediction_a[idx]), "arm_b": float(prediction_b[idx])},
            stratum=strata[idx], id_salt="synthetic-only",
        )
        for risk in card.model_risks:
            map_risk[risk.model][j] = risk.risk_value
            map_probability[risk.model][j] = risk.probability_over_tolerance

    results = {}
    for name in observed_error:
        true = truth_risk[name][test]
        actual = observed_error[name][test]
        global_mean = np.repeat(observed_error[name][train].mean(), len(test))
        rows = {
            "global_mean": global_mean,
            "metis_prediction_disagreement_map": map_risk[name],
            "direct_conditional_loss_regression": conditional[name],
            "horos_crossfit_risk_forecaster": learned[name],
        }
        results[name] = {
            method: _metrics(score, true, actual) for method, score in rows.items()
        }
        results[name]["true_noise_only_oracle"] = _metrics(true, true, actual)
        results[name]["miss_probability_brier"] = {
            "oracle_conditional_probability": float(np.mean((truth_miss[name][test] - observed_miss[name][test])**2)),
            "global_mean": float(np.mean((np.repeat(observed_miss[name][train].mean(), len(test)) - observed_miss[name][test])**2)),
            "metis_map": float(np.mean((map_probability[name] - observed_miss[name][test])**2)),
            "direct_conditional_probability_regression": float(np.mean((conditional_miss[name] - observed_miss[name][test])**2)),
            "horos_crossfit_probability_forecaster": float(np.mean((learned_miss[name] - observed_miss[name][test])**2)),
        }

    return {
        "status": "known_truth_synthetic_benchmark_not_clinical_validation",
        "seed": seed,
        "test_shift_in_x1": float(test_shift),
        "test_noise_multiplier": float(test_noise_multiplier),
        "patients": n_patients,
        "visits_per_patient": visits,
        "train_patients": len(train_patients),
        "test_patients": len(test_patients),
        "patient_overlap": int(np.intersect1d(patient[train], patient[test]).size),
        "dgp": "heteroskedastic Gaussian duration noise; analytic conditional MAE and W30 truth",
        "results": results,
    }


def run_suite(replicates: int = 10, first_seed: int = 20261024,
              test_shift: float = 0., test_noise_multiplier: float = 1.) -> dict:
    if replicates < 2:
        raise ValueError("run_suite requires at least two replicates for reported SDs")
    runs = [run(first_seed + i, test_shift=test_shift,
                test_noise_multiplier=test_noise_multiplier) for i in range(replicates)]
    summary = {}
    for arm in ("arm_a", "arm_b"):
        methods = ("global_mean", "metis_prediction_disagreement_map",
                   "direct_conditional_loss_regression", "horos_crossfit_risk_forecaster")
        rmse = {
            method: np.asarray([
                item["results"][arm][method]["conditional_mae_rmse"] for item in runs
            ]) for method in methods
        }
        winner = np.argmin(np.column_stack([rmse[m] for m in methods]), axis=1)
        brier_methods = (
            "global_mean", "metis_map", "direct_conditional_probability_regression",
            "horos_crossfit_probability_forecaster",
        )
        brier = {
            method: np.asarray([
                item["results"][arm]["miss_probability_brier"][method]
                for item in runs
            ]) for method in brier_methods
        }
        brier_winner = np.argmin(np.column_stack([brier[m] for m in brier_methods]), axis=1)
        summary[arm] = {
            method: {
                "conditional_mae_rmse_mean": float(values.mean()),
                "conditional_mae_rmse_sd_across_replicates": float(values.std(ddof=1)),
                "strict_rmse_wins": int(np.sum(winner == j)),
                "replicates": int(replicates),
                "mean_risk_error_spearman": float(np.mean([
                    item["results"][arm][method]["risk_vs_realized_error_spearman"]
                    for item in runs
                ])),
                "conditional_risk_bias_mean": float(np.mean([
                    item["results"][arm][method]["conditional_risk_mean_bias"]
                    for item in runs
                ])),
            }
            for j, (method, values) in enumerate(rmse.items())
        }
        summary[arm]["w30_brier"] = {
            method: {
                "mean": float(values.mean()),
                "sd_across_replicates": float(values.std(ddof=1)),
                "strict_wins": int(np.sum(brier_winner == j)),
                "replicates": int(replicates),
            }
            for j, (method, values) in enumerate(brier.items())
        }
        summary[arm]["w30_brier"]["oracle_conditional_probability"] = {
            "mean": float(np.mean([
                item["results"][arm]["miss_probability_brier"]["oracle_conditional_probability"]
                for item in runs
            ])),
            "interpretation": "irreducible Brier score when the known conditional probability is used",
        }
    return {
        "status": "known_truth_synthetic_benchmark_not_clinical_validation",
        "replicates": replicates,
        "seeds": [first_seed + i for i in range(replicates)],
        "test_shift_in_x1": float(test_shift),
        "test_noise_multiplier": float(test_noise_multiplier),
        "arms": summary,
        "note": "strict wins compare conditional-risk RMSE within each seed and arm; not a population-level proof",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--first-seed", type=int, default=20261024)
    parser.add_argument("--test-shift", type=float, default=0.)
    parser.add_argument("--test-noise-multiplier", type=float, default=1.)
    args = parser.parse_args()
    print(json.dumps(run_suite(args.replicates, args.first_seed, args.test_shift,
                               args.test_noise_multiplier),
                     indent=2, sort_keys=True))
