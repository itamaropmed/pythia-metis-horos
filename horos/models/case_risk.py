"""Leakage-safe training of second-order, per-model error forecasters.

Duration predictors are already frozen.  The targets here are their honest OOF
errors.  Candidate selection uses patient-grouped cross-fitted forecasts only;
later evaluation outcomes never enter this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold


def _candidates(seed: int) -> dict[str, object]:
    return {
        "hgb_shallow": HistGradientBoostingRegressor(
            loss="squared_error", max_iter=180, learning_rate=.05,
            max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=4.,
            random_state=seed,
        ),
        "hgb_medium": HistGradientBoostingRegressor(
            loss="squared_error", max_iter=220, learning_rate=.04,
            max_leaf_nodes=31, min_samples_leaf=70, l2_regularization=8.,
            random_state=seed,
        ),
        "extra_trees_smooth": ExtraTreesRegressor(
            n_estimators=180, min_samples_leaf=45, max_features=.8,
            n_jobs=-1, random_state=seed,
        ),
    }


@dataclass(frozen=True)
class CandidateRiskResult:
    name: str
    spearman: float
    mean_absolute_calibration_error: float
    mean_bias: float
    brier: float


@dataclass
class FrozenModelRiskForecaster:
    model_name: str
    error_estimator: object
    error_calibrator: IsotonicRegression
    probability_estimator: object
    probability_calibrator: IsotonicRegression
    selected_error_candidate: str
    selected_probability_candidate: str
    candidate_results: tuple[CandidateRiskResult, ...]

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features = np.asarray(features, float)
        raw_error = np.maximum(np.asarray(self.error_estimator.predict(features), float), 0.)
        risk = np.maximum(self.error_calibrator.predict(raw_error), 0.)
        raw_probability = np.clip(
            np.asarray(self.probability_estimator.predict(features), float), 0., 1.
        )
        probability = np.clip(self.probability_calibrator.predict(raw_probability), 0., 1.)
        return risk, probability


def _crossfit(estimator: object, x: np.ndarray, target: np.ndarray,
              groups: np.ndarray, folds: int) -> np.ndarray:
    out = np.empty(len(target), float)
    for fit, val in GroupKFold(folds).split(x, target, groups):
        model = clone(estimator)
        model.fit(x[fit], target[fit])
        out[val] = model.predict(x[val])
    return out


def fit_model_risk_forecaster(
    features: np.ndarray,
    absolute_error: np.ndarray,
    groups: np.ndarray,
    *,
    model_name: str,
    tolerance: float = 30.,
    folds: int = 5,
    seed: int = 0,
) -> FrozenModelRiskForecaster:
    """Select and freeze error/probability forecasters using development only."""
    x = np.asarray(features, float)
    error = np.asarray(absolute_error, float).reshape(-1)
    entity = np.asarray(groups).reshape(-1)
    if x.ndim != 2 or len(x) != len(error) or len(entity) != len(error):
        raise ValueError("risk features, OOF errors, and groups must align")
    if len(error) == 0 or not np.isfinite(x).all() or not np.isfinite(error).all():
        raise ValueError("risk-training inputs must be nonempty and finite")
    if np.any(error < 0) or tolerance <= 0:
        raise ValueError("absolute errors must be nonnegative and tolerance positive")
    if len(np.unique(entity)) < folds:
        raise ValueError("not enough independent groups for requested folds")

    miss = (error > tolerance).astype(float)
    candidate_oof: dict[str, np.ndarray] = {}
    results = []
    for name, estimator in _candidates(seed).items():
        oof = np.maximum(_crossfit(estimator, x, error, entity, folds), 0.)
        candidate_oof[name] = oof
        rho = float(spearmanr(oof, error).statistic)
        if not np.isfinite(rho):
            rho = 0.
        results.append(CandidateRiskResult(
            name=name, spearman=rho,
            mean_absolute_calibration_error=float(np.mean(np.abs(oof - error))),
            mean_bias=float(np.mean(oof - error)), brier=float("nan"),
        ))
    selected_error = max(results, key=lambda value: (value.spearman, -abs(value.mean_bias))).name

    probability_oof: dict[str, np.ndarray] = {}
    probability_results = []
    for name, estimator in _candidates(seed + 1000).items():
        oof = np.clip(_crossfit(estimator, x, miss, entity, folds), 0., 1.)
        probability_oof[name] = oof
        probability_results.append(CandidateRiskResult(
            name=f"probability:{name}", spearman=0.,
            mean_absolute_calibration_error=float(np.mean(np.abs(oof - miss))),
            mean_bias=float(np.mean(oof - miss)),
            brier=float(np.mean((oof - miss) ** 2)),
        ))
    selected_probability_result = min(probability_results, key=lambda value: value.brier)
    selected_probability = selected_probability_result.name.split(":", 1)[1]

    error_raw = candidate_oof[selected_error]
    error_calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.)
    error_calibrator.fit(error_raw, error)
    probability_raw = probability_oof[selected_probability]
    probability_calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0., y_max=1.)
    probability_calibrator.fit(probability_raw, miss)

    candidates = _candidates(seed)
    error_estimator = clone(candidates[selected_error]).fit(x, error)
    probability_candidates = _candidates(seed + 1000)
    probability_estimator = clone(probability_candidates[selected_probability]).fit(x, miss)
    return FrozenModelRiskForecaster(
        model_name=model_name, error_estimator=error_estimator,
        error_calibrator=error_calibrator,
        probability_estimator=probability_estimator,
        probability_calibrator=probability_calibrator,
        selected_error_candidate=selected_error,
        selected_probability_candidate=selected_probability,
        candidate_results=tuple(results + probability_results),
    )


def fit_risk_forecasters(
    features: np.ndarray,
    absolute_errors: Mapping[str, np.ndarray],
    groups: np.ndarray,
    *,
    tolerance: float = 30.,
    folds: int = 5,
    seed: int = 0,
) -> dict[str, FrozenModelRiskForecaster]:
    return {
        name: fit_model_risk_forecaster(
            features, error, groups, model_name=name, tolerance=tolerance,
            folds=folds, seed=seed + position * 17,
        )
        for position, (name, error) in enumerate(absolute_errors.items())
    }
