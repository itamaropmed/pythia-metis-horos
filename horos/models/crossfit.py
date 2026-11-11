"""Generic OOF execution with entity-isolation and validation-blinding checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


FoldPredictor = Callable[[pd.DataFrame, pd.DataFrame, int], np.ndarray]


@dataclass(frozen=True)
class CrossfitResult:
    predictions: np.ndarray
    fold_ids: tuple[int, ...]
    fold_losses: tuple[float, ...]


def run_crossfit(
    frame: pd.DataFrame,
    predictor: FoldPredictor,
    *,
    target_column: str,
    group_column: str,
    feature_columns: Sequence[str],
    fold_column: str = "fold_id",
    seed: int = 0,
) -> CrossfitResult:
    """Run a fold-local predictor without exposing validation labels or IDs.

    Fit rows retain labels/groups/fold metadata for training and grouped
    transformations. Validation rows passed to the predictor contain only
    the explicit feature allowlist; labels and entity/fold identifiers remain
    inside this function and are used only after prediction to score the fold.
    """
    required = {target_column, group_column, fold_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"crossfit frame lacks required columns: {missing}")
    feature_columns = tuple(feature_columns)
    if not feature_columns or len(set(feature_columns)) != len(feature_columns):
        raise ValueError("feature_columns must be a nonempty sequence without duplicates")
    protected = {target_column, group_column, fold_column}
    if protected.intersection(feature_columns):
        raise ValueError("feature_columns cannot include target, entity, or fold metadata")
    missing_features = sorted(set(feature_columns) - set(frame.columns))
    if missing_features:
        raise ValueError(f"crossfit frame lacks declared features: {missing_features}")
    if frame[list(required)].isna().any().any():
        raise ValueError("target, group, and fold columns must be complete")
    folds = tuple(int(value) for value in sorted(frame[fold_column].unique()))
    if len(folds) < 2:
        raise ValueError("cross-fitting requires at least two folds")
    predictions = np.full(len(frame), np.nan)
    losses = []
    groups = frame[group_column].astype(str)
    for fold in folds:
        validation = frame[fold_column].eq(fold).to_numpy()
        fitting = ~validation
        if set(groups.loc[fitting]) & set(groups.loc[validation]):
            raise RuntimeError(f"entity leakage detected in fold {fold}")
        validation_features = frame.loc[validation, list(feature_columns)].copy()
        values = np.asarray(
            predictor(frame.loc[fitting].copy(), validation_features, seed + fold),
            dtype=float,
        ).ravel()
        if values.size != int(validation.sum()) or not np.isfinite(values).all():
            raise RuntimeError(f"predictor returned invalid output for fold {fold}")
        predictions[validation] = values
        losses.append(float(mean_absolute_error(frame.loc[validation, target_column], values)))
    if not np.isfinite(predictions).all():
        raise RuntimeError("OOF prediction coverage is incomplete")
    return CrossfitResult(predictions, folds, tuple(losses))
