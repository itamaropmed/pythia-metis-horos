"""Generic OOF execution with entity-isolation checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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
    fold_column: str = "fold_id",
    seed: int = 0,
) -> CrossfitResult:
    """Run any fold-local predictor and enforce complete entity-safe OOF output."""
    required = {target_column, group_column, fold_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"crossfit frame lacks required columns: {missing}")
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
        values = np.asarray(
            predictor(frame.loc[fitting].copy(), frame.loc[validation].copy(), seed + fold),
            dtype=float,
        ).ravel()
        if values.size != int(validation.sum()) or not np.isfinite(values).all():
            raise RuntimeError(f"predictor returned invalid output for fold {fold}")
        predictions[validation] = values
        losses.append(float(mean_absolute_error(frame.loc[validation, target_column], values)))
    if not np.isfinite(predictions).all():
        raise RuntimeError("OOF prediction coverage is incomplete")
    return CrossfitResult(predictions, folds, tuple(losses))
