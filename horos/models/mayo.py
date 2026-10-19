"""Reproducible Mayo model candidates, trained outside the Pythia boundary."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from horos.models.crossfit import run_crossfit


TUNED_STRUCTURED_PARAMS = {
    "n_estimators": 1213,
    "max_depth": 8,
    "learning_rate": 0.013721653065680284,
    "subsample": 0.8431891099718941,
    "colsample_bytree": 0.5331833452208407,
    "min_child_weight": 9.083480959214418,
    "reg_alpha": 6.006000053619992e-07,
    "reg_lambda": 0.001809354462444587,
    "gamma": 1.0186857689274291,
    "max_bin": 212,
}

LEAKAGE_COLUMNS = {
    "case_durable_id", "patient_durable_id", "fold_id", "patient_birth_date",
    "case_actual_duration_time", "scheduled_setup_start", "scheduled_in_room",
    "scheduled_out_of_room", "scheduled_cleanup_complete",
    "case_actual_patient_in_room", "case_actual_prep_time",
    "case_actual_procedure_time", "case_actual_wrapup_time",
    "all_procs_not_performed",
}


@dataclass(frozen=True)
class ModelReproductionReport:
    status: str
    model: str
    claim: str
    source_hash: str
    manifest_hash: str
    artifact_hash: str
    n_rows: int
    n_patients: int
    mae: float
    fold_mae: tuple[float, ...]
    seed: int
    prediction_column: str

    def to_dict(self) -> dict:
        return asdict(self)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    scheduled = pd.to_datetime(out["scheduled_in_room"], errors="coerce", utc=True)
    out["scheduled_year"] = scheduled.dt.year
    out["scheduled_month"] = scheduled.dt.month
    out["scheduled_dayofweek"] = scheduled.dt.dayofweek
    out["scheduled_hour"] = scheduled.dt.hour
    out["scheduled_is_weekend"] = scheduled.dt.dayofweek.isin([5, 6]).astype("int64")
    out["scheduled_room_minutes"] = (
        pd.to_datetime(out["scheduled_out_of_room"], errors="coerce", utc=True) - scheduled
    ).dt.total_seconds() / 60.0
    out["scheduled_setup_lead_minutes"] = (
        scheduled - pd.to_datetime(out["scheduled_setup_start"], errors="coerce", utc=True)
    ).dt.total_seconds() / 60.0
    out["scheduled_cleanup_minutes"] = (
        pd.to_datetime(out["scheduled_cleanup_complete"], errors="coerce", utc=True)
        - pd.to_datetime(out["scheduled_out_of_room"], errors="coerce", utc=True)
    ).dt.total_seconds() / 60.0
    out["bmi_missing"] = out["bmi"].isna().astype("int64")
    return out


def _feature_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    drop = set(LEAKAGE_COLUMNS)
    drop.update(column for column in frame if column.startswith("case_actual_") or column.endswith("_date"))
    features = frame[[column for column in frame if column not in drop]].copy()
    target = frame["case_actual_duration_time"].astype(float)
    for column in features:
        if pd.api.types.is_bool_dtype(features[column]):
            features[column] = features[column].astype("int64")
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(features[column])]
    categorical = [column for column in features if column not in numeric]
    return features, target, numeric, categorical


def _encode(fit: pd.Series, y: pd.Series, apply: pd.Series, smoothing: float) -> tuple[np.ndarray, np.ndarray]:
    mean = float(y.mean())
    stats = pd.DataFrame({"key": fit.to_numpy(), "y": y.to_numpy()}).groupby("key")["y"].agg(["mean", "count"])
    values = (stats["count"] * stats["mean"] + smoothing * mean) / (stats["count"] + smoothing)
    return (
        fit.map(values).fillna(mean).to_numpy(float),
        apply.map(values).fillna(mean).to_numpy(float),
    )


def _pipeline(numeric: list[str], categorical: list[str], seed: int):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError("Mayo reproduction requires the optional xgboost dependency") from exc
    preprocess = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=10)),
        ]), categorical),
    ], remainder="drop", sparse_threshold=0.3)
    model = XGBRegressor(
        objective="reg:squarederror", tree_method="hist", random_state=seed,
        n_jobs=-1, eval_metric="mae", **TUNED_STRUCTURED_PARAMS,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def reproduce_tuned_structured_oof(
    source_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    seed: int = 42,
    prediction_column: str = "tuned_structured_prediction",
) -> ModelReproductionReport:
    """Regenerate the known tuned tabular candidate on canonical folds.

    This function does not declare the result to be the 31.2623 incumbent.
    It creates one auditable candidate predictor for the Pythia ensemble.
    """
    source_path, manifest_path, output_path = map(Path, (source_path, manifest_path, output_path))
    source = pd.read_parquet(source_path)
    manifest = pd.read_parquet(manifest_path)
    dev = manifest.loc[manifest["outer_role"].eq("development")].sort_values("development_order")
    frame = dev[["case_durable_id", "patient_durable_id", "fold_id"]].merge(
        source, on=["case_durable_id", "patient_durable_id"], how="left",
        validate="one_to_one", sort=False,
    )
    if frame["case_actual_duration_time"].isna().any():
        raise ValueError("canonical development rows do not align with the model source")
    frame = _safe_features(frame)
    x_all, y_all, numeric, categorical = _feature_frame(frame)
    numeric = list(numeric) + ["proc_target_enc", "surgeon_target_enc"]
    model_frame = frame.copy()
    for column in x_all:
        model_frame[f"_feature_{column}"] = x_all[column]

    def predict_fold(fit_frame: pd.DataFrame, val_frame: pd.DataFrame, fold_seed: int) -> np.ndarray:
        feature_columns = [f"_feature_{column}" for column in x_all]
        x_fit = fit_frame[feature_columns].copy()
        x_val = val_frame[feature_columns].copy()
        x_fit.columns = x_all.columns
        x_val.columns = x_all.columns
        y_fit = fit_frame["case_actual_duration_time"].astype(float)
        x_fit["proc_target_enc"], x_val["proc_target_enc"] = _encode(
            x_fit["primary_procedure_name"], y_fit, x_val["primary_procedure_name"], 20.0
        )
        x_fit["surgeon_target_enc"], x_val["surgeon_target_enc"] = _encode(
            x_fit["surgeon_durable_id"], y_fit, x_val["surgeon_durable_id"], 15.0
        )
        # Historical implementation numbered fold seeds from one.
        model = _pipeline(numeric, categorical, fold_seed + 1)
        model.fit(x_fit, y_fit)
        return model.predict(x_val)

    crossfit = run_crossfit(
        model_frame, predict_fold, target_column="case_actual_duration_time",
        group_column="patient_durable_id", fold_column="fold_id", seed=seed,
    )
    predictions = crossfit.predictions
    low, high = np.quantile(y_all.to_numpy(float), [0.001, 0.999])
    predictions = np.clip(predictions, low, high)
    artifact = frame[[
        "case_durable_id", "patient_durable_id", "fold_id", "case_actual_duration_time",
    ]].copy()
    artifact[prediction_column] = predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact.to_parquet(output_path, index=False)
    return ModelReproductionReport(
        status="candidate_reproduced", model="tuned_structured_xgboost_v2",
        claim="auditable OOF candidate; not the unverified 31.2623 incumbent",
        source_hash=_hash(source_path), manifest_hash=_hash(manifest_path),
        artifact_hash=_hash(output_path), n_rows=int(len(artifact)),
        n_patients=int(artifact["patient_durable_id"].nunique()),
        mae=float(mean_absolute_error(y_all, predictions)),
        fold_mae=crossfit.fold_losses, seed=seed, prediction_column=prediction_column,
    )
