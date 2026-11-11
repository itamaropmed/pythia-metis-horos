"""Frozen, model-specific risk maps for genuinely unseen cases.

Metis does not use evaluation outcomes when scoring a new case.  A map is built
once from honest development OOF predictions, frozen, and then applied to later
predictions. The numerical score is an estimate of *model error*, not a
per-person Bayes-risk certificate. Pythia cohort evidence remains in its own
certificate artifact and cannot be injected into a Metis case-risk card.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


RISK_LABELS = ("low", "moderate", "high", "very_high")
_VERIFIED_MODEL_ARTIFACTS_TOKEN = object()


class VerifiedModelArtifacts:
    """Opaque identity returned only after matching deployed files to a signed map."""

    __slots__ = ("calibration_manifest_sha256", "model_sha256", "_token", "_locked")

    def __init__(self, manifest_sha256: str, model_sha256: Mapping[str, str], token: object):
        if token is not _VERIFIED_MODEL_ARTIFACTS_TOKEN:
            raise TypeError("use FrozenRiskMap.verify_deployed_model_artifacts()")
        object.__setattr__(self, "calibration_manifest_sha256", manifest_sha256)
        object.__setattr__(self, "model_sha256", MappingProxyType(dict(model_sha256)))
        object.__setattr__(self, "_token", token)
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name, value):
        if getattr(self, "_locked", False):
            raise AttributeError("verified model artifact identity is immutable")
        object.__setattr__(self, name, value)


def _deep_freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _plain_data(value):
    if is_dataclass(value):
        return {item.name: _plain_data(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {key: _plain_data(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_data(item) for item in value]
    return value


def _finite_1d(value: Sequence[float], name: str) -> np.ndarray:
    out = np.asarray(value, dtype=float).reshape(-1)
    if out.size == 0 or not np.isfinite(out).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return out


def _edges(value: np.ndarray, bins: int) -> tuple[float, ...]:
    if bins < 1:
        raise ValueError("number of bins must be positive")
    if bins == 1:
        return ()
    q = np.quantile(value, np.linspace(0, 1, bins + 1)[1:-1])
    return tuple(float(v) for v in np.unique(q))


def _bin(value: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    return np.searchsorted(np.asarray(edges, dtype=float), value, side="right")


def anonymize_id(value: object, salt: str) -> str:
    """Return a stable project-local pseudonym; the salt must not be committed."""
    if not salt:
        raise ValueError("a nonempty private salt is required")
    return hashlib.sha256(f"{salt}\x1f{value}".encode()).hexdigest()[:16]


@dataclass(frozen=True)
class CellRisk:
    level: str
    key: str
    n_rows: int
    n_entities: int
    expected_absolute_error: float
    error_over_tolerance_probability: float
    mae_upper: float
    probability_upper: float


@dataclass(frozen=True)
class ModelCaseRisk:
    model: str
    prediction: float
    risk_value: float
    risk_unit: str
    risk_category: str
    probability_over_tolerance: float
    support_level: str
    support_entities: int
    uncertainty_status: str
    mae_upper: float | None
    probability_upper: float | None
    risk_method: str = "hierarchical_oof_cell_mean"


@dataclass(frozen=True)
class CaseRiskAssessment:
    anonymous_case_id: str
    model_risks: tuple[ModelCaseRisk, ...]
    interpretation: str = (
        "risk_value estimates the frozen model's absolute error from development "
        "OOF evidence; it is not an individual Bayes-risk lower bound"
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FrozenRiskMap:
    version: str
    model_names: tuple[str, ...]
    tolerance: float
    error_cap: float
    alpha: float
    min_entities: int
    tier_edges: tuple[float, float, float]
    prediction_edges: Mapping[str, tuple[float, ...]]
    disagreement_edges: tuple[float, ...]
    cells: Mapping[str, Mapping[str, CellRisk]]
    calibration_rows: int
    calibration_entities: int
    provenance: Mapping[str, str]
    calibration_evidence_status: str = "caller_asserted_not_verified"
    calibration_manifest_sha256: str | None = None
    model_sha256: Mapping[str, str] | None = None

    def __post_init__(self):
        for name in ("prediction_edges", "cells", "provenance"):
            object.__setattr__(self, name, _deep_freeze(getattr(self, name)))
        if self.model_sha256 is not None:
            object.__setattr__(self, "model_sha256", _deep_freeze(self.model_sha256))

    def to_dict(self) -> dict:
        return _plain_data(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    def verify_deployed_model_artifacts(
        self, artifact_paths: Mapping[str, str | Path],
    ) -> VerifiedModelArtifacts:
        """Hash actual model files and bind them to this signed calibration map."""
        if (self.calibration_evidence_status != "signature_verified_reviewer_attestation"
                or not self.calibration_manifest_sha256 or not self.model_sha256):
            raise ValueError("deployment model binding requires a signature-verified calibration map")
        if set(artifact_paths) != set(self.model_names):
            raise ValueError("deployment model artifact names must match the calibration map")
        verified = {}
        for name in self.model_names:
            path = Path(artifact_paths[name])
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"deployment model artifact is missing or unsafe: {name}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_hash = digest.hexdigest()
            if actual_hash != self.model_sha256.get(name):
                raise ValueError(f"deployment model hash does not match signed calibration: {name}")
            verified[name] = actual_hash
        return VerifiedModelArtifacts(self.calibration_manifest_sha256, verified,
                                      _VERIFIED_MODEL_ARTIFACTS_TOKEN)

    def score(
        self,
        case_id: object,
        predictions: Mapping[str, float],
        *,
        stratum: str | None = None,
        id_salt: str,
        point_risk_overrides: Mapping[str, float] | None = None,
        probability_overrides: Mapping[str, float] | None = None,
        risk_method: str = "hierarchical_oof_cell_mean",
        verified_model_artifacts: VerifiedModelArtifacts | None = None,
    ) -> CaseRiskAssessment:
        if stratum not in (None, "all"):
            raise ValueError(
                "caller-selected strata are not accepted; score with aggregate support "
                "until a trusted case-to-stratum assignment is configured"
            )
        if self.calibration_evidence_status == "signature_verified_reviewer_attestation":
            if (not isinstance(verified_model_artifacts, VerifiedModelArtifacts)
                    or verified_model_artifacts._token is not _VERIFIED_MODEL_ARTIFACTS_TOKEN
                    or verified_model_artifacts.calibration_manifest_sha256 != self.calibration_manifest_sha256
                    or dict(verified_model_artifacts.model_sha256) != dict(self.model_sha256 or {})):
                raise ValueError("signed calibration scoring requires matching verified deployment model artifacts")
        if set(predictions) != set(self.model_names):
            raise ValueError("new-case prediction names do not match the frozen risk map")
        values = np.asarray([predictions[name] for name in self.model_names], float)
        if not np.isfinite(values).all():
            raise ValueError("new-case predictions must be finite")
        disagreement = float(values.max() - values.min())
        dbin = int(_bin(np.array([disagreement]), self.disagreement_edges)[0])
        risks = []
        for name in self.model_names:
            pred = float(predictions[name])
            pbin = int(_bin(np.array([pred]), self.prediction_edges[name])[0])
            candidates = (
                ("prediction", str(pbin)),
                ("global", "all"),
            )
            chosen = None
            for level, key in candidates:
                cell = self.cells[name].get(f"{level}:{key}")
                if cell is not None and cell.n_entities >= self.min_entities:
                    chosen = cell
                    break
            if chosen is None:  # global is always present, but fail closed on malformed maps
                raise ValueError(f"risk map has no supported cell for {name}")
            point_risk = (
                float(point_risk_overrides[name])
                if point_risk_overrides and name in point_risk_overrides
                else chosen.expected_absolute_error
            )
            point_probability = (
                float(probability_overrides[name])
                if probability_overrides and name in probability_overrides
                else chosen.error_over_tolerance_probability
            )
            if not np.isfinite(point_risk) or not 0 <= point_risk <= self.error_cap:
                raise ValueError("point risk forecasts must lie within the declared error support")
            if not np.isfinite(point_probability) or not 0 <= point_probability <= 1:
                raise ValueError("probability overrides must lie in [0, 1]")
            category = RISK_LABELS[int(np.searchsorted(self.tier_edges, point_risk))]
            risks.append(ModelCaseRisk(
                model=name,
                prediction=pred,
                risk_value=point_risk,
                risk_unit="expected absolute error minutes",
                risk_category=category,
                probability_over_tolerance=point_probability,
                support_level=chosen.level,
                support_entities=chosen.n_entities,
                uncertainty_status=(
                    "caller_supplied_model_risk_forecast; not independently checked by score() or a certificate"
                    if ((point_risk_overrides and name in point_risk_overrides)
                        or (probability_overrides and name in probability_overrides)) else
                    ("signed_calibration_and_deployment_model_hashes_match; prediction execution remains caller-controlled"
                     if self.calibration_evidence_status == "signature_verified_reviewer_attestation" else
                     "conditional_on_honest_OOF_and_true_entity_independence; not provenance-verified")
                ),
                # Cell-map endpoints certify the cell-map estimand only. Do not
                # attach them to a separately supplied per-case point forecast.
                mae_upper=(None if point_risk_overrides and name in point_risk_overrides
                           else chosen.mae_upper),
                probability_upper=(None if probability_overrides and name in probability_overrides
                                   else chosen.probability_upper),
                risk_method=risk_method,
            ))
        return CaseRiskAssessment(
            anonymous_case_id=anonymize_id(case_id, id_salt),
            model_risks=tuple(risks),
        )


def _upper_empirical_bernstein(values: np.ndarray, cap: float, alpha: float) -> float:
    """One-sided bounded empirical-Bernstein endpoint for independent entities."""
    values = np.clip(np.asarray(values, float), 0.0, cap)
    n = len(values)
    if n < 2:
        return float(cap)
    variance = float(np.var(values, ddof=1))
    log_term = float(np.log(2.0 / alpha))
    radius = np.sqrt(2.0 * variance * log_term / n) + 7.0 * cap * log_term / (3.0 * (n - 1))
    return float(min(cap, values.mean() + radius))


def fit_case_risk_map(
    y: Sequence[float],
    predictions: Mapping[str, Sequence[float]],
    *,
    strata: Sequence[object] | None = None,
    entity_ids: Sequence[object] | None = None,
    tolerance: float = 30.0,
    error_cap: float = 900.0,
    alpha: float = 0.05,
    prediction_bins: int = 5,
    disagreement_bins: int = 3,
    min_entities: int = 100,
    tier_edges: tuple[float, float, float] = (20.0, 35.0, 60.0),
    version: str = "metis-case-risk-v1",
    provenance: Mapping[str, str] | None = None,
) -> FrozenRiskMap:
    """Build a risk map, assuming caller supplied honest OOF predictions.

    Cell definitions use predictions/strata, never outcomes.  Repeated rows are
    reduced to entity means before uncertainty endpoints are computed.  The
    declared ``error_cap`` must be a justified support bound for absolute
    error; observed violations are refused rather than silently winsorized.
    Thus the reported MAE target is uncapped MAE on the declared support.

    This function can validate array shape/support and entity-ID presence, but
    cannot authenticate that predictions are honest OOF, that the model was
    frozen first, that entity IDs represent true independent clusters, or
    that ``provenance`` is accurate. Endpoints are conditional on these
    caller-supplied assumptions and must not be called verified certificates
    without an external provenance check.
    """
    target = _finite_1d(y, "y")
    pred = {name: _finite_1d(value, name) for name, value in predictions.items()}
    if len(pred) < 2 or any(len(value) != len(target) for value in pred.values()):
        raise ValueError("at least two aligned model predictions are required")
    # A direct or affine copy of calibration outcomes is a common accidental
    # target-encoding failure. This check is only a leakage tripwire: passing
    # it cannot establish honest OOF provenance or exclude nonlinear leakage.
    if np.ptp(target) > 1e-12:
        target_design = np.column_stack([target, np.ones(len(target))])
        for name, values in pred.items():
            if np.ptp(values) <= 1e-12:
                continue  # a legitimate constant predictor is not a target copy
            slope, intercept = np.linalg.lstsq(target_design, values, rcond=None)[0]
            residual = values - (slope * target + intercept)
            scale = max(1.0, float(np.max(np.abs(values))))
            if abs(float(slope)) > 1e-12 and float(np.max(np.abs(residual))) <= 1e-12 * scale:
                raise ValueError(
                    f"prediction {name!r} is an affine copy of calibration targets; "
                    "refusing until OOF provenance is independently verified"
                )
    if tolerance <= 0 or error_cap <= 0 or not 0 < alpha < 1:
        raise ValueError("tolerance/error_cap/alpha are invalid")
    if error_cap < tolerance:
        raise ValueError("error_cap must be at least the over-tolerance threshold")
    observed_max_error = max(
        float(np.max(np.abs(target - value))) for value in pred.values()
    )
    if observed_max_error > error_cap + 1e-12:
        raise ValueError(
            "observed absolute error exceeds error_cap; refusing rather than silently clipping"
        )
    if tuple(sorted(tier_edges)) != tier_edges or tier_edges[0] < 0:
        raise ValueError("tier edges must be nonnegative and increasing")
    n = len(target)
    strata_arr = np.asarray(["all"] * n if strata is None else strata, dtype=str)
    if entity_ids is None:
        raise ValueError(
            "entity_ids are required; row independence must not be assumed implicitly"
        )
    raw_entities = np.asarray(entity_ids).reshape(-1)
    if pd.Series(raw_entities).isna().any():
        raise ValueError("entity_ids must be present for every calibration row")
    entities = pd.Series(raw_entities).astype(str).str.strip().to_numpy()
    if np.any(entities == ""):
        raise ValueError("entity_ids must be nonempty for every calibration row")
    if len(strata_arr) != n or len(entities) != n:
        raise ValueError("strata/entity_ids must align with outcomes")

    names = tuple(pred)
    matrix = np.column_stack([pred[name] for name in names])
    disagreement = matrix.max(axis=1) - matrix.min(axis=1)
    d_edges = _edges(disagreement, disagreement_bins)
    p_edges = {name: _edges(pred[name], prediction_bins) for name in names}
    d_bin = _bin(disagreement, d_edges)
    cells: dict[str, dict[str, CellRisk]] = {}

    def summarize(name: str, level: str, keys: np.ndarray, wanted: str) -> CellRisk:
        mask = keys == wanted
        ae = np.abs(target[mask] - pred[name][mask])
        miss = (ae > tolerance).astype(float)
        grouped = pd.DataFrame({"entity": entities[mask], "ae": ae, "miss": miss}).groupby(
            "entity", sort=False, observed=True
        ).agg(ae=("ae", "mean"), miss=("miss", "mean"))
        return CellRisk(
            level=level, key=wanted, n_rows=int(mask.sum()), n_entities=int(len(grouped)),
            expected_absolute_error=float(grouped.ae.mean()),
            error_over_tolerance_probability=float(grouped.miss.mean()),
            mae_upper=_upper_empirical_bernstein(grouped.ae.to_numpy(), error_cap, alpha),
            probability_upper=_upper_empirical_bernstein(grouped.miss.to_numpy(), 1.0, alpha),
        )

    for name in names:
        p_bin = _bin(pred[name], p_edges[name])
        levels = {
            "stratum_prediction_disagreement": np.asarray([
                f"{s}|{p}|{d}" for s, p, d in zip(strata_arr, p_bin, d_bin)
            ]),
            "stratum_prediction": np.asarray([f"{s}|{p}" for s, p in zip(strata_arr, p_bin)]),
            "prediction": p_bin.astype(str),
            "global": np.asarray(["all"] * n),
        }
        model_cells: dict[str, CellRisk] = {}
        for level, keys in levels.items():
            for key in np.unique(keys):
                cell = summarize(name, level, keys, str(key))
                model_cells[f"{level}:{key}"] = cell
        cells[name] = model_cells

    return FrozenRiskMap(
        version=version, model_names=names, tolerance=float(tolerance),
        error_cap=float(error_cap), alpha=float(alpha), min_entities=int(min_entities),
        tier_edges=tuple(map(float, tier_edges)), prediction_edges=p_edges,
        disagreement_edges=d_edges, cells=cells, calibration_rows=n,
        calibration_entities=int(pd.Series(entities).nunique()), provenance=dict(provenance or {}),
    )
