"""Frozen, model-specific risk maps for genuinely unseen cases.

Metis does not use evaluation outcomes when scoring a new case.  A map is built
once from honest development OOF predictions, frozen, and then applied to later
predictions.  The numerical score is an estimate of *model error*, not a
per-person Bayes-risk certificate; a Pythia cohort floor may be attached as a
separate field without changing that interpretation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


RISK_LABELS = ("low", "moderate", "high", "very_high")


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
    mae_upper: float
    probability_upper: float
    risk_method: str = "hierarchical_oof_cell_mean"
    information_set_lower_bound: float | None = None
    lower_bound_claim_tier: str | None = None


@dataclass(frozen=True)
class CaseRiskAssessment:
    anonymous_case_id: str
    model_risks: tuple[ModelCaseRisk, ...]
    cohort_lower_bound: float | None = None
    cohort_claim_tier: str | None = None
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

    def to_dict(self) -> dict:
        raw = asdict(self)
        return raw

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    def score(
        self,
        case_id: object,
        predictions: Mapping[str, float],
        *,
        stratum: str = "all",
        id_salt: str,
        cohort_lower_bound: float | None = None,
        cohort_claim_tier: str | None = None,
        information_set_floors: Mapping[str, tuple[float, str]] | None = None,
        point_risk_overrides: Mapping[str, float] | None = None,
        probability_overrides: Mapping[str, float] | None = None,
        risk_method: str = "hierarchical_oof_cell_mean",
    ) -> CaseRiskAssessment:
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
                ("stratum_prediction_disagreement", f"{stratum}|{pbin}|{dbin}"),
                ("stratum_prediction", f"{stratum}|{pbin}"),
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
            if not np.isfinite(point_risk) or point_risk < 0:
                raise ValueError("point risk overrides must be finite and nonnegative")
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
                    "point_forecast_calibrated_on_development_oof; cell_upper_endpoints_are_support_diagnostics"
                    if point_risk_overrides else
                    "conditional_on_frozen_map_and_entity_exchangeability"
                ),
                mae_upper=chosen.mae_upper,
                probability_upper=chosen.probability_upper,
                risk_method=risk_method,
                information_set_lower_bound=(
                    float(information_set_floors[name][0])
                    if information_set_floors and name in information_set_floors else None
                ),
                lower_bound_claim_tier=(
                    str(information_set_floors[name][1])
                    if information_set_floors and name in information_set_floors else None
                ),
            ))
        return CaseRiskAssessment(
            anonymous_case_id=anonymize_id(case_id, id_salt),
            model_risks=tuple(risks),
            cohort_lower_bound=cohort_lower_bound,
            cohort_claim_tier=cohort_claim_tier,
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
    """Build a risk map from honest OOF predictions only.

    Cell definitions use predictions/strata, never outcomes.  Repeated rows are
    reduced to entity means before uncertainty endpoints are computed.
    """
    target = _finite_1d(y, "y")
    pred = {name: _finite_1d(value, name) for name, value in predictions.items()}
    if len(pred) < 2 or any(len(value) != len(target) for value in pred.values()):
        raise ValueError("at least two aligned model predictions are required")
    if tolerance <= 0 or error_cap <= 0 or not 0 < alpha < 1:
        raise ValueError("tolerance/error_cap/alpha are invalid")
    if tuple(sorted(tier_edges)) != tier_edges or tier_edges[0] < 0:
        raise ValueError("tier edges must be nonnegative and increasing")
    n = len(target)
    strata_arr = np.asarray(["all"] * n if strata is None else strata, dtype=str)
    entities = np.asarray(np.arange(n) if entity_ids is None else entity_ids, dtype=str)
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
        ae = np.minimum(np.abs(target[mask] - pred[name][mask]), error_cap)
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
