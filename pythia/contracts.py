"""Stable, serialisable contracts for the public certification API."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np


class ClaimTier(str, Enum):
    PROVED = "proved"
    CONDITIONAL = "conditional"
    CALIBRATED = "calibrated"
    EXPLORATORY = "exploratory"
    REFUSED = "refused"


@dataclass(frozen=True)
class TaskCard:
    target: str
    loss: str
    group: str | None = None
    time: str | None = None
    as_of_rule: str | None = None
    operational_unit: str = "loss units"
    quantile: float | None = None
    window: float | None = None


@dataclass(frozen=True)
class Representation:
    name: str
    version: str = "1"
    feature_names: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    source_hash: str | None = None


@dataclass(frozen=True)
class FoldMatrix:
    y: np.ndarray
    P_oof: np.ndarray
    names: tuple[str, ...]
    families: tuple[str, ...]
    fold_id: np.ndarray
    groups: np.ndarray | None = None
    row_ids: np.ndarray | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        y = np.asarray(self.y).ravel()
        p = np.atleast_2d(np.asarray(self.P_oof))
        if p.shape[1] != y.size:
            raise ValueError(f"P_oof has {p.shape[1]} rows but y has {y.size}")
        if p.shape[0] != len(self.names) or p.shape[0] != len(self.families):
            raise ValueError("predictor names/families do not match P_oof")
        if np.asarray(self.fold_id).size != y.size:
            raise ValueError("fold_id does not match y")
        if self.groups is not None and np.asarray(self.groups).size != y.size:
            raise ValueError("groups does not match y")
        if not np.isfinite(y).all() or not np.isfinite(p).all():
            raise ValueError("certification inputs must be finite")


@dataclass(frozen=True)
class PairMatrix:
    """Outcome-blind replicate pairs, optionally grouped into audit blocks."""

    y_left: np.ndarray
    y_right: np.ndarray
    distances: np.ndarray
    left_entities: np.ndarray
    right_entities: np.ndarray
    pair_blocks: np.ndarray
    meta: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        arrays = [
            np.asarray(self.y_left).ravel(), np.asarray(self.y_right).ravel(),
            np.asarray(self.distances).ravel(), np.asarray(self.left_entities).ravel(),
            np.asarray(self.right_entities).ravel(), np.asarray(self.pair_blocks).ravel(),
        ]
        if arrays[0].size < 2 or any(value.size != arrays[0].size for value in arrays[1:]):
            raise ValueError("pair-matrix arrays must align and contain at least two pairs")
        if not all(np.isfinite(value.astype(float)).all() for value in arrays[:3]):
            raise ValueError("pair-matrix outcomes and distances must be finite")


@dataclass(frozen=True)
class GuardSpec:
    version: str
    kind: str
    bound_grid: np.ndarray
    estimator_grid: np.ndarray
    rho_domain: tuple[float, float]
    tau_domain: tuple[float, float]
    calibration_population: str
    source_hash: str
    seed_registry: str | None = None


@dataclass(frozen=True)
class DependenceGuardSpec:
    """Versioned nuisance assumptions for a real-world dependence certificate."""

    version: str
    lambda_max: float
    risk_variance_bound: float
    disagreement_variance_bound: float
    calibration_population: str
    source_hash: str
    evaluation_contract: str = "independent_patients"
    alpha: float = 0.05

    def validate(self) -> None:
        values = np.asarray([
            self.lambda_max, self.risk_variance_bound,
            self.disagreement_variance_bound, self.alpha,
        ], float)
        if not np.isfinite(values).all():
            raise ValueError("dependence guard values must be finite")
        if not 0 <= self.lambda_max < 1:
            raise ValueError("dependence lambda_max must be in [0, 1)")
        if self.risk_variance_bound <= 0 or self.disagreement_variance_bound <= 0:
            raise ValueError("dependence variance bounds must be positive")
        if not 0 < self.alpha < 1:
            raise ValueError("dependence alpha must be in (0, 1)")
        if self.evaluation_contract != "independent_patients":
            raise ValueError("only the independent_patients evaluation contract is proved")
        if not self.version or not self.calibration_population or not self.source_hash:
            raise ValueError("dependence guard provenance fields must be nonempty")


@dataclass(frozen=True)
class RangeDependenceGuardSpec:
    """Versioned bounded-range assumptions for variance-adaptive concentration."""

    version: str
    lambda_max: float
    absolute_error_bound: float
    prediction_difference_bound: float
    calibration_population: str
    source_hash: str
    evaluation_contract: str = "independent_patients"
    alpha: float = 0.05

    def validate(self) -> None:
        values = np.asarray([
            self.lambda_max, self.absolute_error_bound,
            self.prediction_difference_bound, self.alpha,
        ], float)
        if not np.isfinite(values).all():
            raise ValueError("range dependence guard values must be finite")
        if not 0 <= self.lambda_max < 1:
            raise ValueError("dependence lambda_max must be in [0, 1)")
        if self.absolute_error_bound <= 0 or self.prediction_difference_bound <= 0:
            raise ValueError("dependence range bounds must be positive")
        if not 0 < self.alpha < 1:
            raise ValueError("dependence alpha must be in (0, 1)")
        if self.evaluation_contract != "independent_patients":
            raise ValueError("only the independent_patients evaluation contract is proved")
        if not self.version or not self.calibration_population or not self.source_hash:
            raise ValueError("range dependence guard provenance fields must be nonempty")


@dataclass(frozen=True)
class ReplicateGuardSpec:
    """Versioned smoothness and range/clipping assumptions for paired replicates."""

    version: str
    lipschitz_bound: float
    distance_bound: float
    outcome_difference_bound: float | None
    calibration_population: str
    source_hash: str
    alpha: float = 0.05
    concentration: str = "empirical_bernstein"
    outcome_term_clip_grid: tuple[float, ...] | None = None

    def validate(self) -> None:
        values = np.asarray([
            self.lipschitz_bound, self.distance_bound, self.alpha,
        ], float)
        if not np.isfinite(values).all() or self.lipschitz_bound < 0:
            raise ValueError("replicate guard values must be finite and Lipschitz nonnegative")
        if self.distance_bound <= 0:
            raise ValueError("replicate distance bound must be positive")
        if self.outcome_difference_bound is not None and (
            not np.isfinite(self.outcome_difference_bound)
            or self.outcome_difference_bound <= 0
        ):
            raise ValueError("replicate outcome cap must be finite and positive")
        if not 0 < self.alpha < 1:
            raise ValueError("replicate alpha must be in (0, 1)")
        if self.concentration not in {
            "empirical_bernstein", "hoeffding", "conditional_hoeffding",
            "conditional_empirical_bernstein", "conditional_bentkus",
            "conditional_block_hoeffding",
            "conditional_block_empirical_bernstein", "auto",
        }:
            raise ValueError("unsupported replicate concentration method")
        if self.outcome_term_clip_grid is not None:
            clips = np.asarray(self.outcome_term_clip_grid, float).ravel()
            if (clips.size == 0 or not np.isfinite(clips).all()
                    or np.any(clips <= 0) or np.any(np.diff(clips) <= 0)):
                raise ValueError("replicate clip grid must be positive, finite, and strictly increasing")
        if not self.version or not self.calibration_population or not self.source_hash:
            raise ValueError("replicate guard provenance fields must be nonempty")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass(frozen=True)
class Certificate:
    status: str
    claim_tier: ClaimTier
    task: TaskCard
    representation: Representation
    bound: float | None
    estimator: float | None
    achieved_loss: float | None
    gamma: float | None
    assumptions: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    rungs: Mapping[str, Any] = field(default_factory=dict)
    intervals: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    refusal: Mapping[str, Any] | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))
