"""Training-free diagnostics for the predictability of a frozen model's loss.

For loss A and arrival-time information Z, m(Z)=E[A|Z] maximizes correlation
with A among square-integrable Z-measurable forecasts.  The population identity

    Corr(m(Z), A)^2 = Var(E[A|Z]) / Var(A)

separates representation-limited risk prediction from estimator failure.
Pythia does not estimate m here; it audits already frozen, honest forecasts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import spearmanr


@dataclass(frozen=True)
class RiskResolutionDiagnostic:
    n: int
    reference_correlation: float | None
    pearson: float
    spearman: float
    squared_pearson_resolution: float
    reference_squared_resolution: float | None
    resolution_gap_to_reference: float | None
    mean_loss: float
    mean_forecast: float
    mean_calibration_gap: float
    forecast_variance_fraction: float
    status: str
    interpretation: str

    def to_dict(self) -> dict:
        return asdict(self)


def risk_resolution_diagnostic(
    losses: np.ndarray,
    forecasts: np.ndarray,
    *,
    reference_correlation: float | None = None,
) -> RiskResolutionDiagnostic:
    """Audit a frozen risk forecast without fitting or recalibrating it.

    The squared observed Pearson correlation is evidence of resolved loss
    variation.  It is not an objective, validity condition, or upper bound on
    what a richer representation could resolve.  A caller may supply an
    external reference for descriptive reporting; there is deliberately no
    default threshold.  Finite-sample results remain exploratory without a
    sampling contract.
    """
    loss = np.asarray(losses, float).reshape(-1)
    forecast = np.asarray(forecasts, float).reshape(-1)
    if len(loss) < 3 or len(loss) != len(forecast):
        raise ValueError("losses and forecasts must align and contain at least three rows")
    if not np.isfinite(loss).all() or not np.isfinite(forecast).all():
        raise ValueError("losses and forecasts must be finite")
    if np.any(loss < 0):
        raise ValueError("loss values must be nonnegative")
    if reference_correlation is not None and not 0 < reference_correlation < 1:
        raise ValueError("reference correlation must lie in (0, 1)")
    loss_var = float(np.var(loss))
    forecast_var = float(np.var(forecast))
    if loss_var <= 0 or forecast_var <= 0:
        pearson = spearman = 0.
        status = "insufficient_resolution"
    else:
        pearson = float(np.corrcoef(forecast, loss)[0, 1])
        spearman = float(spearmanr(forecast, loss).statistic)
        pearson = pearson if np.isfinite(pearson) else 0.
        spearman = spearman if np.isfinite(spearman) else 0.
        status = "resolved_variation_observed"
    squared = float(max(pearson, 0.) ** 2)
    reference_squared = (
        None if reference_correlation is None else float(reference_correlation ** 2)
    )
    return RiskResolutionDiagnostic(
        n=len(loss), reference_correlation=(
            None if reference_correlation is None else float(reference_correlation)
        ), pearson=pearson,
        spearman=spearman, squared_pearson_resolution=squared,
        reference_squared_resolution=reference_squared,
        resolution_gap_to_reference=(
            None if reference_squared is None else float(max(reference_squared - squared, 0.))
        ),
        mean_loss=float(loss.mean()), mean_forecast=float(forecast.mean()),
        mean_calibration_gap=float(forecast.mean() - loss.mean()),
        forecast_variance_fraction=float(forecast_var / loss_var) if loss_var > 0 else 0.,
        status=status,
        interpretation=(
            "exploratory frozen-forecast diagnostic; Corr(E[loss|Z],loss)^2 equals "
            "the population fraction of loss variance resolved by Z, but this function "
            "does not assert that the supplied forecast equals E[loss|Z]"
        ),
    )
