from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pythia.core.guarded import (
    guarded_mae, guard_value, load_guard, mae_shared_bias_sensitivity, q_cap,
)
from pythia.contracts import GuardSpec


ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived" / "mayo"


def test_qcap_endpoints_and_center():
    assert q_cap(0.0) == pytest.approx(1.0)
    assert q_cap(0.5) == pytest.approx(np.sqrt(2.0))
    assert q_cap(1.0) == pytest.approx(1.0)


def test_shared_bias_curve_subtracts_cap_and_never_tightens():
    result = {
        "global": {"rungs": {
            "unguarded_exploratory": 5.0, "calibrated_2d": 4.0,
            "proved_qcap": 3.0, "worst_case_sqrt2": 2.0,
        }},
        "stratified": None,
    }
    curve = mae_shared_bias_sensitivity(result, np.array([0.0, 1.5, 5.0]))
    assert [row["bound"] for row in curve["curve"]["proved_qcap"]] == [3.0, 1.5, 0.0]
    assert curve["rung_claim_tiers"]["proved_qcap"] == "conditional"


def test_guard_refuses_out_of_domain():
    guard = GuardSpec(
        version="synthetic-domain-test", kind="measured_envelope_2d",
        bound_grid=np.ones((2, 2)), estimator_grid=np.ones((2, 2)),
        rho_domain=(.4, .9), tau_domain=(.1, .9),
        calibration_population="synthetic", source_hash="synthetic",
    )
    with pytest.raises(ValueError):
        guard_value(guard, 0.1, 0.2)


@pytest.mark.skipif(not (DERIVED / "PB.npy").exists(), reason="private Mayo artifacts unavailable")
def test_frozen_mayo_arm_b_reproduces_exactly():
    guard = load_guard(DERIVED)
    df = pd.read_parquet(DERIVED / "df.parquet")
    y = df["case_actual_duration_time"].to_numpy(float)
    p = np.load(DERIVED / "PB.npy")
    best = p[int(np.mean(np.abs(y[None, :] - p), axis=1).argmin())]
    strata = np.asarray(pd.qcut(best, 5, labels=False, duplicates="drop"))
    result = guarded_mae(y, p, guard, strata)
    assert result["global"]["L"] == pytest.approx(32.87001170332562, abs=1e-10)
    assert result["global"]["estimator"] == pytest.approx(31.118642120070444, abs=1e-10)
    assert result["stratified"]["rungs"]["calibrated_2d"] == pytest.approx(
        28.84959960434628, abs=1e-10
    )
    assert result["stratified"]["rungs"]["proved_qcap"] == pytest.approx(
        24.186725940850426, abs=1e-10
    )
