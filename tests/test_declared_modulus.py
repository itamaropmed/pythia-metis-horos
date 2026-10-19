import math

import numpy as np
import pytest

from pythia import declared_feature_modulus_profile, feature_barycentric_dispersion


def test_feature_barycentric_dispersion_known_geometry():
    z = np.array([[0.], [2.], [10.], [14.]])
    blocks = np.array([[0, 1], [2, 3]])
    g, h = feature_barycentric_dispersion(z, blocks)
    assert g == pytest.approx(1.5)
    assert h == pytest.approx(2.5)


def test_mae_curve_is_monotone_and_has_tipping_point():
    result = declared_feature_modulus_profile(
        lower_score_mean=6, retained_fraction=.8, loss="MAE",
        feature_dispersion_g=2, feature_dispersion_h=5,
        declared_moduli=[0, 1, 3, 4], metric_provenance="frozen-z-v1",
        modulus_provenance="external-protocol-v1")
    assert result.bounds == pytest.approx((4.8, 3.2, 0, 0))
    assert result.tipping_modulus == pytest.approx(3)
    assert result.status == "sensitivity_only"


def test_pinball_uses_half_barycentric_penalty_for_every_level():
    common = dict(
        lower_score_mean=4, retained_fraction=1,
        feature_dispersion_g=2, feature_dispersion_h=1,
        declared_moduli=[1], metric_provenance="metric",
        modulus_provenance="declaration")
    assert declared_feature_modulus_profile(loss="Q10", **common).bounds == (3.0,)
    assert declared_feature_modulus_profile(loss="Q90", **common).bounds == (3.0,)


def test_mse_uses_squared_modulus_and_h_geometry():
    result = declared_feature_modulus_profile(
        lower_score_mean=18, retained_fraction=.5, loss="MSE",
        feature_dispersion_g=99, feature_dispersion_h=2,
        declared_moduli=[0, 2, 3], metric_provenance="metric",
        modulus_provenance="declaration")
    assert result.penalties == pytest.approx((0, 8, 18))
    assert result.bounds == pytest.approx((9, 5, 0))
    assert result.tipping_modulus == pytest.approx(3)


def test_declared_modulus_refuses_missing_provenance_and_unsupported_loss():
    kwargs = dict(
        lower_score_mean=1, retained_fraction=1,
        feature_dispersion_g=1, feature_dispersion_h=1,
        declared_moduli=[1], metric_provenance="metric",
        modulus_provenance="declaration")
    with pytest.raises(ValueError, match="supported losses"):
        declared_feature_modulus_profile(loss="W30", **kwargs)
    kwargs["modulus_provenance"] = ""
    with pytest.raises(ValueError, match="provenance"):
        declared_feature_modulus_profile(loss="MAE", **kwargs)


def test_zero_geometry_has_infinite_tipping_modulus():
    result = declared_feature_modulus_profile(
        lower_score_mean=1, retained_fraction=1, loss="MAE",
        feature_dispersion_g=0, feature_dispersion_h=0,
        declared_moduli=[0, 100], metric_provenance="constant metric",
        modulus_provenance="declaration")
    assert result.bounds == (1.0, 1.0)
    assert math.isinf(result.tipping_modulus)
