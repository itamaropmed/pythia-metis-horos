"""P1/P2/P4 sanity: no false certificates on a null text; positive bound when text informs."""
import numpy as np
from pythia.core.instrumented import (
    information_ledger, loo_centre_groups, projection_bound, reader_portfolio,
    regression_test,
)


def _data(n, effect, seed):
    r = np.random.default_rng(seed); cell = r.integers(0, 10, n); z = r.integers(0, 2, n)
    p = 1 / (1 + np.exp(-(cell / 5 - 1 + effect * (z - .5)))); y = (r.random(n) < p).astype(int)
    q = 1 / (1 + np.exp(-(cell / 5 - 1 + 1.5 * (z - .5))))
    return cell, z, y, np.column_stack([1 - q, q])


def test_null_text_no_false_certificates():
    hits = sum(projection_bound(loo_centre_groups(Q, c), y, .05, groups=c).bound > 0
               for c, _, y, Q in (_data(2000, 0.0, s) for s in range(60)))
    assert hits <= 8                      # alpha = .05 -> expect <= 3 of 60; generous slack


def test_informative_text_certified_and_portfolio():
    c, z, y, Q = _data(4000, 2.0, 1)
    assert projection_bound(loo_centre_groups(Q, c), y, .05, groups=c).bound > 0
    res = reader_portfolio(c, {"good": Q, "noise": np.full_like(Q, .5)}, y, .05)
    assert res["selected"] == "good"


def test_regression_test_null_when_new_bins_contain_signal():
    c, z, y, Q = _data(4000, 2.0, 2)
    assert not regression_test(c, z, Q, y, .05)["regression_certified"]


def test_projection_bound_rejects_uncentred_false_certificate():
    y = np.tile([0, 1], 2000)
    groups = np.zeros(len(y), dtype=int)
    H = np.tile([0., 1.], (len(y), 1))
    try:
        projection_bound(H, y, .05, groups=groups)
    except ValueError as exc:
        assert "sum to zero" in str(exc)
    else:
        raise AssertionError("uncentred scores must not produce a P1 certificate")


def test_projection_bound_checks_alignment_and_alpha():
    H = np.tile([-.5, .5], (20, 1)); y = np.tile([0, 1], 10); groups = np.zeros(20)
    try:
        projection_bound(H, y, 0.0, groups=groups)
    except ValueError as exc:
        assert "alpha" in str(exc)
    else:
        raise AssertionError("invalid alpha must be refused")


def test_clustered_bound_matches_row_bound_for_singleton_clusters():
    c, z, y, Q = _data(500, 1.0, 811)
    H = loo_centre_groups(Q, c)
    row = projection_bound(H, y, .05, groups=c)
    singleton = projection_bound(H, y, .05, groups=c, entity_ids=np.arange(len(y)))
    assert np.isclose(row.bound, singleton.bound)
    assert singleton.extra["n_independent_clusters"] == len(y)
    assert singleton.extra["sampling_unit_assumption"] == "independent_clusters_assumption"


def test_clustered_bound_uses_independent_patient_units_and_is_more_conservative():
    rng = np.random.default_rng(920)
    patients, visits = 120, 4
    entities = np.repeat(np.arange(patients), visits)
    q1 = rng.uniform(.15, .85, size=len(entities))
    Q = np.column_stack([1 - q1, q1])
    groups = np.zeros(len(entities), dtype=int)
    H = loo_centre_groups(Q, groups)
    y_patient = rng.binomial(1, .5, size=patients)
    y = np.repeat(y_patient, visits)
    row_bound = projection_bound(H, y, .05, groups=groups)
    cluster_bound = projection_bound(H, y, .05, groups=groups, entity_ids=entities)
    assert cluster_bound.extra["n_independent_clusters"] == patients
    assert cluster_bound.extra["max_rows_per_cluster"] == visits
    assert cluster_bound.bound <= row_bound.bound + 1e-12


def test_cluster_concentration_has_nominal_null_coverage_under_perfect_within_patient_dependence():
    rng = np.random.default_rng(921)
    patients, visits, replicates = 120, 4, 120
    entities = np.repeat(np.arange(patients), visits)
    groups = np.zeros(len(entities), dtype=int)
    false_certificates = 0
    for _ in range(replicates):
        q1 = rng.uniform(.1, .9, size=len(entities))
        Q = np.column_stack([1 - q1, q1])
        H = loo_centre_groups(Q, groups)
        # One Bernoulli draw per patient, copied to all visits: maximal
        # within-patient dependence and exactly zero conditional information.
        y = np.repeat(rng.binomial(1, .5, size=patients), visits)
        bound = projection_bound(H, y, .05, groups=groups, entity_ids=entities)
        false_certificates += bound.bound > 0
    assert false_certificates <= 12


def test_foldwise_reader_portfolio_bounds_pooled_oof_inside_fold_blocks():
    c, z, y, Q = _data(1200, 1.5, 91)
    folds = np.arange(len(y)) % 2
    result = reader_portfolio(
        c, {"reader": Q, "null": np.full_like(Q, .5)}, y, .05, fold_ids=folds,
        entity_ids=np.arange(len(y)),
    )
    manual = []
    for fold in (0, 1):
        ix = folds == fold
        part = reader_portfolio(c[ix], {"reader": Q[ix], "null": np.full_like(Q[ix], .5)},
                                y[ix], .025)
        manual.append(part["per_reader"]["reader"])
    assert result["foldwise"] is True
    assert np.isclose(result["per_reader"]["reader"], np.mean(manual))
    assert result["per_reader"]["null"] == 0.


def test_foldwise_information_ledger_and_regression_return_fold_diagnostics():
    c, z, y, Q = _data(1600, 1.5, 92)
    folds = np.arange(len(y)) % 2
    ledger = information_ledger(c, z, Q, Q, y, .05, fold_ids=folds,
                                entity_ids=np.arange(len(y)))
    regression = regression_test(c, z, Q, y, .05, fold_ids=folds,
                                 entity_ids=np.arange(len(y)))
    assert ledger["foldwise"] is True and len(ledger["fold_totals"]) == 2
    assert ledger["total"] <= ledger["between_a"] + ledger["beyond_a"] + 1e-12
    assert regression["foldwise"] is True and len(regression["fold_bounds"]) == 2
