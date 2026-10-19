import numpy as np
from pythia.core.graph_dominated_saa import (chernoff_kl_lower, dependence_audit, dependency_colouring,
                                            dependence_sensitivity_curve, graph_dominated_saa_lower_bound)


def test_colouring_is_proper_and_optimal_on_cliques():
    r = np.random.default_rng(0)
    n = 400
    units = {"patient": r.integers(0, 150, n), "day": r.integers(0, 40, n)}
    colour, K, omega = dependency_colouring(units, n)
    for v in units.values():
        for u in np.unique(v):
            c = colour[v == u]
            assert len(set(c)) == len(c)          # no two cases of one unit share a colour
    assert K >= omega
    colour, K, omega = dependency_colouring({"row": np.arange(50)}, 50)
    assert K == 1 and omega == 1


def test_chernoff_inversion_properties():
    assert chernoff_kl_lower(0., 1., 100, .05) == 0.
    a, b = chernoff_kl_lower(.3, 1., 100, .05), chernoff_kl_lower(.3, 1., 1000, .05)
    assert 0 < a < b < .3
    curve = dependence_sensitivity_curve(.3, 1., 1000, (1, 4, 16), .05)
    assert curve[0]["bound"] > curve[1]["bound"] > curve[2]["bound"]


def test_refusal_without_dependency_contract():
    res = graph_dominated_saa_lower_bound(np.zeros(10), np.arange(10.), {"row": np.arange(10)}, loss="mae",
                                          caps=(1.,), representation_provenance="x", dependency_graph_attested=False,
                                          outcome_blind_labels_attested=True)
    assert res.status == "refused"


def test_valid_under_shared_day_shocks_where_row_iid_is_not_required():
    from math import sqrt, pi
    r = np.random.default_rng(4)
    miss = 0
    for rep in range(40):
        day = np.repeat(np.arange(30), 20); n = len(day)
        cell = r.integers(0, 2, n)
        y = 2. * cell + r.standard_normal(30)[day] + r.standard_normal(n)
        gold = sqrt(2.) * sqrt(2 / pi)
        res = graph_dominated_saa_lower_bound(cell, y, {"day": day}, loss="mae", caps=(2., 4.), alpha=.1,
                                              representation_provenance="t", dependency_graph_attested=True,
                                              outcome_blind_labels_attested=True)
        miss += res.bound > gold
        assert res.diagnostics["selected_colours"] == 20
    assert miss <= 4


def test_dependence_audit_detects_shared_shocks():
    r = np.random.default_rng(9)
    day = np.repeat(np.arange(60), 10); n = len(day)
    cell = r.integers(0, 3, n)
    y = cell + 2 * r.standard_normal(60)[day] + r.standard_normal(n)
    out = dependence_audit(y, cell, {"day": day, "random": r.integers(0, 60, n)}, n_perm=199)
    assert out["day"]["p_value"] <= .01
    assert out["random"]["p_value"] > .01
