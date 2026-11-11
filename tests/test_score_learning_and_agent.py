"""Score learning (stacking), scalar projection bound, agent question validation (synthetic, no network)."""
import numpy as np
import pytest
from horos.represent.score_learning import variational_scores, oof_scores, stack_weights, stack_scores
from pythia.core.instrumented import loo_centre_groups, projection_bound, projection_bound_scalar, loo_centre_scalar
from metis.agent.reader_search import validate_questions


def _data(n, seed, effect=1.5):
    r = np.random.default_rng(seed); g = r.integers(0, 8, n); a = r.integers(0, 2, n); b = r.integers(0, 2, n)
    p = 1 / (1 + np.exp(-(g / 4 - 1 + effect * (a - .5) + effect * (b - .5)))); y = (r.random(n) < p).astype(int)
    return g, a[:, None].astype(float), b[:, None].astype(float), y


def test_stack_of_complementary_readers_beats_best_single():
    g, A, B, y = _data(8000, 1); fit = np.arange(8000) < 4000; ev = ~fit
    HA = variational_scores(A[fit], y[fit], g[fit], A[ev], g[ev], 2); HB = variational_scores(B[fit], y[fit], g[fit], B[ev], g[ev], 2)
    w = stack_weights([oof_scores(variational_scores, X[fit], y[fit], g[fit], 2, entity_ids=np.arange(int(fit.sum()))) for X in (A, B)], y[fit], g[fit])
    bound = lambda H, a: projection_bound(loo_centre_groups(H, g[ev]), y[ev], a, groups=g[ev]).bound
    assert bound(stack_scores([HA, HB], w), .05) > max(bound(HA, .025), bound(HB, .025))


def test_null_text_no_false_certificate_stack_and_scalar():
    hits = 0
    for s in range(30):
        g, A, B, y = _data(2000, 100 + s, effect=0.0); fit = np.arange(2000) < 1000; ev = ~fit
        w = stack_weights([oof_scores(variational_scores, X[fit], y[fit], g[fit], 2, entity_ids=np.arange(int(fit.sum()))) for X in (A, B)], y[fit], g[fit])
        H = stack_scores([variational_scores(X[fit], y[fit], g[fit], X[ev], g[ev], 2) for X in (A, B)], w)
        hits += projection_bound(loo_centre_groups(H, g[ev]), y[ev], .05, groups=g[ev]).bound > 0
        z = np.random.default_rng(s).random(1000)
        hits += projection_bound_scalar(loo_centre_scalar(A[ev, 0], g[ev]), z, 1.0, .05).bound > 0
    assert hits <= 6


def test_validate_questions_filters_malformed():
    q = validate_questions({"questions": {"Obese?": {"type": "noul", "instructions": "Is the patient obese?",
                                                     "criteria": {"true": "yes", "false": "no"}},
                                          "bad": {"type": "score", "instructions": "x", "criteria": ["a"]}}})
    assert list(q) == ["obese_"]
    with pytest.raises(ValueError):
        validate_questions({"questions": {}})


def test_eb_scalar_bound_null_and_signal():
    from pythia.core.instrumented import eb_scalar_bound, loo_centre_scalar
    r = np.random.default_rng(0); hits = 0
    for s in range(40):
        g = r.integers(0, 6, 1500); h = loo_centre_scalar(r.normal(size=1500), g); y = r.normal(0, 20, 1500)
        hits += eb_scalar_bound(h, y, 60, .05, groups=g) > 0
    assert hits <= 3
    g = r.integers(0, 6, 4000); x = r.normal(size=4000); y = 15 * x + r.normal(0, 20, 4000)
    assert eb_scalar_bound(loo_centre_scalar(x, g), y, 60, .05, groups=g) > 0


def test_eb_scalar_bound_clustered_is_more_conservative():
    from pythia.core.instrumented import eb_scalar_bound, loo_centre_scalar
    r = np.random.default_rng(5); ent = np.repeat(np.arange(800), 4); g = np.zeros(len(ent), int)
    x = np.repeat(r.normal(size=800), 4); y = 12 * x + np.repeat(r.normal(0, 20, 800), 4)
    h = loo_centre_scalar(x, g)
    assert eb_scalar_bound(h, y, 60, .05, groups=g, entity_ids=ent) <= eb_scalar_bound(h, y, 60, .05, groups=g) + 1e-12
