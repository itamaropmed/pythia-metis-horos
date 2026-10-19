import numpy as np
from pythia.core.multiscale_pair_extrapolation import nested_matchings, extrapolation_certificate


def test_nested_matchings_are_perfect_and_nested_in_scale():
    X = np.random.default_rng(0).random((512, 3)); mats = nested_matchings(X, 4)
    for P, s in mats:
        assert sorted(P.ravel().tolist()) == list(range(512))
    S = [s for _, s in mats]
    assert all(S[i] < S[i + 1] for i in range(3))


def test_equal_floor_identity_exact():
    rng = np.random.default_rng(1); X = rng.random((256, 2)); P = rng.dirichlet(np.ones(3), 256)
    G = (1 - (P ** 2).sum(1)).mean()
    for M, _ in nested_matchings(X, 3):
        EQ = (1 - (P[M[:, 0]] * P[M[:, 1]]).sum(1)).mean()
        D = (.5 * ((P[M[:, 0]] - P[M[:, 1]]) ** 2).sum(1)).mean()
        assert abs(EQ - (G + D)) < 1e-12


def test_valid_on_smooth_law():
    viol = 0
    for rep in range(10):
        r = np.random.default_rng(10 + rep); X = r.random((2048, 2))
        p = 1 / (1 + np.exp(-3 * (X[:, 0] + X[:, 1] - 1))); y = (r.random(2048) < p).astype(int)
        b = extrapolation_certificate(nested_matchings(X, 4), y, 2, .05, (1.0, .5))["bound"]
        viol += b > np.minimum(p, 1 - p).mean()
    assert viol == 0


def test_fixed_bet_bounds_cover_non_identical_means():
    from pythia.core.multiscale_pair_extrapolation import betting_lower, betting_upper
    miss = 0
    for rep in range(200):
        r = np.random.default_rng(100 + rep); mu = r.beta(.5, 8, 400)
        z = np.minimum(r.exponential(mu), 1.0); true = np.minimum(r.exponential(mu, (2000, 400)), 1).mean()
        lo, hi = betting_lower(z, .05), betting_upper(z, .05)
        miss += not (lo <= true <= hi)
    assert miss <= 12


def test_brier_target_and_anchor():
    from pythia.core.multiscale_pair_extrapolation import certificate_v2, replicate_anchor_v3
    r = np.random.default_rng(7); n = 4000; X = r.random((n, 2))
    p = np.where(r.random(n) < .5, .3, .7)                      # unresolvable micro-variation
    y = (r.random(n) < p).astype(int); sub = np.sort(r.choice(n, 1600, replace=False)); y2 = (r.random(1600) < p[sub]).astype(int)
    an = replicate_anchor_v3(X, y, sub, y2, 2, 1.0)
    assert an["substudy_bound"] <= .3 and an["T"] > 0
    q = 1 / (1 + np.exp(-3 * (X[:, 0] + X[:, 1] - 1))); y = (r.random(n) < q).astype(int)
    from pythia.core.multiscale_pair_extrapolation import nested_matchings
    b = certificate_v2(nested_matchings(X[:3968], 4), y[:3968], 2, .05, (.5,), "brier", conc="kl")["bound"]
    assert b <= (2 * q[:3968] * (1 - q[:3968])).mean()


def test_sandwich_upper_and_information_are_valid_without_assumptions():
    from pythia.core.multiscale_pair_extrapolation import sandwich
    viol = 0
    for rep in range(10):
        r = np.random.default_rng(300 + rep); n = 2048; X = r.random((n, 2))
        p = np.where(r.random(n) < .5, .2, .8)                        # adversarial micro-variation
        y = (r.random(n) < p).astype(int); P = np.column_stack([1 - p, p])
        sw = sandwich(nested_matchings(X, 4, perm_seed=rep), y, 2, .05, .5)
        G = (1 - (P ** 2).sum(1)).mean(); I = ((P - P.mean(0)) ** 2).sum(1).mean()
        viol += (sw["brier_floor"][1] < G) + (sw["info"][0] > I)
    assert viol == 0


def test_perm_seed_returns_perfect_matchings():
    X = np.random.default_rng(0).random((256, 2))
    for P, s in nested_matchings(X, 3, perm_seed=5):
        assert sorted(P.ravel().tolist()) == list(range(256))


def test_matchings_with_heavy_ties_have_no_self_pairs():
    X = np.random.default_rng(0).integers(0, 2, (512, 3)).astype(float)      # 8 distinct points, massive ties
    for P, s in nested_matchings(X, 4, perm_seed=1):
        assert (P[:, 0] != P[:, 1]).all() and sorted(P.ravel().tolist()) == list(range(512))
