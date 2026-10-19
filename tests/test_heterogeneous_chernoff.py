import numpy as np

from pythia.core.concentration import heterogeneous_chernoff_lower_sum


def test_heterogeneous_chernoff_basic_invariants():
    x = np.array([.05, .2, .7, .1, 1.1])
    caps = np.array([.1, .4, 1.0, .2, 1.5])
    lo = heterogeneous_chernoff_lower_sum(x, caps, .05)
    assert 0 <= lo <= x.sum()
    assert np.isclose(
        heterogeneous_chernoff_lower_sum(3 * x, 3 * caps, .05),
        3 * lo, rtol=2e-6, atol=2e-8,
    )
    perm = np.array([3, 0, 4, 1, 2])
    assert np.isclose(
        heterogeneous_chernoff_lower_sum(x[perm], caps[perm], .05), lo,
        rtol=2e-6, atol=2e-8,
    )


def test_heterogeneous_chernoff_zero_observation_is_zero():
    assert heterogeneous_chernoff_lower_sum(
        np.zeros(6), np.linspace(.1, 1, 6), .05
    ) == 0.0


def test_heterogeneous_chernoff_monte_carlo_coverage():
    # Fixed non-identical Bernoulli endpoints are the extremal distributions
    # used by the proof.  This is a regression/falsification test, not a proof.
    rng = np.random.default_rng(29017)
    caps = np.linspace(.08, 1.0, 18) ** 1.4
    probs = np.linspace(.03, .82, 18)
    truth = float(np.dot(caps, probs))
    violations = 0
    for _ in range(160):
        x = caps * (rng.random(len(caps)) < probs)
        violations += heterogeneous_chernoff_lower_sum(x, caps, .05) > truth + 1e-9
    # A very loose alarm threshold avoids a flaky test while catching gross
    # inversion or optimization errors.
    assert violations <= 18


def test_heterogeneous_chernoff_exact_small_n_coverage():
    # Exhaust every endpoint-Bernoulli outcome.  Endpoint laws maximize the
    # mgf used in the proof, so this is a sharp numerical inversion audit.
    designs = [
        (np.array([.1, .25, .6, 1.0]), np.array([.05, .2, .55, .8])),
        (np.array([.03, .12, .5, 1.4]), np.array([.7, .08, .35, .15])),
        (np.array([.2, .21, .22, .9]), np.array([.1, .4, .8, .3])),
    ]
    alpha = .05
    for caps, probs in designs:
        truth = float(np.dot(caps, probs))
        violation_probability = 0.0
        for mask in range(1 << len(caps)):
            bits = np.array([(mask >> j) & 1 for j in range(len(caps))])
            x = caps * bits
            probability = float(np.prod(np.where(bits, probs, 1 - probs)))
            lower = heterogeneous_chernoff_lower_sum(x, caps, alpha)
            if lower > truth + 1e-8:
                violation_probability += probability
        assert violation_probability <= alpha + 2e-6


def test_heterogeneous_chernoff_pointwise_dominates_hoeffding():
    rng = np.random.default_rng(80211)
    for _ in range(40):
        caps = np.exp(rng.uniform(np.log(.02), np.log(2.0), 20))
        x = caps * (rng.random(20) < rng.uniform(.02, .9, 20))
        alpha = .05
        hoeffding = max(
            float(x.sum())
            - np.sqrt(float(np.dot(caps, caps)) * np.log(1 / alpha) / 2),
            0.0,
        )
        assert heterogeneous_chernoff_lower_sum(x, caps, alpha) >= hoeffding - 1e-12
