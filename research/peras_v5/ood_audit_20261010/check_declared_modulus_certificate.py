"""Algebra, refusal, monotonicity, and selection-accounting checks."""
import numpy as np

from declared_modulus_certificate import declared_modulus_portfolio, loss_coefficient


rng = np.random.default_rng(20261012)
y = rng.normal(50, 10, 240)
blocks = {
    4: np.arange(240).reshape(-1, 4),
    8: np.arange(240).reshape(-1, 8),
}

for loss, caps in (("MAE", (5., 10.)), ("Q10", (2., 5.)),
                   ("Q50", (5., 10.)), ("Q90", (2., 5.)),
                   ("MSE", (50., 200.))):
    values = []
    for discrepancy in (0., .5, 1., 2.) if loss != "MSE" else (0., 5., 10., 20.):
        row = declared_modulus_portfolio(
            y, blocks, loss=loss, score_caps=caps, clip=(0., 100.),
            declared_discrepancy=discrepancy, alpha_family=.05,
            block_provenance="fixed sequential smoke blocks")
        assert row["status"] == "conditional"
        assert 0 <= row["bound"]
        assert row["alpha_per_candidate"] == .05 / (2 * 2)
        values.append(row["bound"])
    assert np.all(np.diff(values) <= 1e-12), (loss, values)

assert loss_coefficient("Q10") == loss_coefficient("Q90") == .5
refusal = declared_modulus_portfolio(
    y, blocks, loss="MAE", score_caps=(10.,), clip=(0., 100.),
    declared_discrepancy=1., block_provenance=None)
assert refusal["status"] == "refused"
print("declared modulus checks passed")
