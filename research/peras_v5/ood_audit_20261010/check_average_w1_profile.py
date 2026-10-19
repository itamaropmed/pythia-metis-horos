"""Algebraic invariants for the average-W1 sensitivity route."""
import numpy as np

from transport_tuple_bound import average_w1_sensitivity, lower_bound


def run() -> None:
    rng = np.random.default_rng(9103)
    y = rng.choice((-10.0, 10.0), size=400)
    idx = (np.arange(400)[:, None] + np.arange(8)[None, :]) % 400
    grid = np.array((0.0, 0.5, 2.0, 5.0, 10.0))
    for loss in ("MAE", "Q90", "MSE"):
        profile = average_w1_sensitivity(
            y, idx, loss=loss, clip=(-10.0, 10.0),
            mean_w1_grid=grid, alpha=0.05 / 3,
        )
        assert profile["status"] == "sensitivity_only"
        assert np.all(np.diff(profile["bounds"]) <= 1e-12)
        for eta, bound in zip(grid, profile["bounds"]):
            equivalent = lower_bound(
                y, idx, np.full(idx.shape, eta), loss=loss,
                clip=(-10.0, 10.0), alpha=0.05 / 3,
                cap_provenance="synthetic algebraic equivalence check",
            )
            assert np.isclose(bound, equivalent["bound"], atol=1e-12)
        assert np.isclose(
            profile["breakpoint"] * profile["outcome_lipschitz"],
            max(0.0, profile["statistic"] - profile["concentration_radius"]),
            atol=1e-12,
        )
    assert average_w1_sensitivity(
        y, idx, loss="W30", clip=(-10.0, 10.0),
        mean_w1_grid=grid,
    )["status"] == "refused"
    try:
        average_w1_sensitivity(y, idx, loss="MAE", clip=(-10.0, 10.0),
                               mean_w1_grid=[1.0, 0.0])
    except ValueError:
        pass
    else:
        raise AssertionError("unsorted grid must fail")
    print("average-W1 profile invariants passed")


if __name__ == "__main__":
    run()
