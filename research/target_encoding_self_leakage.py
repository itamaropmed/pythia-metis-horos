"""Known-truth stress test for self-including target encodings.

This synthetic experiment compares the former self-including category mean,
the current leave-one-patient-out training encoding, and a no-encoding control.
No clinical data are loaded or written.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from horos.models.mayo import _encode


def run(n_seeds: int = 20, n_train: int = 2400, n_test: int = 800,
        n_categories: int = 700) -> dict:
    methods = {"self_including": [], "leave_one_patient_out": [], "tabular_only": []}
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed + 410)
        category_effect = rng.normal(0., 5., n_categories)
        x_train, x_test = rng.normal(size=n_train), rng.normal(size=n_test)
        category_train = rng.integers(0, n_categories, n_train)
        category_test = rng.integers(0, n_categories, n_test)
        y_train = 2. * x_train + category_effect[category_train] + rng.normal(0., 2., n_train)
        y_test = 2. * x_test + category_effect[category_test] + rng.normal(0., 2., n_test)
        groups = np.asarray([f"patient-{i}" for i in range(n_train)])
        keys = pd.Series(category_train.astype(str))
        apply = pd.Series(category_test.astype(str))
        target = pd.Series(y_train)

        # Negative control: the row's own outcome participates in its training
        # category statistic. This is intentionally not the production method.
        prior = float(target.mean())
        stats = pd.DataFrame({"key": keys, "y": target}).groupby("key").y.agg(["mean", "count"])
        mapping = (stats["count"] * stats["mean"] + 10. * prior) / (stats["count"] + 10.)
        self_encoded = keys.map(mapping).fillna(prior).to_numpy(float)
        test_encoded = apply.map(mapping).fillna(prior).to_numpy(float)
        loo_encoded, loo_test_encoded = _encode(keys, target, apply, 10., groups=groups)
        variants = (
            ("self_including", self_encoded, test_encoded),
            ("leave_one_patient_out", loo_encoded, loo_test_encoded),
            ("tabular_only", np.zeros(n_train), np.zeros(n_test)),
        )
        for name, train_encoding, test_encoding in variants:
            model = HistGradientBoostingRegressor(
                max_iter=120, max_leaf_nodes=15, min_samples_leaf=20,
                l2_regularization=1., random_state=seed,
            )
            model.fit(np.column_stack([x_train, train_encoding]), y_train)
            prediction = model.predict(np.column_stack([x_test, test_encoding]))
            methods[name].append(float(mean_absolute_error(y_test, prediction)))

    means = {name: float(np.mean(values)) for name, values in methods.items()}
    paired = np.asarray(methods["leave_one_patient_out"]) - np.asarray(methods["self_including"])
    return {
        "status": "synthetic_known_truth_diagnostic_not_mayo",
        "seeds": n_seeds,
        "n_train": n_train,
        "n_test": n_test,
        "n_categories": n_categories,
        "heldout_mae_mean": means,
        "heldout_mae_sd": {
            name: float(np.std(values, ddof=1)) for name, values in methods.items()
        },
        "paired_loo_minus_self_including_mae": float(np.mean(paired)),
        "loo_wins": int(np.sum(paired < 0)),
        "caveat": "The self-including method is a deliberately leaky training transformation; held-out outcomes are never used by either fitted model.",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
