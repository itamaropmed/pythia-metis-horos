"""Frozen confirmation registered in CURVATURE_CONFIRMATION_REGISTRATION_20261014.md."""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict

import sim


FMAX = {
    "gauss": 1 / np.sqrt(2 * np.pi),
    "t3": 0.36755 * np.sqrt(3),
    "lognorm": np.exp(0.5) / np.sqrt(2 * np.pi) * np.sqrt((np.e - 1) * np.e),
}
CONFIGS = ("learned_key", "raw_align")
LOSSES = (("mae", 0.5, 2.0, 1.0), ("q90", 0.9, 1.0, 0.9))


def _one(name: str, seed: int, n: int = 4000):
    rng = np.random.default_rng(seed)
    X, f, s, y, noise = sim.draw(name, n, rng)
    nc = sim.noise_constants(noise)
    a = np.arange(n) < n // 2
    xa, ya, xb, yb, fb, sb = X[a], y[a], X[~a], y[~a], f[~a], s[~a]
    za = (xa - xa.mean(0)) / (xa.std(0) + 1e-12)
    zb = (xb - xa.mean(0)) / (xa.std(0) + 1e-12)

    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, random_state=0)
    fhat_a = cross_val_predict(model, xa, ya, cv=5)
    model.fit(xa, ya)
    fhat_b = model.predict(xb)
    noise_draw = sim.noise_sampler(noise)(np.random.default_rng(5), 1_000_000)

    for config in CONFIGS:
        if config == "learned_key":
            key_a, key_b = fhat_a[:, None], fhat_b[:, None]
            shift_a, shift_b = np.zeros_like(ya), np.zeros_like(yb)
        else:
            key_a, key_b = za, zb
            shift_a, shift_b = fhat_a, fhat_b
        blocks_a = sim.kd_blocks(key_a, 8, np.random.default_rng(seed + 17))
        blocks_b = sim.kd_blocks(key_b, 8, np.random.default_rng(seed + 19))
        y_train = ya - shift_a
        y_cert = yb - shift_b
        mu_cert = fb - shift_b

        for loss, q, multiplier, action_lipschitz in LOSSES:
            train_scores = sim.block_scores(y_train[blocks_a], loss)
            cap = max(1e-12, 1.5 * float(np.quantile(train_scores, 0.99)))
            scores = sim.block_scores(y_cert[blocks_b], loss)
            clipped = np.minimum(scores, cap)
            lcb = max(
                sim.bentkus_lower_mean(clipped, cap, 0.025),
                sim.dkelly_lower_mean(clipped, cap, 0.025),
            )
            mub, sbb = mu_cert[blocks_b], sb[blocks_b]
            bary_abs, _, _ = sim.penalties(mub, sbb, nc["qgrid"], nc["mean"])
            bary_multiplier = 1.0 if loss == "mae" else 0.5
            bary_penalty = bary_multiplier * bary_abs

            noise_action = float(np.quantile(noise_draw, q))
            action = mub + sbb * noise_action
            centre = action.mean(axis=1, keepdims=True)
            dev = action - centre
            mad_by_block = np.mean(np.abs(dev), axis=1)
            var_by_block = np.mean(dev ** 2, axis=1)
            kappa_by_block = multiplier * FMAX[noise] / np.min(sbb, axis=1)
            curvature_penalty = float(np.mean(np.minimum(
                action_lipschitz * mad_by_block,
                0.5 * kappa_by_block * var_by_block,
            )))
            gold = float(np.mean(sb) * nc[loss])
            bary_bound = max(0.0, lcb - bary_penalty)
            curvature_bound = max(0.0, lcb - curvature_penalty)
            yield {
                "mechanism": name,
                "seed": seed,
                "n": n,
                "configuration": config,
                "loss": loss.upper(),
                "gold": gold,
                "score_cap": cap,
                "lower_score_mean": lcb,
                "barycentric_penalty": bary_penalty,
                "curvature_penalty": curvature_penalty,
                "barycentric_bound": bary_bound,
                "curvature_bound": curvature_bound,
                "barycentric_ratio": bary_bound / gold,
                "curvature_ratio": curvature_bound / gold,
                "barycentric_violation": bary_bound > gold + 1e-12,
                "curvature_violation": curvature_bound > gold + 1e-12,
            }


if __name__ == "__main__":
    output = sys.argv[1]
    default_mechanisms = (
        "linear5", "friedman5", "sparse20", "hifreq2", "hetero5",
        "t3_5", "skew5", "discrete3", "highSNR5", "lowSNR5",
    )
    mechanisms = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else default_mechanisms
    with open(output, "w", encoding="utf-8") as handle:
        for mechanism in mechanisms:
            for run_seed in range(41001, 41011):
                for row in _one(mechanism, run_seed):
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(mechanism, flush=True)
