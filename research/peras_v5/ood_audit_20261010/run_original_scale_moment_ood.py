"""Fresh 10-mechanism audit of original-scale moment SAA certificates.

This is a concentration/witness audit, not a predictor benchmark.  No model is
trained and no feature representation is selected.  For every mechanism we
compare:

* an untruncated SAA endpoint using declared analytic moments + Cantelli;
* the published dKelly bounded-mean endpoint applied to explicitly truncated
  SAA scores over a predeclared scale-relative cap grid.

The loss family (MAE, Q90, MSE), block-size grid, and score-cap grid share a
Bonferroni ledger.  MSE is refused when the fourth moment needed by the moment
route is infinite.  Gold risks and anchor moments are numerical population
integrals (or exact SciPy moments), never estimated from an audit sample.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import optimize, stats

from peras import (
    cantelli_saa_lower_bound,
    catoni_second_moment_saa_lower_bound,
    location_saa_score_variance_cap,
)
from repeated_environment_saa import dkelly_betting_lower_mean


LOSSES = ("MAE", "Q90", "MSE")
KS = (2, 4, 8, 16)
CAP_MULTIPLIERS = (.5, 1., 2., 4., 8.)


class Law:
    def __init__(self, name, components):
        self.name = name
        self.components = tuple(components)
        self.shift = self._ppf_raw(.5)

    def sample(self, n, rng):
        weights = np.asarray([w for w, _ in self.components], float)
        which = rng.choice(len(weights), size=n, p=weights)
        out = np.empty(n)
        for j, (_, dist) in enumerate(self.components):
            mask = which == j
            out[mask] = dist.rvs(size=int(mask.sum()), random_state=rng)
        return out - self.shift

    def _cdf_raw(self, x):
        return sum(w * dist.cdf(x) for w, dist in self.components)

    def _ppf_raw(self, q):
        lo, hi = -1., 1.
        while self._cdf_raw(lo) > q:
            lo *= 2
        while self._cdf_raw(hi) < q:
            hi *= 2
        return float(optimize.brentq(lambda x: self._cdf_raw(x) - q, lo, hi))

    def ppf(self, q):
        return self._ppf_raw(q) - self.shift

    def expect(self, fn):
        return float(sum(
            w * dist.expect(lambda x: fn(x - self.shift))
            for w, dist in self.components))

    def moments_and_gold(self):
        m1 = self.expect(lambda x: abs(x))
        m2 = self.expect(lambda x: x * x)
        mean = self.expect(lambda x: x)
        q90 = self.ppf(.9)
        pin90 = self.expect(lambda x: max(.9 * (x - q90), -.1 * (x - q90)))
        mae = m1  # every law was shifted to population median zero
        try:
            m4 = self.expect(lambda x: x ** 4)
        except Exception:
            m4 = float("inf")
        if not np.isfinite(m4) or self.name == "student_t3":
            m4 = float("inf")
        return {
            "m1": m1, "m2": m2, "m4": m4,
            "gold": {"MAE": mae, "Q90": pin90, "MSE": m2 - mean * mean},
        }


LAWS = (
    Law("normal", ((1., stats.norm()),)),
    Law("laplace", ((1., stats.laplace()),)),
    Law("logistic", ((1., stats.logistic()),)),
    Law("student_t3", ((1., stats.t(3)),)),
    Law("student_t5", ((1., stats.t(5)),)),
    Law("uniform", ((1., stats.uniform(-np.sqrt(3.), 2 * np.sqrt(3.))),)),
    Law("gamma2", ((1., stats.gamma(2.)),)),
    Law("exponential", ((1., stats.expon()),)),
    Law("lognormal_08", ((1., stats.lognorm(.8)),)),
    Law("contaminated_normal", ((.95, stats.norm()), (.05, stats.norm(scale=8.))),),
)


def saa_scores(y, loss, k):
    block = y[:len(y) // k * k].reshape(-1, k)
    if loss == "MAE":
        action = np.median(block, axis=1)
        return np.mean(np.abs(block - action[:, None]), axis=1)
    if loss == "Q90":
        action = np.quantile(block, .9, axis=1, method="inverted_cdf")
        residual = block - action[:, None]
        return np.mean(np.maximum(.9 * residual, -.1 * residual), axis=1)
    if loss == "MSE":
        return np.var(block, axis=1)
    raise ValueError(loss)


def plugin_estimate(y, loss):
    if loss == "MAE":
        return float(np.mean(np.abs(y - np.median(y))))
    if loss == "Q90":
        a = np.quantile(y, .9, method="inverted_cdf")
        u = y - a
        return float(np.mean(np.maximum(.9 * u, -.1 * u)))
    if loss == "MSE":
        return float(np.var(y))
    raise ValueError(loss)


def run_one(law, seed, n=16_000, alpha_family=.05):
    rng = np.random.default_rng(seed)
    y = law.sample(n, rng)
    population = law.moments_and_gold()
    rows = {}
    alpha_loss = alpha_family / len(LOSSES)
    for loss in LOSSES:
        gold = population["gold"][loss]
        scale = np.sqrt(population["m2"]) if loss != "MSE" else population["m2"]
        cached = {k: saa_scores(y, loss, k) for k in KS}

        moment_candidates = []
        cantelli_candidates = []
        moment_refusal = None
        for k in KS:
            try:
                cap = location_saa_score_variance_cap(
                    loss=loss, block_size=k,
                    first_absolute_moment=population["m1"],
                    second_moment=population["m2"],
                    fourth_moment=population["m4"])
            except ValueError as exc:
                moment_refusal = str(exc)
                break
            result = catoni_second_moment_saa_lower_bound(
                cached[k], loss=loss,
                score_second_moment_bounds=cap["second_score_moment_bound"],
                alpha=alpha_loss / len(KS),
                cap_source=f"analytic moments of registered {law.name} law",
                cap_frozen_before_evaluation=True)
            cantelli = cantelli_saa_lower_bound(
                cached[k], loss=loss,
                score_variance_bounds=cap["variance_bound"],
                alpha=alpha_loss / len(KS),
                cap_source=f"analytic moments of registered {law.name} law",
                cap_frozen_before_evaluation=True)
            moment_candidates.append({
                "k": k, "bound": result.lower, "radius": result.radius,
                "score_mean": result.observed_mean,
                "score_variance_bound": cap["variance_bound"],
            })
            cantelli_candidates.append({
                "k": k, "bound": cantelli.lower, "radius": cantelli.radius,
                "score_mean": cantelli.observed_mean,
            })
        if moment_refusal is None:
            moment = max(moment_candidates, key=lambda row: row["bound"])
            cantelli = max(cantelli_candidates, key=lambda row: row["bound"])
            moment_status = "conditional"
        else:
            moment = {"bound": None}
            cantelli = {"bound": None}
            moment_status = "refused"

        dkelly_candidates = []
        alpha_candidate = alpha_loss / (len(KS) * len(CAP_MULTIPLIERS))
        for k in KS:
            for multiplier in CAP_MULTIPLIERS:
                cap = float(multiplier * scale)
                values = np.minimum(cached[k], cap)
                dkelly_candidates.append({
                    "k": k, "cap_multiplier": multiplier, "score_cap": cap,
                    "bound": dkelly_betting_lower_mean(values, cap, alpha_candidate),
                    "score_mean": float(np.mean(values)),
                })
        dkelly = max(dkelly_candidates, key=lambda row: row["bound"])
        rows[loss] = {
            "gold": gold,
            "plugin_estimate": plugin_estimate(y, loss),
            "moment_status": moment_status,
            "moment_refusal": moment_refusal,
            "moment": moment,
            "cantelli_negative_control": cantelli,
            "dkelly": dkelly,
            "winner": ("moment" if moment["bound"] is not None
                       and moment["bound"] > dkelly["bound"] + 1e-12
                       else "dkelly" if moment["bound"] is None
                       or dkelly["bound"] > moment["bound"] + 1e-12 else "tie"),
        }
    return {
        "mechanism": law.name, "seed": seed, "n": n,
        "alpha_family": alpha_family,
        "population_moments": {k: (v if np.isfinite(v) else None)
                               for k, v in population.items() if k != "gold"},
        "losses": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--first-seed", type=int, default=126001)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--n", type=int, default=16_000)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists")
    with args.out.open("w") as handle:
        for law in LAWS:
            for seed in range(args.first_seed, args.first_seed + args.reps):
                row = run_one(law, seed, n=args.n)
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
                print(law.name, seed, {
                    loss: (round(cell["moment"]["bound"] or 0., 4),
                           round(cell["dkelly"]["bound"], 4), cell["winner"])
                    for loss, cell in row["losses"].items()
                }, flush=True)
