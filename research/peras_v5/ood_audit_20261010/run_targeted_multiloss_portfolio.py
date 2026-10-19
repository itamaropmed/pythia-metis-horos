"""Fresh selection-safe 8-loss portfolio with one targeted MSE upper route."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from clipped_risk_gold import risks
from finite_group_multiloss_interval import (
    LOSSES, SCORE_CAPS, interval, randomized_mixture_bentkus_interval,
)
from run_exact_law_battery import DESIGNS, make

ALPHA_EVENT = .05 / 17  # 16 Hoeffding endpoints + 1 randomized MSE upper


def run(design, seed):
    calibration = make(design, 20_000, seed + 2_000_000)
    audit = make(design, 800, seed)
    samples = defaultdict(list)
    for code, concept, y in zip(calibration["code"], calibration["concepts"][:, 0],
                                calibration["y"]):
        samples[(int(code), int(concept))].append(float(y))
    groups = [(int(code), int(concept)) for code, concept
              in zip(audit["code"], audit["concepts"][:, 0])]
    counts = Counter(groups)
    weights = {g: n / 800 for g, n in counts.items()}
    parameters = {}
    for i, group in enumerate(groups):
        parameters[group] = (audit["fam"][i], audit["mu"][i], audit["s"][i])
    truth = {loss: sum(w * risks(*parameters[g])[loss] for g, w in weights.items())
             for loss in LOSSES}
    baseline = {loss: interval(samples, weights, loss=loss,
                               score_cap=SCORE_CAPS[loss],
                               alpha_endpoint=.05 / 16) for loss in LOSSES}
    hoeffding = {loss: interval(samples, weights, loss=loss,
                                score_cap=SCORE_CAPS[loss],
                                alpha_endpoint=ALPHA_EVENT) for loss in LOSSES}
    random_mse = randomized_mixture_bentkus_interval(
        samples, counts, loss="MSE", score_cap=SCORE_CAPS["MSE"], k=16,
        n_blocks=200, randomization_seed=seed + 8_000_000,
        alpha_endpoint=ALPHA_EVENT)
    portfolio = {loss: {"lower": row["lower"], "upper": row["upper"],
                        "upper_route": "weighted_hoeffding"}
                 for loss, row in hoeffding.items()}
    if random_mse["status"] == "conditional":
        portfolio["MSE"]["upper"] = min(
            portfolio["MSE"]["upper"], random_mse["upper"])
        if random_mse["upper"] < hoeffding["MSE"]["upper"]:
            portfolio["MSE"]["upper_route"] = "registered_randomized_bentkus"
    return {"design": design, "seed": seed, "gold_clipped": truth,
            "baseline_alpha_16": baseline, "hoeffding_alpha_17": hoeffding,
            "randomized_mse": random_mse, "portfolio": portfolio,
            "joint_claim": "unconditional >=95% over registered randomization and calibration outcomes for 16 Hoeffding endpoints plus one randomized MSE upper event; refusals make no numeric claim; all structural assumptions remain conditional"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first-seed", type=int, default=95001)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("output exists")
    with args.out.open("w") as f:
        for design in DESIGNS:
            for seed in range(args.first_seed, args.first_seed + args.reps):
                row = run(design, seed)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(design, seed, row["randomized_mse"]["status"],
                      row["portfolio"]["MSE"]["upper_route"], flush=True)
