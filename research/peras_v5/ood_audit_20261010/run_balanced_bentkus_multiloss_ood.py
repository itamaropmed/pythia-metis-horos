"""Fresh comparison of balanced Bentkus and weighted Hoeffding intervals."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from clipped_risk_gold import risks
from finite_group_multiloss_interval import (
    LOSSES, SCORE_CAPS, balanced_bentkus_interval, interval,
    randomized_mixture_bentkus_interval,
)
from run_exact_law_battery import DESIGNS, make


def run(design, seed, *, k=16, n_random_blocks=200):
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
    hoeffding = {loss: interval(samples, weights, loss=loss, k=k,
                                score_cap=SCORE_CAPS[loss]) for loss in LOSSES}
    bentkus = {loss: balanced_bentkus_interval(
        samples, counts, loss=loss, k=k, score_cap=SCORE_CAPS[loss]) for loss in LOSSES}
    randomized = {loss: randomized_mixture_bentkus_interval(
        samples, counts, loss=loss, k=k, score_cap=SCORE_CAPS[loss],
        n_blocks=n_random_blocks,
        randomization_seed=seed + 8_000_000) for loss in LOSSES}
    return {"design": design, "seed": seed, "gold_clipped": truth,
            "hoeffding": hoeffding, "balanced_bentkus": bentkus,
            "randomized_bentkus": randomized}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first-seed", type=int, default=93001)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--random-blocks", type=int, default=200)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("output exists")
    with args.out.open("w") as f:
        for design in DESIGNS:
            for seed in range(args.first_seed, args.first_seed + args.reps):
                row = run(design, seed, k=args.k,
                          n_random_blocks=args.random_blocks)
                f.write(json.dumps(row) + "\n")
                f.flush()
                refused = sum(v["status"] == "refused"
                              for v in row["balanced_bentkus"].values())
                random_refused = sum(v["status"] == "refused"
                                     for v in row["randomized_bentkus"].values())
                print(design, seed, "exact_refused", refused,
                      "random_refused", random_refused, flush=True)
