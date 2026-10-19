"""Fresh ten-design known-risk audit of eight clipped group losses."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from clipped_risk_gold import risks
from finite_group_multiloss_interval import LOSSES, SCORE_CAPS, interval
from run_exact_law_battery import DESIGNS, make


def run(design, seed):
    calibration = make(design, 20_000, seed + 2_000_000)
    audit = make(design, 800, seed)
    samples = defaultdict(list)
    for code, concept, y in zip(calibration["code"], calibration["concepts"][:, 0],
                                calibration["y"]):
        samples[(int(code), int(concept))].append(float(y))
    groups = [(int(code), int(concept)) for code, concept
              in zip(audit["code"], audit["concepts"][:, 0])]
    weights = {g: n / 800 for g, n in Counter(groups).items()}
    parameters = {}
    for i, group in enumerate(groups):
        parameters[group] = (audit["fam"][i], audit["mu"][i], audit["s"][i])
    truth = {loss: sum(w * risks(*parameters[g])[loss] for g, w in weights.items())
             for loss in LOSSES}
    result = {loss: interval(samples, weights, loss=loss, score_cap=SCORE_CAPS[loss])
              for loss in LOSSES}
    return {"design": design, "seed": seed, "n_cal": 20_000, "n_audit": 800,
            "gold_clipped": truth, "intervals": result}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first-seed", type=int, default=92001)
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
                print(design, seed, flush=True)
