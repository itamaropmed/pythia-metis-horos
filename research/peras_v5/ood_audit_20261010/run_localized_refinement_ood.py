"""Fresh ten-mechanism tabular/text target-disclosure sensitivity audit."""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from clipped_risk_gold import risks
from finite_group_multiloss_interval import LOSSES, SCORE_CAPS, interval
from refinement_gap_profile import localized_refinement_profile
from run_exact_law_battery import DESIGNS, make

LEVELS = (0., .10, .25, .50, 1.)
PATTERN = re.compile(
    r"^duration outcome: ([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?) minutes\.$")


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
    per_group_gold = {g: risks(*parameters[g]) for g in weights}
    base = {loss: interval(samples, weights, loss=loss,
                           score_cap=SCORE_CAPS[loss]) for loss in LOSSES}
    rng = np.random.default_rng(seed + 7_000_000)
    disclosure_score = rng.random(800)
    rows = []
    for level in LEVELS:
        mask = disclosure_score < level
        tabular = np.where(mask, audit["y"], np.nan)
        notes = [f"duration outcome: {float(value)!r} minutes." if reveal
                 else "no duration value disclosed."
                 for value, reveal in zip(audit["y"], mask)]
        parsed = np.array([float(m.group(1)) if (m := PATTERN.match(note)) else np.nan
                           for note in notes])
        parser_equivalent = bool(np.array_equal(tabular, parsed, equal_nan=True))
        shuffled = ["".join(rng.permutation(list(note))) for note in notes]
        shuffled_exact_matches = sum(PATTERN.match(note) is not None for note in shuffled)
        epsilon = float(mask.mean())
        losses = {}
        for loss in LOSSES:
            full_gold = float(np.mean([
                0. if mask[i] else per_group_gold[groups[i]][loss]
                for i in range(800)]))
            profile = localized_refinement_profile(base[loss], [epsilon])
            losses[loss] = {"known_rich_gold": full_gold,
                            "localized_bound": profile["bounds"][0],
                            "epsilon_breakpoint": profile["epsilon_breakpoint"]}
        rows.append({"nominal_disclosure": level, "realized_epsilon": epsilon,
                     "tabular_text_parser_equivalent": parser_equivalent,
                     "shuffled_exact_parser_matches": shuffled_exact_matches,
                     "losses": losses})
    return {"design": design, "seed": seed, "levels": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first-seed", type=int, default=97001)
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
