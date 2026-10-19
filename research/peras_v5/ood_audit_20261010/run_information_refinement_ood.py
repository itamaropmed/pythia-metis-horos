"""Fresh coarse-to-fine information-budget audit on ten known-law mechanisms."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from clipped_risk_gold import risks
from finite_group_multiloss_interval import LOSSES, SCORE_CAPS, interval
from refinement_gap_profile import (
    information_refinement_profile,
    registered_conditional_entropy,
)
from run_exact_law_battery import DESIGNS, make


def run(design, seed):
    calibration = make(design, 20_000, seed + 2_000_000)
    audit = make(design, 800, seed)
    coarse_samples = defaultdict(list)
    fine_samples = defaultdict(list)
    for code, concept, y in zip(calibration["code"], calibration["concepts"][:, 0],
                                calibration["y"]):
        coarse_samples[int(code)].append(float(y))
        fine_samples[(int(code), int(concept))].append(float(y))
    coarse_labels = [int(code) for code in audit["code"]]
    fine_labels = [(int(code), int(concept)) for code, concept
                   in zip(audit["code"], audit["concepts"][:, 0])]
    coarse_weights = {g: n / 800 for g, n in Counter(coarse_labels).items()}
    fine_weights = {g: n / 800 for g, n in Counter(fine_labels).items()}
    entropy = registered_conditional_entropy(coarse_labels, fine_labels)
    parameters = {}
    for i, group in enumerate(fine_labels):
        current = (audit["fam"][i], audit["mu"][i], audit["s"][i])
        if group in parameters and parameters[group] != current:
            raise AssertionError("fine label is not an exact law key")
        parameters[group] = current
    fine_gold_by_group = {g: risks(*parameters[g]) for g in fine_weights}

    out = {}
    for loss in LOSSES:
        coarse = interval(coarse_samples, coarse_weights, loss=loss,
                          score_cap=SCORE_CAPS[loss])
        direct_fine = interval(fine_samples, fine_weights, loss=loss,
                               score_cap=SCORE_CAPS[loss])
        profile = information_refinement_profile(
            coarse, [entropy],
            representation_provenance="fixed pre-outcome (code, note-concept) target labels",
            pre_outcome_registered=True)
        fine_gold = sum(fine_weights[g] * fine_gold_by_group[g][loss]
                        for g in fine_weights)
        out[loss] = {
            "fine_gold": float(fine_gold),
            "coarse_lower": coarse["lower"],
            "information_transfer_lower": profile["bounds"][0],
            "information_gap_cap": profile["gap_caps"][0],
            "information_route": profile["routes"][0],
            "direct_fine_lower": direct_fine["lower"],
        }
    return {"design": design, "seed": seed, "n_audit": 800,
            "n_calibration": 20_000,
            "conditional_refinement_entropy_nats": entropy,
            "losses": out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--first-seed", type=int, default=98001)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("output exists")
    with args.out.open("w") as handle:
        for design in DESIGNS:
            for seed in range(args.first_seed, args.first_seed + args.reps):
                row = run(design, seed)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                print(design, seed, flush=True)
