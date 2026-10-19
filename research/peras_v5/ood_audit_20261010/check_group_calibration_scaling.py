"""Sample-size diagnostic for independent DKW caps on exact-law designs.

Reports the *minimum* DKW cross-group penalty, setting empirical W1 to zero.
Thus it diagnoses nonvacuity without seeing or optimizing audit outcomes.
"""
from __future__ import annotations

import json

import numpy as np

import gold
from run_exact_law_battery import DESIGNS, make


def run(n_cal: int = 20_000, delta: float = 0.025,
        clip: tuple[float, float] = (-100.0, 400.0)) -> list[dict]:
    width = clip[1] - clip[0]
    rows = []
    for design, (n_codes, _, _) in DESIGNS.items():
        d = make(design, n_cal, 880_000)
        labels = np.column_stack([d["code"], d["concepts"][:, 0]])
        unique, counts = np.unique(labels, axis=0, return_counts=True)
        groups = {tuple(map(int, g)): int(n) for g, n in zip(unique, counts)}
        G = len(groups)
        eps = {g: float(np.sqrt(np.log(2 * G / delta) / (2 * n)))
               for g, n in groups.items()}
        # Pairs of adjacent procedure codes with the *same* note concept.
        # This is an optimistic continuous-neighborhood analogue: true W1
        # and empirical W1 are both omitted from the reported penalty.
        minimum_penalties = []
        for code in range(n_codes - 1):
            for concept in range(3):
                a, b = (code, concept), (code + 1, concept)
                if a in groups and b in groups:
                    cross_cap_floor = min(width, width * (eps[a] + eps[b]))
                    minimum_penalties.append(cross_cap_floor / 2)
        floor = gold.pointwise(d["fam"], d["s"])["MAE"].mean()
        # In the ideal balanced case, a mixed pair has average DKW penalty
        # D*eps. This sample count merely pushes that uncertainty below the
        # *original* Bayes floor, ignoring true W1, empirical W1, score
        # concentration, imbalance, and clipping loss.
        ideal_n_per_group = int(np.ceil(
            width ** 2 * np.log(2 * G / delta) / (2 * floor ** 2)))
        rows.append({"design": design, "n_cal": n_cal, "n_groups": G,
                     "median_group_size": float(np.median(counts)),
                     "median_minimum_mean_w1_penalty": float(np.median(minimum_penalties)),
                     "known_mae_floor": float(floor),
                     "penalty_over_floor": float(np.median(minimum_penalties) / floor),
                     "optimistic_total_n_for_penalty_below_floor": G * ideal_n_per_group,
                     "n_adjacent_group_pairs": len(minimum_penalties)})
    print(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    run()
