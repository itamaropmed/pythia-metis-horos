"""Known-noise partial target disclosure: group versus full-information risk.

Tabular disclosure and exact-grammar text disclosure are equivalent here.
The information-gap caps come from the synthetic DGP, not inferred by Peras.
"""
from __future__ import annotations

import json
import re

import numpy as np

from group_calibration_floor import grouped_floor
from refinement_gap_profile import sensitivity_profile, transfer_bound


def run(reps: int = 100) -> dict:
    probabilities = np.array([.5, .51])
    gold = {"MAE": 20 * np.minimum(probabilities, 1 - probabilities),
            "Q90": 2 * (1 - probabilities),
            "MSE": 400 * probabilities * (1 - probabilities)}
    clips = {"MAE": 10., "Q90": 2., "MSE": 100.}
    regex = re.compile(r"^duration outcome: (-?\d+\.\d+) minutes\.$")
    levels = (0., .25, .5, .75, 1.)
    count = 0
    gap_exceedances = w1_exceedances = 0
    full_reveal_false_zero_positive = 0
    example = []
    for seed in range(reps):
        rng = np.random.default_rng(seed + 670_000)
        calibration = {g: rng.choice((-10., 10.), 800, p=(1 - p, p))
                       for g, p in enumerate(probabilities)}
        group = np.tile((0, 1), 200)
        y = np.where(rng.random(400) < probabilities[group], 10., -10.)
        results = {loss: grouped_floor(calibration, {0: .5, 1: .5},
                                       loss=loss, clip=(-10., 10.),
                                       score_clip=clips[loss]) for loss in gold}
        for level in levels:
            mask = rng.random(400) < level
            tabular = np.where(mask, y, np.nan)
            notes = [f"duration outcome: {v:.1f} minutes." if reveal
                     else "no duration value disclosed." for v, reveal in zip(y, mask)]
            extracted = np.array([float(m.group(1)) if (m := regex.match(note))
                                  else np.nan for note in notes])
            assert np.array_equal(tabular, extracted, equal_nan=True)
            # Shuffle is a parser-specific erasure, NOT evidence that all
            # raw character strings are uninformative (digits/sign survive).
            shuffled = ["".join(rng.permutation(list(note))) for note in notes]
            assert not any(regex.match(note) for note in shuffled)
            p = probabilities[group]
            conditional_w1 = np.where(y > 0, 20 * (1 - p), 20 * p)
            eta = float(np.mean(mask * conditional_w1))
            for loss, values in gold.items():
                raw_gold = float(np.mean((~mask) * values[group]))
                gap = float(np.mean(mask * values[group]))
                base = results[loss]
                assert transfer_bound(base)["status"] == "refused"
                direct = transfer_bound(base, risk_gap_cap=gap,
                                        cap_provenance="known synthetic disclosure gap")
                transport = transfer_bound(base, mean_w1_cap=eta,
                                           cap_provenance="known synthetic clipped-law gap")
                gap_exceedances += direct["bound"] > raw_gold + 1e-12
                w1_exceedances += transport["bound"] > raw_gold + 1e-12
                count += 1
                if level == 1:
                    assert direct["bound"] == transport["bound"] == raw_gold == 0
                    false = transfer_bound(base, risk_gap_cap=0.,
                                           cap_provenance="deliberately false sufficiency cap")
                    full_reveal_false_zero_positive += false["bound"] > 0
                if seed == 0:
                    example.append({"disclosure": level, "loss": loss,
                                    "raw_gold": raw_gold,
                                    "group_bound": base["tuple_bound"],
                                    "gap_corrected": direct["bound"],
                                    "w1_corrected": transport["bound"]})
        for loss, base in results.items():
            profile = sensitivity_profile(base, np.linspace(0, 100, 21),
                                          cap_kind="risk_gap")
            assert np.all(np.diff(profile["bounds"]) <= 1e-12)
            assert profile["bounds"][-1] == 0
    assert gap_exceedances == w1_exceedances == 0
    out = {"reps": reps, "corrected_loss_cases": count,
           "risk_gap_exceedances": gap_exceedances,
           "w1_exceedances": w1_exceedances,
           "false_zero_cap_positive_at_full_disclosure": full_reveal_false_zero_positive,
           "tabular_text_representation_equivalence": "passed",
           "shuffled_exact_parser_erasure": "passed; raw-text information not certified absent",
           "example_first_seed": example}
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
