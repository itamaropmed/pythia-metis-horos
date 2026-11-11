"""Known-truth cluster-coverage red team for Pythia's projection lower bound.

Rows are repeated within patient clusters. The reader score is fixed without
labels; the null design makes it informative about a binary concept that has
zero outcome value conditional on the declared tabular cell. The signal design
adds that concept to the outcome law. Only aggregate simulation statistics are
printed or saved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

from pythia.core.instrumented import loo_centre_groups, projection_bound


def run(replicates: int = 500, patients: int = 180, visits: int = 4,
        seed: int = 20261024, alpha: float = 0.05) -> dict[str, object]:
    if replicates < 1 or patients < 20 or visits < 1 or not 0 < alpha < 1:
        raise ValueError("invalid simulation size or alpha")
    rng = np.random.default_rng(seed)
    results: dict[str, object] = {
        "status": "synthetic_known_truth_red_team_not_clinical_validation",
        "seed": seed,
        "replicates_per_condition": replicates,
        "patients_per_replicate": patients,
        "visits_per_patient": visits,
        "alpha": alpha,
        "independent_unit": "patient",
        "within_patient_outcomes": "perfectly_shared_binary_draw",
        "reader_score_outcome_blind": True,
    }

    for condition in ("null", "signal"):
        positive = 0
        above_truth = 0
        ratios: list[float] = []
        bounds: list[float] = []
        truths: list[float] = []
        for _ in range(replicates):
            cells_patient = rng.integers(0, 15, patients)
            concepts_patient = rng.integers(0, 2, patients)
            cells = np.repeat(cells_patient, visits)
            concepts = np.repeat(concepts_patient, visits)
            p0_patient = 0.15 + 0.7 * cells_patient / 14
            p0 = np.repeat(p0_patient, visits)

            # The reader has label-free, within-cell variation. Under null,
            # that variation is unrelated to Y once the table cell is known.
            q1 = expit(logit(p0) + 1.8 * (concepts - 0.5))
            q = np.column_stack((1 - q1, q1))
            h = loo_centre_groups(q, cells)

            effect = 1.1 if condition == "signal" else 0.0
            p_y_patient = expit(
                logit(p0_patient) + effect * (concepts_patient - 0.5)
            )
            patient_y = rng.binomial(1, p_y_patient)
            y = np.repeat(patient_y, visits)
            patient_ids = np.repeat(np.arange(patients), visits)

            result = projection_bound(
                h, y, alpha, groups=cells, entity_ids=patient_ids,
            )
            bound = float(result.bound)
            positive += bound > 1e-12
            bounds.append(bound)
            if condition == "signal":
                p_by_concept = np.column_stack((
                    expit(logit(p0_patient) + effect * (0 - 0.5)),
                    expit(logit(p0_patient) + effect * (1 - 0.5)),
                ))
                # Design-conditional Brier spread beyond the tabular cell.
                truth = float(np.mean(np.var(p_by_concept, axis=1, ddof=0)))
                truths.append(truth)
                above_truth += bound > truth + 1e-12
                ratios.append(bound / truth if truth > 0 else 0.0)

        summary: dict[str, object] = {
            "positive_bound_rate": positive / replicates,
            "mean_lower_bound": float(np.mean(bounds)),
            "median_lower_bound": float(np.median(bounds)),
        }
        if condition == "signal":
            summary.update({
                "fraction_lower_bound_above_design_truth": above_truth / replicates,
                "mean_design_truth": float(np.mean(truths)),
                "median_lower_bound_to_truth_ratio": float(np.median(ratios)),
                "fraction_with_positive_bound": positive / replicates,
            })
        else:
            summary["null_false_certificate_rate"] = positive / replicates
        results[condition] = summary
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--patients", type=int, default=180)
    parser.add_argument("--visits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.replicates, args.patients, args.visits, args.seed, args.alpha)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
