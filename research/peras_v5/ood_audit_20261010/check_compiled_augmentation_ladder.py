"""End-to-end certificate-graph audit under partial outcome disclosure.

The DGP supplies the true decision-risk gap.  This checks composition and
logical augmentation behavior; it does not estimate a real-data gap.
"""
from __future__ import annotations

import json
import numpy as np

from group_calibration_floor import grouped_floor
from peras import (
    EvidenceContext,
    compile_certificate_graph,
    gap_evidence_from_cap,
    lower_evidence_from_result,
    structural_zero_gap,
)


def run(repetitions: int = 100) -> dict:
    probabilities = np.array([.5, .51])
    gold = {
        "MAE": 20 * np.minimum(probabilities, 1 - probabilities),
        "Q90": 2 * (1 - probabilities),
        "MSE": 400 * probabilities * (1 - probabilities),
    }
    score_clips = {"MAE": 10., "Q90": 2., "MSE": 100.}
    levels = (0., .25, .5, .75, 1.)
    cases = violations = nonmonotone = 0
    full_disclosure_nonzero = false_zero_positive = 0
    examples = []
    for seed in range(repetitions):
        rng = np.random.default_rng(880_000 + seed)
        calibration = {
            g: rng.choice((-10., 10.), 800, p=(1 - p, p))
            for g, p in enumerate(probabilities)
        }
        groups = np.tile((0, 1), 200)
        outcomes = np.where(rng.random(400) < probabilities[groups], 10., -10.)
        disclosure_score = rng.random(400)
        for loss, per_group_gold in gold.items():
            base_result = grouped_floor(
                calibration, {0: .5, 1: .5}, loss=loss,
                clip=(-10., 10.), score_clip=score_clips[loss])
            context = EvidenceContext(
                target="synthetic duration", population="fixed 400-case audit mixture",
                loss=loss, outcome_scale="synthetic minutes",
                decision_time="before registered disclosure", outcome_clip=(-10., 10.))
            base = lower_evidence_from_result(
                base_result, context=context, node="group-only", evidence_id=f"base-{seed}-{loss}",
                source="grouped SAA", provenance={"seed": seed, "loss": loss},
                bound_key="tuple_bound", event_id=f"base-event-{seed}-{loss}")
            previous = float("inf")
            for level in levels:
                disclosed = disclosure_score < level
                richer_gold = float(np.mean((~disclosed) * per_group_gold[groups]))
                true_gap = float(np.mean(disclosed * per_group_gold[groups]))
                edge = gap_evidence_from_cap(
                    context=context, poorer="group-only", richer="disclosed-tabular",
                    upper_gap=true_gap, evidence_id=f"gap-{seed}-{loss}-{level}",
                    kind="declared_sensitivity", source="known synthetic DGP",
                    provenance={"seed": seed, "level": level},
                    assumptions=("DGP oracle gap supplied only for theorem validation",))
                equivalence = structural_zero_gap(
                    context=context, poorer="disclosed-tabular", richer="parsed-note",
                    evidence_id=f"equiv-{seed}-{loss}-{level}",
                    source="frozen exact-grammar parser equivalence",
                    provenance={"parser": "duration outcome: VALUE minutes"},
                    assumptions=("tabular disclosure and parsed note value are identical case by case",))
                compiled = compile_certificate_graph(
                    target_node="parsed-note", lower_evidence=(base,),
                    gap_evidence=(edge, equivalence), family_alpha=.05)
                assert compiled.status == "complete"
                violations += compiled.bound > richer_gold + 1e-12
                nonmonotone += compiled.bound > previous + 1e-12
                previous = compiled.bound
                cases += 1
                if level == 1.:
                    full_disclosure_nonzero += compiled.bound > 1e-12
                    false_edge = structural_zero_gap(
                        context=context, poorer="group-only", richer="parsed-note",
                        evidence_id=f"false-{seed}-{loss}", source="deliberately false assumption",
                        provenance={"negative_control": True},
                        assumptions=("FALSE: disclosed target has no decision value",))
                    false_result = compile_certificate_graph(
                        target_node="parsed-note", lower_evidence=(base,),
                        gap_evidence=(false_edge,), family_alpha=.05)
                    false_zero_positive += false_result.bound > 0
                if seed == 0:
                    examples.append({"loss": loss, "disclosure": level,
                                     "known_richer_risk": richer_gold,
                                     "compiled_bound": compiled.bound})
    assert violations == nonmonotone == full_disclosure_nonzero == 0
    return {
        "repetitions": repetitions, "compiled_loss_cases": cases,
        "known_risk_exceedances": violations,
        "nonmonotone_disclosure_paths": nonmonotone,
        "nonzero_bounds_at_full_disclosure": full_disclosure_nonzero,
        "false_structural_zero_positive_at_full_disclosure": false_zero_positive,
        "expected_false_zero_cases": repetitions * len(gold),
        "example_first_seed": examples,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
