"""Frozen runner for REFINEMENT_SIGNAL_CONFIRMATION_REGISTRATION.md."""
from __future__ import annotations

import json
import sys

import numpy as np

from peras import finite_refinement_permutation_test
import run_family_lattice_confirmation as lattice
import run_intervention_battery as base


def run_one(mechanism, seed):
    rng = np.random.default_rng(seed)
    n = 1024
    procedure = rng.integers(0, 20, n)
    concept = rng.integers(0, 2, n)
    nuisance = rng.integers(0, 4, n)
    laws = lattice._laws(mechanism)
    index = np.minimum((rng.random(n) * laws.shape[-1]).astype(int), laws.shape[-1] - 1)
    y = laws[procedure, concept, index]
    common = dict(
        coarse_labels=procedure, outcomes=y, loss="mae", permutations=99,
        registry_frozen_before_outcomes=True, outcome_blind_attested=True,
        conditional_exchangeability_attested=True)
    informative = finite_refinement_permutation_test(
        refinement_labels=concept, seed=seed * 2,
        provenance={"registry": "pre-outcome-binary-concept-v1"}, **common)
    null = finite_refinement_permutation_test(
        refinement_labels=nuisance, seed=seed * 2 + 1,
        provenance={"registry": "independent-four-level-nuisance-v1"}, **common)
    return {
        "mechanism": mechanism, "seed": seed,
        "informative_gain": informative.observed_gain,
        "informative_p": informative.p_value,
        "informative_reject_0025": informative.p_value <= .025,
        "null_gain": null.observed_gain,
        "null_p": null.p_value,
        "null_reject_0025": null.p_value <= .025,
    }


if __name__ == "__main__":
    output = sys.argv[1]
    with open(output, "w", encoding="utf-8") as handle:
        for mechanism in base.MECHANISMS:
            for seed in range(65001, 65051):
                handle.write(json.dumps(run_one(mechanism, seed), sort_keys=True) + "\n")
            print(mechanism, flush=True)
