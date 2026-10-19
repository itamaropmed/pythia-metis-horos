"""Frozen runner for REFINEMENT_NULL_CALIBRATION_REGISTRATION.md."""
from __future__ import annotations

import json
import sys

import numpy as np

from peras import finite_refinement_permutation_test
import run_family_lattice_confirmation as lattice
import run_intervention_battery as base


if __name__ == "__main__":
    output = sys.argv[1]
    with open(output, "w", encoding="utf-8") as handle:
        for mechanism_index, mechanism in enumerate(base.MECHANISMS):
            for repetition in range(100):
                seed = 660000 + 1000 * mechanism_index + repetition
                rng = np.random.default_rng(seed)
                n = 512
                procedure = rng.integers(0, 20, n)
                concept = rng.integers(0, 2, n)
                nuisance_states = (2, 4, 8)[repetition % 3]
                nuisance = rng.integers(0, nuisance_states, n)
                laws = lattice._laws(mechanism)
                index = np.minimum(
                    (rng.random(n) * laws.shape[-1]).astype(int), laws.shape[-1] - 1)
                y = laws[procedure, concept, index]
                result = finite_refinement_permutation_test(
                    procedure, nuisance, y, loss="mae", permutations=199,
                    seed=seed + 17, registry_frozen_before_outcomes=True,
                    outcome_blind_attested=True,
                    conditional_exchangeability_attested=True,
                    provenance={"registry": "independent-nuisance-null-v2"})
                handle.write(json.dumps({
                    "mechanism": mechanism, "seed": seed,
                    "nuisance_states": nuisance_states,
                    "gain": result.observed_gain, "p_value": result.p_value,
                    "reject_005": result.p_value <= .05,
                }, sort_keys=True) + "\n")
            print(mechanism, flush=True)
