"""Known-gold development runner for fixed-policy information value."""
from __future__ import annotations

import json
import sys

import numpy as np

from peras import fixed_policy_information_value_lower_bound
import run_family_lattice_confirmation as lattice
import run_intervention_battery as base


def run_one(mechanism, seed, n=2048):
    rng = np.random.default_rng(seed)
    procedure = rng.integers(0, 20, n)
    concept = rng.integers(0, 2, n)
    laws, fine_gold, coarse_gold, _, _ = lattice._tables(mechanism)
    index = np.minimum((rng.random(n) * laws.shape[-1]).astype(int), laws.shape[-1] - 1)
    y = laws[procedure, concept, index]
    actions = np.median(laws, axis=2)
    policy_loss = np.abs(y - actions[procedure, concept])
    gold = float(np.mean(coarse_gold[procedure] - fine_gold[procedure, concept]))
    cell_caps = {str(cell): float(np.ptp(laws[cell])) for cell in range(20)}
    policy_cap = float(np.max(np.abs(
        laws - actions[:, :, None])))
    common = dict(
        coarse_labels=procedure, outcomes=y, entities=np.arange(n),
        policy_losses=policy_loss, loss="mae", loss_range_bound=policy_cap,
        alpha=.05, representation_provenance="synthetic-procedure-v1",
        support_provenance=f"analytic-{mechanism}-support-v1",
        policy_provenance=f"analytic-{mechanism}-concept-median-v1",
        iid_rows_attested=True, outcome_blind_coarse_attested=True,
        policy_frozen_before_evaluation=True,
        policy_refinement_measurable_attested=True,
        policy_preoutcome_attested=True)
    improved = fixed_policy_information_value_lower_bound(
        outcome_range_bound=None, cell_outcome_range_bounds=cell_caps,
        cell_support_provenance=f"analytic-{mechanism}-procedure-support-v1",
        upper_methods=("dKelly",),
        **common)
    baseline = fixed_policy_information_value_lower_bound(
        outcome_range_bound=float(np.ptp(laws)), upper_methods=("Hoeffding",),
        **common)
    return {
        "mechanism": mechanism, "seed": seed, "gold_value": gold,
        "bound": improved.value_lower_bound,
        "ratio": improved.value_lower_bound / gold if gold > 0 else None,
        "violation": improved.value_lower_bound > gold + 1e-12,
        "positive": improved.value_lower_bound > 0,
        "selected_upper": improved.selected_upper_method,
        "policy_risk": improved.policy_risk_estimate,
        "policy_upper": improved.policy_risk_upper,
        "coarse_lower": improved.coarse_risk_lower,
        "baseline_bound": baseline.value_lower_bound,
        "baseline_ratio": baseline.value_lower_bound / gold if gold > 0 else None,
        "strict_baseline_win": improved.value_lower_bound > baseline.value_lower_bound + 1e-12,
    }


if __name__ == "__main__":
    output = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 67001
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
    with open(output, "w", encoding="utf-8") as handle:
        for mechanism in base.MECHANISMS:
            for seed in range(start, start + count):
                handle.write(json.dumps(run_one(mechanism, seed, n), sort_keys=True) + "\n")
            print(mechanism, flush=True)
