"""Untouched multi-loss confirmation of direct finite decision-gap caps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from peras import (
    bounded_loss_gap_from_cmi,
    finite_cmi_upper_bound,
    finite_coordinate_bernstein_gap_upper_bound,
    finite_decision_gap_upper_bound,
    finite_policy_gap_upper_bound,
)
from run_finite_cmi_gap_ood import MECHANISMS


Y = np.arange(3, dtype=float)
A = np.arange(3, dtype=float)
RESIDUAL = Y[:, None] - A[None, :]
LOSSES = {
    "MAE": np.abs(RESIDUAL),
    "MSE": RESIDUAL ** 2,
    "Q10": np.maximum(.1 * RESIDUAL, -.9 * RESIDUAL),
    "Q50": np.maximum(.5 * RESIDUAL, -.5 * RESIDUAL),
    "Q90": np.maximum(.9 * RESIDUAL, -.1 * RESIDUAL),
    "W_exact_miss": (RESIDUAL != 0).astype(float),
}
FROZEN_GAP_ROUTER = {
    "MAE": "finite_policy_hoeffding",
    "MSE": "finite_policy_hoeffding",
    "Q10": "coordinate_empirical_bernstein",
    "Q50": "finite_policy_hoeffding",
    "Q90": "coordinate_empirical_bernstein",
    "W_exact_miss": "finite_policy_hoeffding",
}


def population_gap(joint, matrix):
    coarse = np.einsum("yzg,ya->ga", joint, matrix)
    fine = np.einsum("yzg,ya->gza", joint, matrix)
    return (float(np.min(coarse, axis=1).sum()),
            float(np.min(fine, axis=2).sum()))


def run(name, joint, seed, n=3_000):
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, joint.ravel())
    flat = np.repeat(np.arange(12), counts)
    y, z, g = flat // 4, (flat // 2) % 2, flat % 2
    alpha_loss = .05 / len(LOSSES)
    cmi = finite_cmi_upper_bound(
        y, z, g, y_alphabet=(0, 1, 2), z_alphabet=(0, 1),
        g_alphabet=(0, 1), alpha=alpha_loss,
        alphabet_frozen_before_evaluation=True,
        provenance={"registry": "finite-decision-multiloss-v1", "mechanism": name})
    rows = {}
    for loss, matrix in LOSSES.items():
        amplitude = float(matrix.max() - matrix.min())
        pinsker = bounded_loss_gap_from_cmi(
            cmi, loss=loss, loss_range=amplitude, maximum_bayes_risk=amplitude)
        direct = finite_decision_gap_upper_bound(
            y, z, g, y_alphabet=(0, 1, 2), z_alphabet=(0, 1),
            g_alphabet=(0, 1), loss=loss, loss_matrix=matrix,
            alpha=alpha_loss, alphabet_frozen_before_evaluation=True,
            provenance={"registry": "finite-decision-multiloss-v1", "mechanism": name},
            maximum_bayes_risk=amplitude)
        policy = finite_policy_gap_upper_bound(
            matrix[y], z, g, loss=loss, action_labels=(0, 1, 2),
            z_alphabet=(0, 1), g_alphabet=(0, 1),
            loss_bounds=(float(matrix.min()), float(matrix.max())),
            alpha=alpha_loss, registry_frozen_before_evaluation=True,
            action_class_complete_for_target=True,
            provenance={"registry": "finite-policy-multiloss-v1", "mechanism": name},
            maximum_bayes_risk=amplitude)
        coordinate = finite_coordinate_bernstein_gap_upper_bound(
            matrix[y], z, g, loss=loss, action_labels=(0, 1, 2),
            z_alphabet=(0, 1), g_alphabet=(0, 1),
            loss_bounds=(float(matrix.min()), float(matrix.max())),
            alpha=alpha_loss, registry_frozen_before_evaluation=True,
            action_class_complete_for_target=True,
            provenance={"registry": "finite-coordinate-multiloss-v1", "mechanism": name},
            maximum_bayes_risk=amplitude)
        selected_name = FROZEN_GAP_ROUTER[loss]
        selected = coordinate if selected_name == "coordinate_empirical_bernstein" else policy
        coarse, fine = population_gap(joint, matrix)
        truth = coarse - fine
        rows[loss] = {
            "true_coarse_risk": coarse, "true_fine_risk": fine,
            "true_gap": truth, "plugin_gap": direct.plugin_gap,
            "direct_gap_cap": direct.gap_cap,
            "policy_gap_cap": policy.gap_cap,
            "coordinate_gap_cap": coordinate.gap_cap,
            "selected_route": selected_name,
            "selected_gap_cap": selected.gap_cap,
            "pinsker_gap_cap": pinsker["gap_cap"],
            "direct_covered": bool(direct.gap_cap + 1e-12 >= truth),
            "pinsker_covered": bool(pinsker["gap_cap"] + 1e-12 >= truth),
            "direct_strict_win": bool(direct.gap_cap < pinsker["gap_cap"] - 1e-12),
            "policy_covered": bool(policy.gap_cap + 1e-12 >= truth),
            "policy_strict_win_vs_joint_l1": bool(policy.gap_cap < direct.gap_cap - 1e-12),
            "policy_strict_win_vs_pinsker": bool(policy.gap_cap < pinsker["gap_cap"] - 1e-12),
            "selected_covered": bool(selected.gap_cap + 1e-12 >= truth),
            "selected_strict_win_vs_policy": bool(selected.gap_cap < policy.gap_cap - 1e-12),
            "selected_strict_win_vs_joint_l1": bool(selected.gap_cap < direct.gap_cap - 1e-12),
        }
    return {"mechanism": name, "seed": seed, "n": n,
            "alpha_family": .05, "alpha_per_loss": alpha_loss,
            "losses": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--first-seed", type=int, default=130001)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists")
    with args.out.open("w") as handle:
        for name, joint in MECHANISMS:
            for seed in range(args.first_seed, args.first_seed + args.reps):
                row = run(name, joint, seed)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
            print("completed", name, flush=True)
