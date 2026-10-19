"""Ten-law audit of finite-alphabet CMI upper confidence edges.

No model is trained.  Every law fixes a complete finite alphabet and an exact
joint distribution for coarse group G, pre-outcome refinement Z, and outcome
Y.  The audit checks CMI upper coverage and the induced bounded 0-1 Bayes-risk
gap cap at three sample sizes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from peras import (
    bounded_loss_gap_from_cmi,
    finite_cmi_upper_bound,
    finite_decision_gap_upper_bound,
)


def mechanism(name, pz, py):
    pz = np.asarray(pz, float)          # g,z
    py = np.asarray(py, float)          # g,z,y
    pg = np.asarray([.55, .45])
    joint = np.zeros((3, 2, 2))         # y,z,g
    for g in range(2):
        for z in range(2):
            joint[:, z, g] = pg[g] * pz[g, z] * py[g, z]
    return name, joint


BASE0 = np.array([.70, .20, .10])
BASE1 = np.array([.15, .30, .55])

MECHANISMS = (
    mechanism("conditional_independent", [[.5, .5], [.5, .5]],
              [[BASE0, BASE0], [BASE1, BASE1]]),
    mechanism("weak_refinement", [[.5, .5], [.5, .5]],
              [[[.74, .17, .09], [.66, .23, .11]],
               [[.13, .29, .58], [.17, .31, .52]]]),
    mechanism("moderate_refinement", [[.5, .5], [.5, .5]],
              [[[.82, .12, .06], [.58, .28, .14]],
               [[.08, .24, .68], [.22, .36, .42]]]),
    mechanism("strong_refinement", [[.5, .5], [.5, .5]],
              [[[.94, .04, .02], [.46, .36, .18]],
               [[.03, .12, .85], [.27, .48, .25]]]),
    mechanism("rare_refinement", [[.93, .07], [.90, .10]],
              [[[.70, .20, .10], [.05, .10, .85]],
               [[.15, .30, .55], [.85, .10, .05]]]),
    mechanism("asymmetric_groups", [[.8, .2], [.25, .75]],
              [[[.78, .16, .06], [.38, .42, .20]],
               [[.04, .20, .76], [.30, .40, .30]]]),
    mechanism("group_explains_all", [[.1, .9], [.9, .1]],
              [[BASE0, BASE0], [BASE1, BASE1]]),
    mechanism("label_switch", [[.5, .5], [.5, .5]],
              [[[.90, .08, .02], [.05, .90, .05]],
               [[.05, .15, .80], [.80, .15, .05]]]),
    mechanism("noisy_target_proxy", [[.5, .5], [.5, .5]],
              [[[.88, .10, .02], [.10, .82, .08]],
               [[.08, .20, .72], [.72, .20, .08]]]),
    mechanism("near_disclosure", [[.5, .5], [.5, .5]],
              [[[.98, .01, .01], [.01, .98, .01]],
               [[.01, .01, .98], [.98, .01, .01]]]),
)


def entropy(p):
    positive = np.asarray(p)[np.asarray(p) > 0]
    return -float(np.sum(positive * np.log(positive)))


def truth(joint):
    pyg, pzg, pg = joint.sum(1), joint.sum(0), joint.sum((0, 1))
    cmi = entropy(pyg) + entropy(pzg) - entropy(pg) - entropy(joint)
    coarse = sum(pg[g] - np.max(pyg[:, g]) for g in range(2))
    fine = sum(np.sum(joint[:, z, g]) - np.max(joint[:, z, g])
               for z in range(2) for g in range(2))
    return cmi, float(coarse), float(fine), float(coarse - fine)


def run(name, joint, n, seed):
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, joint.ravel())
    flat = np.repeat(np.arange(12), counts)
    y, z, g = flat // 4, (flat // 2) % 2, flat % 2
    result = finite_cmi_upper_bound(
        y, z, g, y_alphabet=(0, 1, 2), z_alphabet=(0, 1),
        g_alphabet=(0, 1), alpha=.05,
        alphabet_frozen_before_evaluation=True,
        provenance={"registry": "finite-cmi-10-law-v1", "mechanism": name})
    gap = bounded_loss_gap_from_cmi(
        result, loss="zero_one", loss_range=1., maximum_bayes_risk=2 / 3)
    direct = finite_decision_gap_upper_bound(
        y, z, g, y_alphabet=(0, 1, 2), z_alphabet=(0, 1),
        g_alphabet=(0, 1), loss="zero_one",
        loss_matrix=np.ones((3, 3)) - np.eye(3), alpha=.05,
        alphabet_frozen_before_evaluation=True,
        provenance={"registry": "finite-cmi-10-law-v1", "mechanism": name},
        maximum_bayes_risk=2 / 3)
    true_cmi, coarse, fine, true_gap = truth(joint)
    return {
        "mechanism": name, "n": n, "seed": seed,
        "true_cmi": true_cmi, "plugin_cmi": result.plugin_cmi,
        "upper_cmi": result.upper_cmi,
        "deterministic_cmi_cap": float(np.log(2.)),
        "true_coarse_bayes_error": coarse,
        "true_fine_bayes_error": fine,
        "true_information_gap": true_gap,
        "pinsker_gap_cap": gap["gap_cap"],
        "direct_decision_gap_cap": direct.gap_cap,
        "plugin_decision_gap": direct.plugin_gap,
        "cmi_covered": bool(result.upper_cmi + 1e-12 >= true_cmi),
        "gap_covered": bool(gap["gap_cap"] + 1e-12 >= true_gap),
        "direct_gap_covered": bool(direct.gap_cap + 1e-12 >= true_gap),
        "direct_strictly_tighter": bool(direct.gap_cap < gap["gap_cap"] - 1e-12),
        "nontrivial_vs_alphabet_cap": bool(result.upper_cmi < np.log(2.) - 1e-12),
        "tv_radius": result.tv_radius,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--first-seed", type=int, default=128001)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists")
    with args.out.open("w") as handle:
        for n in (3_000, 30_000, 300_000):
            for name, joint in MECHANISMS:
                for seed in range(args.first_seed, args.first_seed + args.reps):
                    row = run(name, joint, n, seed)
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
            print("completed", n, flush=True)
