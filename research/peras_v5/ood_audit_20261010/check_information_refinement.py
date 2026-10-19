"""Property checks for the conditional-information refinement theorem."""
import json

import numpy as np

from finite_group_multiloss_interval import LOSSES
from refinement_gap_profile import (
    information_refinement_profile,
    registered_conditional_entropy,
)


Y = np.array([0., 20., 70., 150., 300.])


def bayes_risk(prob, loss):
    prob = np.asarray(prob, float)
    prob = prob / prob.sum()
    if loss == "MSE":
        action = float(prob @ Y)
        return float(prob @ ((Y - action) ** 2))
    if loss == "MAE":
        candidates = Y
        values = [prob @ np.abs(Y - a) for a in candidates]
    elif loss.startswith("Q"):
        q = float(loss[1:]) / 100
        values = [prob @ np.where(Y >= a, q * (Y - a),
                                      (1 - q) * (a - Y)) for a in Y]
    elif loss.startswith("W"):
        h = float(loss[1:])
        candidates = np.unique(np.concatenate((Y, Y - h, Y + h)))
        values = [prob @ (np.abs(Y - a) > h) for a in candidates]
    else:
        raise ValueError(loss)
    return float(min(values))


def exact_risks(joint, loss):
    # joint[g,z,y]
    coarse = 0.
    fine = 0.
    for g in range(joint.shape[0]):
        pg = joint[g].sum()
        if pg:
            coarse += pg * bayes_risk(joint[g].sum(axis=0) / pg, loss)
        for z in range(joint.shape[1]):
            pgz = joint[g, z].sum()
            if pgz:
                fine += pgz * bayes_risk(joint[g, z] / pgz, loss)
    return float(coarse), float(fine)


def exact_entropy(joint):
    entropy = 0.
    for g in range(joint.shape[0]):
        pg = joint[g].sum()
        for z in range(joint.shape[1]):
            pgz = joint[g, z].sum()
            if pgz:
                entropy -= pgz * np.log(pgz / pg)
    return float(entropy)


def run(reps=1_000):
    rng = np.random.default_rng(994_311)
    violations = 0
    gap_cap_violations = 0
    cases = 0
    for _ in range(reps):
        joint = rng.dirichlet(np.full(2 * 4 * len(Y), .35)).reshape(2, 4, len(Y))
        entropy = exact_entropy(joint)
        for loss in LOSSES:
            coarse, fine = exact_risks(joint, loss)
            result = {"loss": loss, "outcome_clip": [0., 300.], "lower": coarse}
            profile = information_refinement_profile(
                result, [entropy], representation_provenance="known finite joint law",
                pre_outcome_registered=True)
            bound = profile["bounds"][0]
            cap = profile["gap_caps"][0]
            violations += bound > fine + 1e-10
            gap_cap_violations += coarse - fine > cap + 1e-10
            cases += 1

    coarse = [0, 0, 0, 1, 1]
    refined = ["a", "a", "b", "c", "d"]
    weights = [.1, .2, .2, .25, .25]
    entropy = registered_conditional_entropy(coarse, refined, weights)
    expected = -(.3 * np.log(.3 / .5) + .2 * np.log(.2 / .5)
                 + .25 * np.log(.25 / .5) + .25 * np.log(.25 / .5))
    assert abs(entropy - expected) < 1e-12
    assert registered_conditional_entropy(coarse, coarse) == 0

    refused = information_refinement_profile(
        {"loss": "MAE", "outcome_clip": [0., 300.], "lower": 10.}, [.1],
        representation_provenance="outcome copied into a note",
        pre_outcome_registered=False)
    assert refused["status"] == "refused" and refused["bound"] is None
    window_t1 = information_refinement_profile(
        {"loss": "W30", "outcome_clip": [0., 300.], "lower": .1}, [.1],
        representation_provenance="registered finite concept",
        pre_outcome_registered=True, t1_constant=10.)
    assert window_t1["status"] == "refused"
    assert violations == 0 and gap_cap_violations == 0
    out = {"random_joint_laws": reps, "loss_cases": cases,
           "richer_gold_exceedances": violations,
           "information_gap_cap_exceedances": gap_cap_violations,
           "entropy_identity_checks": 2,
           "outcome_derived_refusal": True,
           "window_T1_refusal": True}
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
