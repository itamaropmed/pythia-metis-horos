"""Frozen numeric and pre-outcome text-like duration designs with known risk."""
from __future__ import annotations

import numpy as np


MULTILOSS_DESIGNS = (
    "numeric_uniform", "numeric_heteroskedastic", "numeric_nonlinear",
    "numeric_sparse4", "numeric_extreme_scale",
    "note_shift", "note_scale", "note_negation", "note_community",
    "note_rare",
)


def extract_note_concept(note: str, design: str) -> float:
    """Frozen literal extractor; no fit, prompt, outcome, or post-cutoff text."""
    if design == "note_community":
        for value, token in enumerate(("community alpha", "community beta", "community gamma")):
            if token in note:
                return value / 2
        raise ValueError("missing community token")
    if "no prior implant" in note or "primary planned" in note:
        return 0.
    if "prior implant present" in note or "revision planned" in note:
        return 1.
    raise ValueError("missing or ambiguous preoperative concept")


def multiloss_design(rng: np.random.Generator, name: str, n: int) -> dict:
    """Return fixed metric, outcomes, analytic Bayes risks, and note provenance.

    Noise is Uniform(-s(x), s(x)). For pinball tau, the Bayes action is
    center + (2*tau-1)*s and risk is tau*(1-tau)*s. For a width-w absolute
    window, the optimal miss risk is max(0, 1-w/s); center is a valid selected
    optimal action and its stated Lipschitz constant uses mean L1 distance.
    """
    if name not in MULTILOSS_DESIGNS or n < 4:
        raise ValueError("registered design and at least four entities required")
    x = rng.uniform(size=n)
    notes = None
    if name == "numeric_uniform":
        center, scale, cap = 100 + 60 * x, np.full(n, 40.), 60.
    elif name == "numeric_heteroskedastic":
        center, scale, cap = 100 + 50 * x, 20 + 30 * x, 75.
    elif name == "numeric_nonlinear":
        center, scale, cap = 100 + 30 * np.abs(x - .5), np.full(n, 35.), 30.
    elif name == "numeric_sparse4":
        center, scale, cap = 100 + 70 * x, 25 + 15 * x, 330.
    elif name == "numeric_extreme_scale":
        center, scale, cap = 100 + 25 * x, 5 + 55 * x, 70.
    else:
        z = (
            (rng.uniform(size=n) < .1).astype(int)
            if name == "note_rare" else
            rng.integers(0, 3 if name == "note_community" else 2, size=n)
        )
        if name == "note_shift":
            center, scale, cap = 100 + 60 * x + 30 * z, np.full(n, 40.), 120.
            notes = [
                "preop note: revision planned; elective assessment."
                if value else "preop note: primary planned; elective assessment."
                for value in z
            ]
        elif name == "note_scale":
            center, scale, cap = 100 + 60 * x + 10 * z, 20 + 30 * z, 120.
            notes = [
                "preop note: revision planned; expected complexity reviewed."
                if value else "preop note: primary planned; expected complexity reviewed."
                for value in z
            ]
        elif name == "note_negation":
            center, scale, cap = 100 + 40 * x + 20 * z, 25 + 20 * z, 80.
            notes = [
                "preop history: prior implant present; approach discussed."
                if value else "preop history: no prior implant; approach discussed."
                for value in z
            ]
        elif name == "note_rare":
            center, scale, cap = 100 + 50 * x + 70 * z, 25 + 30 * z, 200.
            notes = [
                "preop note: revision planned; rare complexity marker present."
                if value else "preop note: primary planned; routine pathway."
                for value in z
            ]
        else:
            center, scale, cap = 100 + 50 * x + 15 * z, 30 + 10 * z, 100.
            tokens = ("community alpha", "community beta", "community gamma")
            notes = [f"preop referral: {tokens[value]}; procedure scheduled."
                     for value in z]
        extracted = np.asarray([extract_note_concept(note, name) for note in notes])
        expected = z / 2 if name == "note_community" else z.astype(float)
        if not np.array_equal(extracted, expected):
            raise AssertionError("frozen note extractor changed concept meaning")
    representation = (
        np.column_stack((x, extracted)) if notes is not None else
        np.column_stack((x, rng.uniform(size=(n, 3))))
        if name == "numeric_sparse4" else x[:, None]
    )
    y = center + rng.uniform(-scale, scale)
    return {
        "representation": representation,
        "outcomes": y,
        "preoutcome_notes": notes,
        "center": center,
        "noise_scale": scale,
        "lipschitz_cap": cap,
        "truth_q10_pinball": .1 * .9 * scale,
        "truth_q90_pinball": .9 * .1 * scale,
        **{
            f"truth_w{width}_miss": np.maximum(0., 1. - width / scale)
            for width in (15, 30, 60)
        },
        **{
            f"truth_w{width}_percentage": 100 * np.minimum(1., width / scale)
            for width in (15, 30, 60)
        },
        "noise_known": True,
        "notes_preoutcome": notes is not None,
        "pythia_trained_predictive_model": False,
    }
