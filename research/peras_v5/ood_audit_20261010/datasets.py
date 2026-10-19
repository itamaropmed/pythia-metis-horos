"""Ten known-noise designs: tabular features + a free-text pre-procedure note.

Every design specifies the full conditional law  Y|x = mu(x) + s(x) Z_{f(x)},
so the per-case Bayes risk of every loss is known exactly (gold.pointwise).
Outcomes are in minutes so W30 / W60 are meaningful.  Clinical concepts that
drive the outcome are written into the note with synonyms, negation and
nuisance sentences; some designs put signal only in the text.
"""
from __future__ import annotations
import numpy as np

COMPLEX = {
    0: ["Routine procedure planned.", "Straightforward case expected.",
        "Standard ablation with uncomplicated anatomy.", "Simple anatomy on prior imaging."],
    1: ["Moderately complex case.", "Some anatomical complexity anticipated.",
        "Intermediate difficulty expected per imaging."],
    2: ["Highly complex substrate expected.", "Extensive substrate mapping anticipated.",
        "Complex case with challenging anatomy.", "Difficult substrate, prolonged mapping likely."],
}
REDO = {
    1: ["Prior ablation in {y}.", "Status post previous ablation.", "Redo procedure after recurrence.",
        "Underwent ablation {y}, now recurrent."],
    0: ["No prior ablation.", "Denies previous ablation.", "First-time procedure.",
        "Ablation history: none."],
}
RISK = {
    1: ["History of difficult vascular access.", "Known IVC filter, access may be difficult.",
        "Prior groin complication noted.", "Venous occlusion on prior study."],
    0: ["Access expected to be unremarkable.", "No history of difficult access.",
        "Vascular access uncomplicated previously."],
}
NUISANCE = [
    "Patient lives with spouse.", "Allergies: penicillin (rash).", "Takes metoprolol daily.",
    "Vitals stable at clinic visit.", "BP {a}/{b} at intake.", "Heart rate {a} bpm.",
    "Non-smoker.", "Former smoker, quit {y}.", "Weight {a} kg.", "Creatinine {c} mg/dL.",
    "Patient counselled on risks and benefits.", "Consent signed.", "NPO after midnight.",
    "Hold anticoagulation per protocol.", "Echo reviewed with patient.", "Retired teacher.",
    "Enjoys gardening.", "Family history of hypertension.", "Hemoglobin {c} g/dL.",
    "Discussed sedation plan.", "Follow-up with primary care.", "No known drug allergies.",
]


def _note(rng, c, r, k):
    s = [rng.choice(COMPLEX[c]), rng.choice(REDO[r]), rng.choice(RISK[k])]
    s += list(rng.choice(NUISANCE, size=rng.integers(3, 7), replace=False))
    rng.shuffle(s)
    return " ".join(x.format(y=rng.integers(2005, 2024), a=rng.integers(50, 180),
                             b=rng.integers(50, 100), c=round(rng.uniform(0.6, 15), 1))
                    for x in s)


def _base(n, rng, p=6):
    c = rng.choice(3, n, p=[0.45, 0.35, 0.20])
    r = (rng.random(n) < 0.3).astype(int)
    k = (rng.random(n) < 0.25).astype(int)
    booked = np.choose(np.clip(c + rng.integers(-1, 2, n), 0, 2),
                       [np.full(n, 90.0), np.full(n, 150.0), np.full(n, 240.0)])
    booked += rng.choice([-30.0, 0.0, 30.0], n)
    X = rng.standard_normal((n, p))
    return c, r, k, booked, X


def make(design: int, n: int, seed: int):
    rng = np.random.default_rng(seed)
    p = 30 if design == 5 else 6
    c, r, k, b, X = _base(n, rng, p)
    fam = np.full(n, "normal", dtype=object)
    extra = None

    if design == 1:      # smooth Gaussian, text irrelevant
        mu = b + 15 * np.sin(X[:, 0]) + 8 * X[:, 1] ** 2
        s = np.full(n, 20.0)
    elif design == 2:    # heteroscedastic skewed
        mu = b + 12 * X[:, 0] + 8 * X[:, 1] * X[:, 2]
        s = 12 + 18 / (1 + np.exp(-2 * X[:, 3])); fam[:] = "lognorm"; s = s * 1.8
    elif design == 3:    # Laplace  (pair-twin of design 4)
        mu = b + 25 * c + 10 * X[:, 0]
        s = np.full(n, 14.0); fam[:] = "laplace"
    elif design == 4:    # Gamma(2): identical pair-difference law to design 3
        mu = b + 25 * c + 10 * X[:, 0]
        s = np.full(n, 14.0); fam[:] = "gamma2"
    elif design == 5:    # sparse 30-D, 3 signal coords + 1 scale coord
        mu = b + 20 * np.tanh(X[:, 0]) + 15 * X[:, 1] + 10 * X[:, 2] * X[:, 0]
        s = 15 + 10 * (X[:, 3] > 0)
    elif design == 6:    # signal only in text, t5 noise
        mu = b + 30 * c + 20 * r
        s = np.full(n, 16.0); fam[:] = "t5"
    elif design == 7:    # text-driven heteroscedasticity
        mu = b + 20 * c
        s = 12 * (1 + 1.5 * r); fam[:] = "laplace"
    elif design == 8:    # text-driven complication mixture (W30-sensitive)
        mu = b + 15 * c + 8 * X[:, 0]
        s = np.full(n, 15.0); fam[:] = np.where(k == 1, "mix35", "mix05")
    elif design == 9:    # heavy-tailed nonlinear
        mu = b + 20 * np.sin(X[:, 0] * X[:, 1]) + 10 * X[:, 2]
        s = np.full(n, 14.0); fam[:] = "t3"
    elif design == 10:   # discrete states with genuine replicates
        code = rng.integers(0, 40, n)
        prng = np.random.default_rng(12345)
        mu_c = prng.uniform(60, 260, 40); s_c = prng.uniform(10, 30, 40)
        g_c = prng.random(40) < 0.5
        mu = mu_c[code] + 20 * c
        s = s_c[code]; fam[:] = np.where(g_c[code], "gamma2", "normal")
        b = np.round(mu_c[code] / 30) * 30
        X = np.eye(40)[code]
        extra = code
    else:
        raise ValueError(design)

    z = np.empty(n)
    for f in np.unique(fam):
        m = fam == f
        from gold import sample
        z[m] = sample(f, int(m.sum()), rng)
    y = mu + s * z
    notes = [_note(rng, c[i], r[i], k[i]) for i in range(n)]
    tab = np.column_stack([b, X])
    return dict(tab=tab, notes=notes, y=y, fam=fam.astype(str), s=s, mu=mu,
                concepts=np.column_stack([c, r, k]), code=extra)


NAMES = {1: "smooth-gauss", 2: "hetero-lognorm", 3: "laplace-twin", 4: "gamma-twin",
         5: "sparse-30D", 6: "text-signal-t5", 7: "text-hetero", 8: "text-mixture",
         9: "heavy-t3", 10: "discrete-states"}
