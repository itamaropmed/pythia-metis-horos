"""Simulated clinical text + simulated LLM readers with a known conditional law.

Design (conditional truth is exact):
  cell  ~ 20 table cells (age x procedure family proxy), baseline logit a_cell
  z     ~ 6 binary "note concepts" (some correlated with the cell): what the note actually says
  Y     ~ Bernoulli(sigmoid(a_cell + z @ beta + gamma * z0 * z1))
  p_i   = P(Y=1 | cell_i, z_i) is known -> true NLI(T|X) and every V(G) are exact.

A simulated LLM reader = (subset of concepts it asks about, per-concept error rate, hallucinated extras).
It sees z (the text) only, never Y -- except the deliberately broken 'leaky' reader.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

K = 2
BETA = np.array([1.2, 0.9, 0.7, 0.5, 0.0, 0.0])
GAMMA = 0.8


def simulate(n, rng, beta=BETA, gamma=GAMMA, n_cells=20):
    cell = rng.integers(0, n_cells, n)
    a = np.linspace(-1.5, 1.0, n_cells)[cell]
    prev = np.array([.30, .25, .40, .20, .35, .50])
    shift = 0.15 * (cell[:, None] / n_cells - .5)          # concepts correlate with the table
    z = (rng.random((n, 6)) < np.clip(prev + shift, .02, .98)).astype(int)
    logit = a + z @ beta + gamma * z[:, 0] * z[:, 1]
    p1 = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < p1).astype(int)
    P = np.column_stack([1 - p1, p1])
    return dict(cell=cell, z=z, y=y, P=P)


def read(d, rng, concepts, err=0.1, n_hallucinated=0, leak=0.0):
    """Simulated LLM reader output: noisy copies of the asked concepts + hallucinated junk features.
    leak > 0: with that probability a feature is overwritten by the outcome (post-outcome note leakage)."""
    z = d["z"][:, concepts]
    flip = rng.random(z.shape) < err
    r = np.where(flip, 1 - z, z)
    if n_hallucinated:
        r = np.column_stack([r, (rng.random((len(z), n_hallucinated)) < .3).astype(int)])
    if leak > 0:
        m = rng.random(len(z)) < leak
        r = r.copy(); r[m, 0] = d["y"][m]
    return r


def design_matrix(cell, r, n_cells=20, interactions=True):
    X = [np.eye(n_cells)[cell], r]
    if interactions and r.shape[1] >= 2:
        X.append(r[:, :1] * r[:, 1:2])
    return np.column_stack(X)


def crossfit_probs(d, r, folds, double_dip=False):
    """Held-out probabilities: model for fold f trained on the other folds (label-free on fold f).
    double_dip=True trains on the evaluation fold itself (invalid; used to show the failure)."""
    Q = np.zeros((len(d["y"]), K))
    X = design_matrix(d["cell"], r)
    for f in np.unique(folds):
        te = folds == f; tr = ~te if not double_dip else te
        m = LogisticRegression(C=1.0, max_iter=500).fit(X[tr], d["y"][tr])
        Q[te] = m.predict_proba(X[te])
    return Q
