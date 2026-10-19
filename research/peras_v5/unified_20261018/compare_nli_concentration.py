"""Frozen OOD comparison of valid NLI concentration routes.

The predictive score is cross-fitted and frozen before the held-out labels are
used by ``nli_cohort_bound``.  The known conditional law supplies the exact
finite-cohort NLI target.  This script compares inference only; it is not a
claim that Pythia trains the predictive model.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "peras_v5", "afi_20261017"))
sys.path.insert(0, os.path.join(ROOT, "research", "peras_v5", "nli_20261017"))

from generator import MECH, make
from generator_ood import FAMILIES, make_ood
from pythia.core.unified import nli_cohort_bound


def exact_nli(P, cells, rows):
    out = 0.0
    for c in np.unique(cells[rows]):
        idx = rows[cells[rows] == c]
        m = len(idx)
        if m >= 2:
            out += m / (m - 1) * ((P[idx] - P[idx].mean(0)) ** 2).sum()
    return float(out / len(rows))


OOD_BETA = {
    "O1_grammar_severity": .95, "O2_long_range_negation": .75,
    "O3_numbers_in_words": .85, "O4_copy_forward_template": .95,
    "O5_two_languages": .85, "O6_local_rare_token": 3.0,
}


def run(name, n, seed, suite="mechanisms"):
    d = (make(name, n, seed) if suite == "mechanisms"
         else make_ood(name, n, OOD_BETA[name], seed))
    y, cells, P, K = d["y"], d["cell"], d["P"], d["K"]
    X = TfidfVectorizer(
        token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, sublinear_tf=True
    ).fit_transform(d["notes"])
    X = hstack([csr_matrix(np.eye(cells.max() + 1)[cells]), X]).tocsr()
    fold = np.random.default_rng(seed + 77).permutation(n) % 2
    methods = ("hoeffding", "bentkus", "empirical_bernstein",
               "heterogeneous_chernoff", "stratified_bentkus", "safe_portfolio")
    bounds = {m: 0.0 for m in methods}
    truth = 0.0
    selected = []
    for f in (0, 1):
        te = np.flatnonzero(fold == f)
        tr = np.flatnonzero(fold != f)
        model = LogisticRegression(C=2, max_iter=3000).fit(X[tr], y[tr])
        Q = np.zeros((len(te), K))
        Q[:, model.classes_] = model.predict_proba(X[te])
        weight = len(te) / n
        truth += weight * exact_nli(P, cells, te)
        for method in methods:
            bound, info = nli_cohort_bound(
                cells[te], Q, y[te], .025, concentration=method
            )
            bounds[method] += weight * bound
            if method == "safe_portfolio":
                selected.append(info["selected_concentration"])
    best_fixed = max(("hoeffding", "bentkus", "empirical_bernstein",
                      "heterogeneous_chernoff", "stratified_bentkus"),
                     key=lambda m: bounds[m])
    return {
        "dataset": name, "n": n, "seed": seed, "truth": truth,
        "bounds": bounds, "best_fixed": best_fixed,
        "portfolio_selected": selected,
        "valid": {m: bounds[m] <= truth + 1e-12 for m in methods},
    }


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 95101
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    suite = sys.argv[4] if len(sys.argv) > 4 else "mechanisms"
    names = MECH if suite == "mechanisms" else FAMILIES
    rows = []
    for rep in range(reps):
        for j, name in enumerate(names):
            row = run(name, n, seed + 100_003 * rep + 1009 * j, suite)
            rows.append(row)
            if reps == 1:
                print(json.dumps(row), flush=True)
    if reps > 1:
        informative = [r for r in rows if max(
            r["bounds"]["heterogeneous_chernoff"], r["bounds"]["hoeffding"]
        ) > 0]
        wins = sum(r["bounds"]["heterogeneous_chernoff"]
                   > r["bounds"]["hoeffding"] + 1e-12 for r in informative)
        losses = sum(r["bounds"]["heterogeneous_chernoff"] + 1e-12
                     < r["bounds"]["hoeffding"] for r in informative)
        ratios = [r["bounds"]["heterogeneous_chernoff"]
                  / r["bounds"]["hoeffding"] for r in informative
                  if r["bounds"]["hoeffding"] > 0]
        print(json.dumps({
            "suite": suite, "n": n, "reps": reps, "datasets": len(rows),
            "informative": len(informative), "strict_wins": wins,
            "strict_losses": losses, "ties": len(informative) - wins - losses,
            "median_bound_ratio": float(np.median(ratios)),
            "all_valid": all(all(r["valid"].values()) for r in rows),
            "violations_by_method": {
                m: sum(not r["valid"][m] for r in rows)
                for m in rows[0]["valid"]
            },
        }, indent=2))
