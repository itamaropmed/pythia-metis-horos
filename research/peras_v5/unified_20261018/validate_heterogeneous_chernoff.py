"""Untouched 10,000-draw validation of heterogeneous-support Chernoff."""
from __future__ import annotations

import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import norm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT)
from pythia.core.concentration import heterogeneous_chernoff_lower_sum


def design(index):
    rng = np.random.default_rng(71000 + index)
    n = 12 + 3 * index
    caps = np.exp(rng.uniform(math.log(.025), math.log(1.8), n))
    probs = rng.beta(.5 + .15 * index, 1.1 + .08 * index, n)
    return caps, probs


def run_design(args):
    index, draws, alpha, seed = args
    caps, probs = design(index)
    truth = float(np.dot(caps, probs))
    rng = np.random.default_rng(seed)
    violations = wins = positives = 0
    ratios = []
    hoeffding_radius = math.sqrt(float(np.dot(caps, caps)) * math.log(1 / alpha) / 2)
    for _ in range(draws):
        x = caps * (rng.random(len(caps)) < probs)
        hc = heterogeneous_chernoff_lower_sum(x, caps, alpha)
        hf = max(float(x.sum()) - hoeffding_radius, 0.0)
        violations += hc > truth + 1e-8
        wins += hc > hf + 1e-10
        positives += hc > 0
        if hf > 0:
            ratios.append(hc / hf)
    return {
        "design": index, "draws": draws, "truth": truth,
        "violations": int(violations), "wins": int(wins),
        "positive": int(positives),
        "median_ratio_when_hoeffding_positive": (
            float(np.median(ratios)) if ratios else None),
    }


def wilson_lower(successes, total, confidence=.95):
    z = float(norm.ppf(.5 + confidence / 2))
    p = successes / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (centre - radius) / denominator


if __name__ == "__main__":
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    alpha = .05
    jobs = [(i, draws, alpha, 990001 + 100003 * i) for i in range(10)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(run_design, jobs))
    total = sum(r["draws"] for r in rows)
    violations = sum(r["violations"] for r in rows)
    print(json.dumps({
        "draws": total, "designs": 10, "alpha": alpha,
        "coverage": 1 - violations / total,
        "coverage_wilson_lower_95": wilson_lower(total - violations, total),
        "violations": violations,
        "strict_wins_vs_heterogeneous_hoeffding": sum(r["wins"] for r in rows),
        "per_design": rows,
    }, indent=2))
