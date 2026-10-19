"""Post-hoc numerical oracle check, never used to construct the bound."""
from __future__ import annotations

import argparse
import json

import numpy as np

import gold
from run_direct_w1_ood import CLIP
from run_exact_law_battery import make


def run(path: str) -> dict:
    grid = np.linspace(*CLIP, 12_001)
    cdf_cache = {}
    distance_cache = {}
    gaps = []
    rows = [json.loads(line) for line in open(path)]
    for row in rows:
        design, seed = row["design"], row["seed"]
        d = make(design, row["n_audit"], seed)
        labels = [(int(d["code"][i]), int(d["concepts"][i, 0]))
                  for i in range(row["n_audit"])]
        for i, label in enumerate(labels):
            key = (design, label)
            if key not in cdf_cache:
                cdf_cache[key] = gold.FAMILIES[d["fam"][i]](
                    (grid - d["mu"][i]) / d["s"][i])
        order = np.lexsort((np.arange(row["n_audit"]), d["code"],
                            d["concepts"][:, 0]))
        blocks = order.reshape(-1, 2)
        total = 0.0
        for i, j in blocks:
            a, b = labels[i], labels[j]
            if a == b:
                continue
            key = (design, frozenset((a, b)))
            if key not in distance_cache:
                distance_cache[key] = float(np.trapezoid(
                    np.abs(cdf_cache[(design, a)] - cdf_cache[(design, b)]), grid))
            # Each mixed pair contributes two ordered distances among four.
            total += distance_cache[key] / 2
        true_mean = total / len(blocks)
        gaps.append(row["direct_cap"] - true_mean)
    result = {"n_runs": len(rows), "quadrature_points": len(grid),
              "direct_cap_population_misses": sum(gap < -1e-5 for gap in gaps),
              "minimum_cap_minus_numeric_population_w1": float(min(gaps)),
              "median_cap_minus_numeric_population_w1": float(np.median(gaps))}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    run(ap.parse_args().path)
