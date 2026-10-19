"""Design-level audit of the untouched OOD suite."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument("--input", type=Path, default=Path(__file__).with_name("ood_results.jsonl"))
args = ap.parse_args()
PATH = args.input
rows = [json.loads(line) for line in open(PATH)]
by_design = defaultdict(list)
for row in rows:
    by_design[row["design"]].append(row)

losses = ("MSE", "MAE", "Q90", "W30", "W60")
est_methods = ("knn20", "rice", "gamma", "gbm_oof", "learning_curve")


def value(row, method, loss):
    if method == "knn20":
        return row["knn"]["20"].get(loss)
    return row[method].get(loss)


def rel_error(v, gold):
    return abs(v - gold) / gold


for loss in losses:
    design_rows = []
    for design, rr in sorted(by_design.items()):
        peras_error = float(np.median([rel_error(r["peras"][loss], r["gold"][loss]) for r in rr]))
        rivals = {}
        for method in est_methods:
            v = [rel_error(value(r, method, loss), r["gold"][loss]) for r in rr
                 if value(r, method, loss) is not None]
            if v:
                rivals[method] = float(np.median(v))
        best_name = min(rivals, key=rivals.get)
        best_error = rivals[best_name]
        gold = float(np.median([r["gold"][loss] for r in rr]))
        est = float(np.median([r["peras"][loss] for r in rr]))
        bound = float(np.median([r["bound"][loss] for r in rr]))
        violations = sum(r["bound"][loss] > r["gold"][loss] for r in rr)
        design_rows.append(dict(design=design, name=rr[0]["name"], gold=gold,
                                peras=est, peras_rel_error=peras_error,
                                best_rival=best_name, best_rival_rel_error=best_error,
                                bound=bound, violations=violations,
                                nonpositive=sum(r["bound"][loss] <= 0 for r in rr)))
    wins = sum(x["peras_rel_error"] < x["best_rival_rel_error"] for x in design_rows)
    log_ratios = np.array([np.log(x["best_rival_rel_error"] / x["peras_rel_error"])
                           for x in design_rows])
    print(json.dumps(dict(loss=loss, wins=wins, designs=len(design_rows),
                          sign_p=binomtest(wins, len(design_rows), 0.5, alternative="greater").pvalue,
                          wilcoxon_p=wilcoxon(log_ratios, alternative="greater").pvalue,
                          median_peras_rel_error=float(np.median([x["peras_rel_error"] for x in design_rows])),
                          median_best_rival_rel_error=float(np.median([x["best_rival_rel_error"] for x in design_rows])),
                          bound_violations=sum(x["violations"] for x in design_rows),
                          bound_checks=len(rows),
                          bound_nonpositive=sum(x["nonpositive"] for x in design_rows),
                          rows=design_rows)))
