"""Real clinical tabular (PMLB Bayesian-network samples, iid by construction): finite-cell Peras floors over
an information ladder, vs held-out model risk (sandwich) and certified information value."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from peras.core.dominated_saa import dominated_saa_lower_bound
from peras.core.truncated_saa import dkelly_lower_mean
from sklearn.ensemble import HistGradientBoostingRegressor
out = {}
for nm, n in (("1201_BNG_breastTumor", 20000), ("1196_BNG_pharynx", 20000), ("1193_BNG_lowbwt", 20000)):
    d = pd.read_csv(__import__("os").environ.get("PERAS_DATA", __import__("os").environ.get("PERAS_DATA", "/home/claude/data")) + f"/{nm}.tsv.gz", sep="\t").sample(n, random_state=0).reset_index(drop=True)
    y = d.pop("target").to_numpy(float); cols = list(d.columns)
    A = np.arange(n) < n // 2; B = ~A                       # A: train model; B: certify + evaluate
    # information ladder: nested feature prefixes ordered by an outcome-blind rule (column order) + full
    ladder = {f"{k}_features": cols[:k] for k in (1, 2, 4, len(cols))}
    caps = tuple(np.quantile(np.abs(y - np.median(y)), [.5, .9, .99]) * np.array([1, 1, 1]))
    res = {"n_cert": int(B.sum()), "target_sd": float(y.std()), "n_features": len(cols), "nodes": {}}
    for name, cs in ladder.items():
        lab = d.loc[B, cs].astype(str).agg("|".join, axis=1).to_numpy()
        r = dominated_saa_lower_bound(lab, y[B], np.arange(B.sum()), loss="mae", caps=caps, alpha=.05 / 5,
                                      inequalities=("bentkus",), representation_provenance=name,
                                      independent_clusters_attested=True, outcome_blind_labels_attested=True)
        res["nodes"][name] = {"cells": int(len(set(lab))), "lower": r.bound,
                              "insample_saa_untruncated": float(np.mean([np.abs(v - np.median(v)).mean() * len(v)
                                  for v in pd.Series(y[B]).groupby(lab).apply(np.asarray)]) * 0 +
                                  sum(np.abs(v - np.median(v)).sum() for v in pd.Series(y[B]).groupby(lab).apply(np.asarray)) / B.sum())}
    m = HistGradientBoostingRegressor(loss="absolute_error", max_iter=400, random_state=0).fit(d.loc[A].to_numpy(float), y[A])
    loss = np.abs(y[B] - m.predict(d.loc[B].to_numpy(float)))
    cap = float(np.abs(y - np.median(y)).max() * 2)            # working outcome-range cap (exploratory)
    from peras.core.truncated_saa import dkelly_lower_mean as _d
    upper = cap - _d(cap - np.minimum(loss, cap), cap=cap, alpha=.05 / 5)   # dKelly upper endpoint via reflection
    res["model_heldout_mae"] = float(loss.mean()); res["model_upper"] = float(upper)
    full = res["nodes"][f"{len(cols)}_features"]["lower"]
    res["headroom_certified_max"] = float(res["model_upper"] - full)
    res["value_beyond"] = {k: max(0.0, v["lower"] - res["model_upper"]) for k, v in res["nodes"].items()}
    out[nm] = res
    print(nm, json.dumps(res, indent=1), flush=True)
json.dump(out, open("tabular_bng_results.json", "w"), indent=1)
