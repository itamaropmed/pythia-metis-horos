"""Theorem D on tabular data: certify the FINE representation S5 (Age in 20 bins, mostly singleton cells) through
the coarse quotient S4 (Age in 5 bins) with a declared within-cell TV slack delta.  Pair route pays delta^2,
min-count pays 2*delta.  True max TV between S5 laws inside an S4 cell is computed on the 1M pool."""
import sys, os, json, runpy
import numpy as np
L = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tabular_replicate_ladder.py"), run_name="lib")
from peras.core.replicate_pair_certificates import pair_certificate, mincount_certificate, chunk_cells
D, LAWS, LADDER = L["D"], L["LAWS"], L["LADDER"]
res = []
for K in (2, 3):
    P = LAWS[K]
    k4 = D.groupby(LADDER["S4"], sort=False).ngroup().to_numpy(); k5 = D.groupby(LADDER["S5"], sort=False).ngroup().to_numpy()
    p5 = np.zeros((k5.max() + 1, K)); np.add.at(p5, k5, P); n5 = np.bincount(k5); p5 /= n5[:, None]
    par = np.zeros(k5.max() + 1, int); par[k5] = k4
    # true delta: max over S4 cells of max pairwise TV between member S5 laws (cells with >= 50 pool rows)
    tv = 0.0; tvq = []
    for c in np.unique(par):
        mem = np.flatnonzero((par == c) & (n5 >= 50))
        if len(mem) > 1:
            Q = p5[mem]; t = .5 * np.abs(Q[:, None, :] - Q[None, :, :]).sum(2).max(); tvq.append(t); tv = max(tv, t)
    for rep in range(6):
        r = np.random.default_rng(7000 + rep + 10 * K); idx = r.integers(len(D), size=20000)
        y = np.array([r.choice(K, p=p) for p in P[idx]])
        gold5 = float((1 - p5[k5[idx]].max(1)).mean())
        c4 = k4[idx]
        out = dict(K=K, rep=rep, gold_S5=gold5, true_delta_max=tv, true_delta_median=float(np.median(tvq)),
                   direct_S5_mincount=mincount_certificate(k5[idx], y, K, .05)["bound"])
        for dl in (0.05, 0.1, 0.2, 0.3):
            out[f"pair_via_S4_delta{dl}"] = pair_certificate(chunk_cells(c4, 8, rep), y, K, .025, slack=dl)["bound"]
            out[f"mincount_via_S4_delta{dl}"] = mincount_certificate(c4, y, K, .025, slack=dl)["bound"]
        res.append(out); print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}), flush=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tabular_quotient_slack.json"), "w"), indent=1)
