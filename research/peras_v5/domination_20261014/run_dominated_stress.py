"""Validity stress: small samples, many cells, unequal clusters, shared within-cluster effects."""
import sys, json, math
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "research/peras_v5/interventions_20261014")
import run_support_free_cluster_block_confirmation as R
from peras.core.dominated_saa import dominated_saa_lower_bound
reps = int(sys.argv[1]); base = int(sys.argv[2]); out = sys.argv[3]
rows = []
for mi, (name, gen, _) in enumerate(R.LAWS):
    # marginal law of a case: 0.6*u + 0.8*e, u ~ same law shared within cluster
    r0 = np.random.default_rng(91); M = 2_000_000
    z = 0.6 * gen(r0, M) + 0.8 * gen(r0, M)
    med = np.median(z); unit_mae = float(np.mean(np.abs(z - med)))
    a9 = np.quantile(z, .9); d = z - a9; unit_q90 = float(np.mean(np.where(d >= 0, .9*d, -.1*d)))
    zs = np.sort(z)
    _memo = {}
    def wf(s, w=1.0):
        if s in _memo: return _memo[s]
        e = zs * s; right = np.searchsorted(e, e + 2*w, side="right"); _memo[s] = 1 - np.max(right - np.arange(len(e))) / len(e); return _memo[s]
    for rep in range(reps):
        seed = base + 100 * mi + rep; rng = np.random.default_rng(seed)
        G = int(rng.choice([60, 120, 250])); C = int(rng.choice([5, 20, 40]))
        sizes = rng.integers(1, 7, G); cc_cl = rng.integers(0, C, G)
        u = gen(rng, G)
        lab, y, cl = [], [], []
        for g in range(G):
            # cells may differ inside a cluster (case-level label from cluster cell + offset)
            for j in range(sizes[g]):
                c = (cc_cl[g] + (j % 2)) % C
                s = .5 + c / C
                lab.append(c); cl.append(g); y.append(3. * c + s * (0.6 * u[g] + 0.8 * gen(rng, 1)[0]))
        lab, y, cl = np.array(lab), np.array(y), np.array(cl)
        sc = .5 + lab / C
        WF = {c: wf(.5 + c / C) for c in np.unique(lab)}
        m = 3
        keep_cases = None
        for loss, L, kw, gold_case in (("mae", "mae", {}, unit_mae * sc), ("q90", "pinball", {"quantile": .9}, unit_q90 * sc),
                                      ("w1", "window_miss", {"window": 1.0}, np.array([WF[c] for c in lab]))):
            for mm in (None, m):
                res = dominated_saa_lower_bound(lab, y, cl, loss=L, caps=R.CAPS, alpha=.05, max_cases_per_cluster=mm,
                        representation_provenance="stress", independent_clusters_attested=True,
                        outcome_blind_labels_attested=True, **kw)
                gold = float(np.mean(gold_case))
                rows.append(dict(mech=name, seed=seed, G=G, C=C, n=len(y), loss=loss, m=mm, gold=gold,
                                 bound=res.bound, ratio=res.bound / gold, viol=bool(res.bound > gold * (1 + 1e-3))))
open(out, "w").write("".join(json.dumps(r) + "\n" for r in rows))
print("rows", len(rows), "viol", sum(r["viol"] for r in rows))
