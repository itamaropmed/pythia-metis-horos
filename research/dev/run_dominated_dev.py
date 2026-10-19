import sys, json, math, time
import numpy as np
sys.path.insert(0, "research/peras_v5/interventions_20261014"); sys.path.insert(0, ".")
import run_support_free_cluster_block_confirmation as R
from peras import clustered_block_saa_portfolio, truncated_saa_portfolio
from peras.core.dominated_saa import dominated_saa_lower_bound
from scipy.stats import norm
seeds_per = int(sys.argv[1]); base = int(sys.argv[2]); out = sys.argv[3]
rows = []
t = time.time()
MC = {}
def mc_floor(gen, loss):  # base-noise floor for q90 / window via large MC
    key = (gen.__name__ if hasattr(gen,'__name__') else id(gen), loss)
    r = np.random.default_rng(77); e = gen(r, 2_000_000)
    if loss == "q90":
        a = np.quantile(e, .9); d = e - a; return float(np.mean(np.where(d >= 0, .9*d, -.1*d)))
    return None
for mi, (name, gen, base_floor) in enumerate(R.LAWS):
    eq = np.sort(gen(np.random.default_rng(78), 2_000_000))
    a9 = np.quantile(eq, .9); d = eq - a9
    q90_unit = float(np.mean(np.where(d >= 0, .9*d, -.1*d)))
    for rep in range(seeds_per):
        seed = base + 50 * mi + rep
        rng = np.random.default_rng(seed)
        G = 4096
        cc = rng.integers(0, 8, G); scale = .7 + .12 * cc; loc = 5. * cc
        yo = loc + scale * gen(rng, G)
        labels, outcomes, clusters = np.repeat(cc, 2), np.repeat(yo, 2), np.repeat(np.arange(G), 2)
        # window-miss truth: per cell scale s, half-window w: 1 - max_a P(|s e - a|<=w)
        w = 1.0
        def wfloor(s):
            e = eq * s
            right = np.searchsorted(e, e + 2*w, side="right"); return 1 - np.max(right - np.arange(len(e))) / len(e)
        golds = {"mae": base_floor * np.mean(scale), "q90": q90_unit * np.mean(scale),
                 "w1": float(np.mean([wfloor(s) for s in scale[:0]] or [0]))}
        wf = {c: wfloor(.7 + .12*c) for c in range(8)}
        golds["w1"] = float(np.mean([wf[c] for c in cc]))
        common = dict(representation_provenance="frozen-eight-cell-map-v1", independent_clusters_attested=True,
                      outcome_blind_labels_attested=True)
        for loss, kw in (("mae", {}), ("q90", {"quantile": .9}), ("w1", {"window": w})):
            L = {"mae": "mae", "q90": "pinball", "w1": "window_miss"}[loss]
            dom = dominated_saa_lower_bound(labels, outcomes, clusters, loss=L, caps=R.CAPS, alpha=.05, **kw, **common)
            blk = clustered_block_saa_portfolio(labels, outcomes, clusters, loss=L, block_cluster_sizes=R.BLOCK_SIZES,
                    score_caps=R.CAPS if L != "window_miss" else (1.0,), alpha=.05, partition_seed=4401,
                    representation_provenance="m", partition_provenance="p", independent_clusters_attested=True,
                    outcome_blind_labels_attested=True, partition_frozen_before_outcomes=True, **kw)
            row = dict(mech=name, seed=seed, loss=loss, gold=golds[loss], dom=dom.bound, dom_cap=dom.selected_cap,
                       dom_ineq=dom.selected_inequality, block=blk.bound, block_route=blk.selected_route)
            if loss == "mae":
                ps, ret = R._pair_scores(cc, yo, 4401)
                row["pair"] = truncated_saa_portfolio({"p": ps}, loss="mae", score_caps=R.CAPS, alpha_family=.05,
                    retained_fractions={"p": ret}, route_provenance={"x": 1}, portfolio_frozen_before_evaluation=True).bound
            rows.append(row)
    print(name, round(time.time()-t,1), flush=True)
open(out, "w").write("".join(json.dumps(r) + "\n" for r in rows))
