#!/usr/bin/env python3
"""Frozen run for DOMINATED_SAA_CONFIRMATION_REGISTRATION.md (seeds 2,200,001-2,200,500)."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "interventions_20261014")); sys.path.insert(0, str(HERE.parents[2]))
import run_support_free_cluster_block_confirmation as R
from peras import clustered_block_saa_portfolio, truncated_saa_portfolio
from peras.core.dominated_saa import dominated_saa_lower_bound

def main():
    rows = []
    for mi, (name, gen, base_floor) in enumerate(R.LAWS):
        eq = np.sort(gen(np.random.default_rng(78), 2_000_000))
        d = eq - np.quantile(eq, .9); q90_unit = float(np.mean(np.where(d >= 0, .9 * d, -.1 * d)))
        def wfloor(s, w=1.0):
            e = eq * s; right = np.searchsorted(e, e + 2 * w, side="right")
            return 1 - np.max(right - np.arange(len(e))) / len(e)
        wf = {c: wfloor(.7 + .12 * c) for c in range(8)}
        for rep in range(50):
            seed = 2_200_001 + 50 * mi + rep
            rng = np.random.default_rng(seed)
            G = 4096
            cc = rng.integers(0, 8, G); scale = .7 + .12 * cc
            yo = 5. * cc + scale * gen(rng, G)
            lab, y, cl = np.repeat(cc, 2), np.repeat(yo, 2), np.repeat(np.arange(G), 2)
            gold = {"mae": base_floor * float(np.mean(scale)), "q90": q90_unit * float(np.mean(scale)),
                    "w1": float(np.mean([wf[c] for c in cc]))}
            for loss, L, kw in (("mae", "mae", {}), ("q90", "pinball", {"quantile": .9}),
                                ("w1", "window_miss", {"window": 1.0})):
                dom = dominated_saa_lower_bound(lab, y, cl, loss=L, caps=R.CAPS, alpha=.05, inequalities=("bentkus",),
                        representation_provenance="frozen-eight-cell-map-v1", independent_clusters_attested=True,
                        outcome_blind_labels_attested=True, **kw)
                blk = clustered_block_saa_portfolio(lab, y, cl, loss=L, block_cluster_sizes=R.BLOCK_SIZES,
                        score_caps=R.CAPS if L != "window_miss" else (1.0,), alpha=.05, partition_seed=4401,
                        representation_provenance="frozen-eight-cell-map-v1",
                        partition_provenance="seeded-disjoint-patient-blocks-v1", independent_clusters_attested=True,
                        outcome_blind_labels_attested=True, partition_frozen_before_outcomes=True, **kw)
                row = dict(mechanism=name, seed=seed, loss=loss, gold=gold[loss], dominated=dom.bound,
                           dominated_cap=dom.selected_cap, block=blk.bound, block_route=blk.selected_route)
                if loss == "mae":
                    ps, ret = R._pair_scores(cc, yo, 4401)
                    row["pair"] = truncated_saa_portfolio({"p": ps}, loss="mae", score_caps=R.CAPS, alpha_family=.05,
                        retained_fractions={"p": ret}, route_provenance={"partition": "pairs"},
                        portfolio_frozen_before_evaluation=True).bound
                rows.append(row)
        print(name, flush=True)
    (HERE / "dominated_saa_confirmation_results.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))

if __name__ == "__main__":
    main()
