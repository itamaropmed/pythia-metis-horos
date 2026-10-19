import json, sys, numpy as np, collections
rows=[json.loads(l) for f in sys.argv[1:] for l in open(f)]
M=("pair_first","pair_second","knn_group_first","peras")
for loss in ("mae","q90"):
    R=[r for r in rows if r["loss"]==loss]
    print(f"\n=== {loss}: n_rows={len(R)} violations:", {m:sum(r[m]>r["gold"]*(1+1e-3) for r in R) for m in M})
    print("design      n    "+"".join(m.rjust(16) for m in M)+"   peras>best_accepted")
    g=collections.defaultdict(list)
    for r in R: g[(r["design"],r["n"])].append(r)
    for (d,n),v in g.items():
        med={m:np.median([x[m]/x["gold"] for x in v]) for m in M}
        wins=sum(x["peras"]>max(x["pair_first"],x["pair_second"],x["knn_group_first"])+1e-12 for x in v)
        ties=sum(abs(x["peras"]-max(x["pair_first"],x["pair_second"],x["knn_group_first"]))<=1e-12 for x in v)
        print(f"{d:11s}{n:5d} "+"".join(f"{med[m]:16.3f}" for m in M)+f"   {wins}/{len(v)} (ties {ties})")
