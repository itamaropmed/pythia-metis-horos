import json, sys, numpy as np, collections
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
print("rows", len(rows), "violations:", sum(r["viol"] for r in rows))
mechs = sorted({r["mech"] for r in rows})
def best(m, cfg, loss, pt, wit="const"):
    c = collections.defaultdict(list)
    for r in rows:
        if r["mech"] == m and r["cfg"] == cfg and r["loss"] == loss and r["pen_type"] == pt and r["wit"] == wit:
            c[r["k"]].append(r)
    if not c: return None
    k, v = max(c.items(), key=lambda kv: np.median([r["ratio"] for r in kv[1]]))
    return np.median([r["ratio"] for r in v]), k, np.median([r["pen"] for r in v]), np.median([min(r["tip"],999) for r in v])
for loss in ("mae", "q90"):
    print(f"\n=== {loss}: median bound/gold [penalty/floor] (best k) ===")
    for cfg in ("raw_kd", "learned_key", "raw_align", "oracle_key"):
        print(f"-- {cfg}")
        print("   mech       " + "".join(p.rjust(20) for p in ("bary", "curv", "jensen(oracle)", "LL+curv")))
        for m in mechs:
            line = f"   {m:11s}"
            for pt, wit in (("bary", "const"), ("curv", "const"), ("jensen", "const"), ("curv", "ll")):
                b = best(m, cfg, loss, pt, wit)
                line += ("".rjust(20) if b is None else f"{b[0]:6.2f}[{b[2]:.2f}]k{b[1]:<2d} tip{b[3]:4.1f}".rjust(20))
            print(line)
