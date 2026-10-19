import json, sys, numpy as np, collections
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
cert = [r for r in rows if "lcb" in r]
est = [r for r in rows if r.get("loss") == "mse_est"]
print("violations:", sum(r["viol"] for r in cert), "/", len(cert),
      "| dKelly tighter than Bentkus:", sum(r["kelly_win"] for r in cert), "/", len(cert))
g = collections.defaultdict(list)
for r in cert:
    g[(r["mech"], r["n"], r["strat"], r["k"], r["loss"])].append(r)
mechs = sorted({r["mech"] for r in cert}); strats = ["raw_kd", "learned_key", "raw_align", "learned_align", "oracle_key", "oracle_align"]
for loss in ("mae", "q90", "mseu"):
    print(f"\n=== {loss}: median bound/gold (best k per strategy) ===")
    print("mech".ljust(11), "".join(s[:13].rjust(14) for s in strats))
    for m in mechs:
        line = m.ljust(11)
        for s in strats:
            best = max((np.median([r["ratio"] for r in v]), k) for (mm, n, ss, k, L), v in g.items()
                       if mm == m and ss == s and L == loss) if any(
                (mm == m and ss == s and L == loss) for (mm, n, ss, k, L) in g) else (np.nan, 0)
            line += f"{best[0]:9.2f}(k{best[1]:>2})"
        print(line)
print("\n=== decomposition (median over mechanisms & seeds), fraction of true floor ===")
print("strat         k loss  opt   meanS  pen   width  bound  tip")
for s in strats:
    for k in (2, 8, 32):
        for loss in ("mae", "q90", "mseu"):
            v = [r for r in cert if r["strat"] == s and r["k"] == k and r["loss"] == loss]
            if not v: continue
            med = lambda key: np.median([r[key] for r in v])
            print(f"{s:13s}{k:2d} {loss:5s}{med('opt'):5.2f} {med('meanS'):6.2f} {med('pen'):5.2f} {med('width'):6.2f} {med('ratio'):6.2f} {med('tip'):6.2f}")
print("\n=== estimate tier (MSE point estimates / truth, median [p10,p90]) ===")
for s in ("nn_raw", "nn_learned", "hgb_oof"):
    for m in mechs:
        v = [r["ratio"] for r in est if r["strat"] == s and r["mech"] == m]
        print(f"{s:11s}{m:11s} {np.median(v):6.2f} [{np.quantile(v,.1):.2f},{np.quantile(v,.9):.2f}]")
