"""Summarise NLI OOD runs: size, power, paired exact McNemar vs baselines, certificate validity."""
import json, sys, math, numpy as np
from scipy.stats import binomtest

merged = {}
for f in sys.argv[1:]:
    for l in open(f):
        if l.startswith("{"):
            r = json.loads(l); merged.setdefault((r["family"], str(r["level"]), r["rep"]), {}).update(r)
rows = [r for r in merged.values() if "nli_perm_p" in r and "nli_proj_perm_p" in r]
for r in rows:
    r["level"] = {"null": 0, "beta=1.25": 1, "beta=2.0": 2}.get(r["level"], r["level"])
rows = [r for r in rows if r["level"] in (0, 1, 2)]
A = .05
for r in rows:
    r["NLI-combo"] = min(1.0, 2 * min(r["nli_perm_p"], r["nli_perm_learned_p"])) <= A   # pre-specified primary
    r["NLI-geom"] = r["nli_perm_p"] <= A
    r["NLI-learned"] = r["nli_perm_learned_p"] <= A
    r["NLI-cert>0"] = r["nli_lower"] > 0
    r["Boot-dBrier"] = r["model_boot_reject"]
    r["LocalPerm-dBrier"] = r["model_perm_p"] <= A
    r["CODEC-perm"] = r["codec_p"] <= A
    r["NLI-proj-perm"] = r["nli_proj_perm_p"] <= A
    r["NLI-proj-cf"] = r["nli_proj_cf_p"] <= A
    r["NLI-proj-cert>0"] = r["nli_proj_cert"] > 0
    r["WGCM-normal"] = r["wgcm_p"] <= A
M = ["NLI-proj-perm", "NLI-proj-cf", "NLI-proj-cert>0", "NLI-combo", "NLI-cert>0", "Boot-dBrier", "LocalPerm-dBrier", "CODEC-perm", "WGCM-normal"]
BASE = ["Boot-dBrier", "LocalPerm-dBrier", "CODEC-perm", "WGCM-normal"]
OURS = ["NLI-proj-perm", "NLI-proj-cf", "NLI-proj-cert>0", "NLI-combo"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def mcnemar(a, b):
    b01 = sum(x and not y for x, y in zip(a, b)); b10 = sum(y and not x for x, y in zip(a, b))
    p = binomtest(b01, b01 + b10, .5).pvalue if b01 + b10 else 1.0
    return b01, b10, p


fams = sorted({r["family"] for r in rows})
print("| Family | Level | true NLI | n runs | " + " | ".join(M) + " |")
print("|---" * (4 + len(M)) + "|")
for f in fams:
    for lv in (0, 1, 2):
        R = [r for r in rows if r["family"] == f and r["level"] == lv]
        if not R:
            continue
        tn = np.mean([r["true_nli"] for r in R])
        print(f"| {f} | {lv} | {tn:.4f} | {len(R)} | " + " | ".join(f"{np.mean([r[m] for r in R]):.2f}" for m in M) + " |")

print("\n## Size (level 0, all families pooled)")
R0 = [r for r in rows if r["level"] == 0]
for m in M:
    k = sum(r[m] for r in R0); lo, hi = wilson(k, len(R0)); print(f"- {m}: {k}/{len(R0)} = {k/len(R0):.3f}  [{lo:.3f}, {hi:.3f}]")

print("\n## Power (levels 1-2 pooled) and paired exact McNemar of NLI-combo vs each baseline")
for lvset, name in (((1,), "level 1"), ((2,), "level 2"), ((1, 2), "levels 1+2")):
    R = [r for r in rows if r["level"] in lvset]
    print(f"\n### {name} (n = {len(R)})")
    for m in M:
        k = sum(r[m] for r in R); lo, hi = wilson(k, len(R)); print(f"- power {m}: {k/len(R):.3f} [{lo:.3f}, {hi:.3f}]")
    for ours in OURS:
        for b in BASE:
            b01, b10, p = mcnemar([r[ours] for r in R], [r[b] for r in R])
            print(f"- {ours} vs {b}: ours-only {b01}, base-only {b10}, exact McNemar p = {p:.2g}")

print("\n## Per-family McNemar, NLI-proj-perm vs best VALID baseline (levels 1+2)")
for f in fams:
    R = [r for r in rows if r["family"] == f and r["level"] > 0]
    best = max(["LocalPerm-dBrier", "CODEC-perm", "Boot-dBrier"], key=lambda b: sum(r[b] for r in R))
    b01, b10, p = mcnemar([r["NLI-proj-perm"] for r in R], [r[best] for r in R])
    print(f"- {f}: NLI-proj-perm {np.mean([r['NLI-proj-perm'] for r in R]):.2f} vs {best} {np.mean([r[best] for r in R]):.2f};"
          f" ours-only {b01}, base-only {b10}, p = {p:.2g}")

print("\n## Certificate validity and estimation")
viol = sum(r["nli_lower"] > r["true_nli"] + 1e-12 for r in rows); viol2 = sum(r["nli_proj_cert"] > r["true_nli"] + 1e-12 for r in rows)
vb = sum(r["model_delta_lo"] > r["true_nli"] + 1e-12 for r in rows)
print(f"- pair certificate above truth: {viol}/{len(rows)}; projection certificate above truth: {viol2}/{len(rows)}; bootstrap dBrier lower CI above truth: {vb}/{len(rows)}")
for lv in (1, 2):
    R = [r for r in rows if r["level"] == lv]
    print(f"- level {lv}: mean true {np.mean([r['true_nli'] for r in R]):.4f}, mean certified {np.mean([r['nli_lower'] for r in R]):.4f},"
          f" mean projection cert {np.mean([r['nli_proj_cert'] for r in R]):.4f} (ratio to truth {np.mean([r['nli_proj_cert']/r['true_nli'] for r in R]):.2f}),"
          f" mean model dBrier {np.mean([r['model_delta'] for r in R]):.4f}, mean boot lower {np.mean([r['model_delta_lo'] for r in R]):.4f}, share model dBrier < 0: {np.mean([r['model_delta'] < 0 for r in R]):.2f}")
