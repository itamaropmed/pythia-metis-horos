import json, sys, numpy as np
from scipy.stats import beta
R = [json.loads(l) for f in sys.argv[1:] for l in open(f) if l.startswith("{")]
def cp(k, n):   # Clopper-Pearson 95%
    return (0 if k == 0 else beta.ppf(.025, k, n - k + 1), 1 if k == n else beta.ppf(.975, k + 1, n - k))
V = ["oof_all_overlap", "oof_eligible_contrast", "split_all_overlap", "split_eligible_overlap",
     "split_eligible_contrast", "split_partition_overlap", "split_partition_contrast"]
print(f"replicates: {len(R)}")
print("| Variant | Node-family coverage (reps) | Node misses / cells | Routed-population misses / cells | Wrong certified directions (rows, routed truth) | Certified share of later cases | Median width | Charged tests |")
print("|---|---|---|---|---|---|---|---|")
for v in V:
    S = [r[v] for r in R]
    fam = sum(s["miss_cells"] == 0 for s in S); lo, hi = cp(fam, len(S))
    cells = sum(s["cells_checked"] for s in S)
    rm = sum(s.get("routed_miss_cells", 0) for s in S); nm = sum(s["miss_cells"] for s in S)
    fd = sum(s["false_dir"] for s in S); cert = sum(s["certified"] for s in S); n = sum(s["n"] for s in S)
    nt = np.median([s.get("n_tests") or np.nan for s in S])
    print(f"| {v} | {fam}/{len(S)} [{lo:.3f}, {hi:.3f}] | {nm}/{cells} | {rm}/{cells} | {fd} / {cert} certified | {cert/n:.3f} | {np.median([s['median_width'] for s in S]):.3f} | {nt:.0f} |")
print("\nRows with a wrong certified direction (routed truth), per replicate:",
      {v: int(sum(r[v]["false_dir"] > 0 for r in R)) for v in V})
print("Later MAE  OOF-final A/B: %.2f / %.2f   split A/B: %.2f / %.2f" % (
    np.mean([r["mae_final"]["A"] for r in R]), np.mean([r["mae_final"]["B"] for r in R]),
    np.mean([r["mae_split"]["A"] for r in R]), np.mean([r["mae_split"]["B"] for r in R])))
