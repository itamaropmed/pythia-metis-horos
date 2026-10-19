import json, sys, numpy as np
R = [json.loads(l) for f in sys.argv[1:] for l in open(f) if l.startswith("{")]
print(f"replicates: {len(R)} (two models each)")
print("| Scenario | Cells | Later truth inside dev interval (Λ=1) | Λ=1.25 | Λ=1.5 | Λ=2 | Certified rows | Wrong certified direction (later) | Harmful cells (truth > upper) | Monitor alarms: harmful / non-harmful | Median alarm delay (patients) | Median tipping Λ* |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for s in R[0]:
    if s in ("rep", "sec"):
        continue
    S = [r[s] for r in R]; c = sum(x["cells"] for x in S)
    cov = {l: sum(x[f"cover_L{l}"] for x in S) / c for l in (1.0, 1.25, 1.5, 2.0)}
    harm = sum(x["harmful"] for x in S); al = sum(x["alarms"] for x in S); ah = sum(x["alarms_on_harmful"] for x in S)
    d = [x["median_delay"] for x in S if x["median_delay"]]; tp = [x["median_tip"] for x in S if x["median_tip"]]
    print(f"| {s} | {c} | {cov[1.0]:.3f} | {cov[1.25]:.3f} | {cov[1.5]:.3f} | {cov[2.0]:.3f} | {sum(x['certified'] for x in S)} | "
          f"{sum(x['wrong_dir_later'] for x in S)} | {harm} | {ah}/{harm} vs {al-ah}/{c-harm} | {np.median(d) if d else '—'} | {np.median(tp) if tp else '—'} |")
