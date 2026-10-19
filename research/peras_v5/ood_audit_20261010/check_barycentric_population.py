"""Post-hoc numeric check: barycentric caps vs population common-quantile dispersions (5,000-point quadrature),
for the unaligned [0,300] route and the env-median-aligned [-150,150] route, all 50 registered datasets."""
import json, numpy as np
from audit_ood import make
import peras5 as P
from barycentric_coupled_bound import calibrate_bary, environment_bary_costs
from run_coupled_environment_ood import KS, OUTCOME_CLIP, CAP_DELTA_FAMILY, repeated_outcomes, ppf
from run_disjoint_transport_audit import greedy_feature_blocks
from run_barycentric_ood import RESIDUAL_CLIP
u = (np.arange(5000) + .5) / 5000; gaps = {"abs": [], "sq": []}; miss = 0; n = 0
for design in range(1, 11):
    for seed in range(103001, 103006):
        d = make(design, 800, seed); rep = P.represent(d)
        z = np.empty((5000, 800))
        for fam in np.unique(d["fam"]):
            m = d["fam"] == fam; z[:, m] = ppf(fam, u[:, None])
        pop = d["mu"][None, :] + d["s"][None, :] * z
        env = repeated_outcomes(d, 5000, seed + 9_400_000, common_quantile=True)
        learn, cal = np.array_split(env, 2, axis=0); shift = np.median(learn, axis=0)
        for k in KS:
            b = greedy_feature_blocks(rep, k)
            for clip, E, sh in [(OUTCOME_CLIP, env, None), (RESIDUAL_CLIP, cal, shift)]:
                cap = calibrate_bary(E, b, clip=clip, delta_family=CAP_DELTA_FAMILY, coupling_provenance="x", shift=sh, shift_provenance="x" if sh is not None else None)
                a, s, _, _ = environment_bary_costs(pop, b, clip=clip, shift=sh)
                for key, c, p in [("abs", cap["abs_cap"], a.mean()), ("sq", cap["sq_cap"], s.mean())]:
                    gaps[key].append(c - p); n += 1; miss += (c - p) < -1e-3 * max(1, p)
print(json.dumps({"cap_checks": n, "population_cap_misses": int(miss),
                  "min_margin_abs": float(min(gaps["abs"])), "median_margin_abs": float(np.median(gaps["abs"])),
                  "min_margin_sq": float(min(gaps["sq"])), "median_margin_sq": float(np.median(gaps["sq"]))}, indent=1))
