import json, sys, numpy as np
R = [json.loads(l) for f in sys.argv[1:] for l in open(f) if l.startswith('{"src"')]
print("| Dataset | true G | PERAS-U G interval | U/true | old LP-1 interval | old U/true | AFI ceiling/true | true NLI V | **PERAS-U V ≥** | old LP V ≥ | old AFI ≥ | table [L,U] vs true | binding | viol |")
print("|---" * 14 + "|")
tot = 0; viol_all = 0
for ds in dict.fromkeys(r["dataset"] for r in R):
    S = [r for r in R if r["dataset"] == ds]; m = lambda f: np.mean([f(r) for r in S])
    v = sum((r["U_G"][0] > r["true_G"] + 1e-9) or (r["U_G"][1] < r["true_G"] - 1e-9) or (r["U_V_lower"] > r["true_V"] + 1e-9)
            or not (r["U_table"][0] - 1e-9 <= r["true_table"] <= r["U_table"][1] + 1e-9)
            or (r["U_Gpop"][0] > r["true_G_all"] + 1e-9) or (r["U_Gpop"][1] < r["true_G_all"] - 1e-9)
            or (r["S_G"][0] > r["true_G_strat"] + 1e-9) or (r["S_G"][1] < r["true_G_strat"] - 1e-9) or (r["S_V"] > r["true_V_strat"] + 1e-9)
            for r in S)
    tot += len(S); viol_all += v
    bind = {b: sum(r["U_binding"] == b for r in S) for b in ("pair", "table-NLI", "model")}
    print(f"| {ds} | {m(lambda r: r['true_G']):.3f} | [{m(lambda r: r['U_G'][0]):.3f}, {m(lambda r: r['U_G'][1]):.3f}] | {m(lambda r: r['U_G'][1]/r['true_G']):.3f} |"
          f" [{m(lambda r: r['LP_G'][0]):.3f}, {m(lambda r: r['LP_G'][1]):.3f}] | {m(lambda r: r['LP_G'][1]/r['LP_true_G']):.3f} | {m(lambda r: r['AFI_GXT_U']/r['true_G']):.3f} |"
          f" {m(lambda r: r['true_V']):.3f} | **{m(lambda r: r['U_V_lower']):.3f}** | {m(lambda r: r['LP_V'] or 0):.3f} | {m(lambda r: r['AFI_dI']):.3f} |"
          f" [{m(lambda r: r['U_table'][0]):.3f}, {m(lambda r: r['U_table'][1]):.3f}] vs {m(lambda r: r['true_table']):.3f} | {bind} | {v}/{len(S)} |")
print(f"\nAll-bound violations (cohort G both sides, V, table, population G, stratified G and V): {viol_all}/{tot}")
print("refuted A(1/2):", sum(r["U_refuted"] for r in R), "/", len(R), "; mean tipping eps*:", np.mean([r["U_eps_tip"] for r in R]).round(4))
print("stratified (age band) G upper / true:", np.mean([r["S_G"][1] / r["true_G_strat"] for r in R]).round(3), " stratified V lower / true V:",
      np.mean([r["S_V"] / max(r["true_V_strat"], 1e-9) for r in R if r["true_V_strat"] > 1e-4]).round(3))
print("pooled V lower / true V:", np.mean([r["U_V_lower"] / r["true_V"] for r in R if r["true_V"] > 1e-4]).round(3),
      " old LP:", np.mean([(r["LP_V"] or 0) / r["true_V"] for r in R if r["true_V"] > 1e-4]).round(3),
      " old AFI:", np.mean([r["AFI_dI"] / r["true_V"] for r in R if r["true_V"] > 1e-4]).round(3))
