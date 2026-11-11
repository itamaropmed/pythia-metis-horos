"""Data-owner scorer: joins frozen runner outputs with the withheld oracle files. Run ONLY after all predictions are
frozen (hashes recorded).  Claude never runs this on withheld replicates.

Usage: python3 score_with_oracle.py RUN_ROOT ORACLE_DIR OUT.json
  RUN_ROOT/<observed stem>/{certificates.json, predictions.csv.gz, run_meta.json}
  ORACLE_DIR/<stem with 'observed' replaced by 'oracle'>.csv(.gz)   (row_id + exact conditional mean + noise sd [+ true W30 risk])
Per replicate and method: coverage (bound <= V_true), tightness (bound / V_true), test MAE vs the oracle-optimal MAE,
W30 risk accuracy/correlation vs exact risk, labeler wrong-direction rate.  Aggregates with Clopper-Pearson CIs.
"""
import glob, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm, spearmanr, pearsonr, beta

ROOT, ORA, OUTF = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])


def cp(k, n, a=.05):
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.; hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.; return [float(lo), float(hi)]


def eclip(a, s, c):
    al, be = (-c - a) / s, (c - a) / s
    return -c * norm.cdf(al) + c * norm.sf(be) + a * (norm.cdf(be) - norm.cdf(al)) + s * (norm.pdf(al) - norm.pdf(be))


def col(df, pats):
    for p in pats:
        m = [c for c in df.columns if re.fullmatch(p, c)]
        if m:
            return m[0]
    return None


rows = []
for d in sorted(p for p in ROOT.iterdir() if (p / "certificates.json").exists()):
    C = json.loads((d / "certificates.json").read_text()); M = json.loads((d / "run_meta.json").read_text())
    P = pd.read_csv(d / "predictions.csv.gz")
    cands = glob.glob(str(ORA / (d.name.replace("observed", "oracle") + ".csv*")))
    if not cands:
        print("no oracle for", d.name); continue
    O = pd.read_csv(cands[0]); X = P.merge(O, on="row_id", how="left", validate="one_to_one")
    mu = X[col(X, [r"conditional_mean_table_plus_text", r".*mean.*plus.*text.*", r"true_mean.*"])].to_numpy(float)
    sd = X[col(X, [r"known_noise_sd", r".*noise_sd.*"])].to_numpy(float)
    rk = col(X, [r"true_risk_w30", r".*risk.*w30.*", r".*w30.*risk.*"])
    split = X["split"].astype(str).str.lower(); cal, te = split.str.startswith("cal").to_numpy(), (split == "test").to_numpy()
    g = X["cell"].astype(str).to_numpy(); f = X["f_table"].to_numpy(float); cap = C["cap_min"]
    m = eclip(mu - f, sd, cap); mc = m[cal] - pd.Series(m[cal]).groupby(g[cal]).transform("mean").to_numpy()
    V = float(np.mean(mc ** 2)); stem = d.name; design = re.sub(r"[._-]?(rep|replicate|seed)?[._-]?\d+.*$", "", stem.replace(".observed", ""))
    rec = {"run": stem, "design": design, "V_true_cal_capped": V, "methods": {}}
    for k, v in C["readers"].items():
        e = {}
        for meth in ("betting", "mp", "clt"):
            e[meth] = {"bound": v[meth], "covered": bool(v[meth] <= V + 1e-9), "tightness": v[meth] / V if V > 0 else None}
        e["vimp_if"] = {"bound": v["vimp_if"]["V_lower_equiv"], "covered": bool(v["vimp_if"]["V_lower_equiv"] <= V + 1e-9)}
        e["plugin_bootstrap"] = {"bound": v["plugin_bootstrap_lower"], "covered": bool(v["plugin_bootstrap_lower"] <= V + 1e-9)}
        y = None
        rec["methods"][k] = e
    yv = X[col(X, [r"outcome_duration_minutes", r"duration.*"])] if col(X, [r"outcome_duration_minutes", r"duration.*"]) else None
    rec["duration_test"] = {"oracle_mean_MAE_vs_mu": 0.0}
    for cc in [c for c in P.columns if c.startswith("dur_")] + ["f_table"]:
        rec["duration_test"][cc] = {"MAE_to_true_mean": float(np.abs(X[cc].to_numpy()[te] - mu[te]).mean())}
    if rk:
        pt = X[rk].to_numpy(float); rec["risk_test"] = {}
        for cc in [c for c in P.columns if c.startswith("risk_")]:
            q = X[cc].to_numpy(float)[te]
            rec["risk_test"][cc] = {"MAE": float(np.abs(q - pt[te]).mean()), "pearson": float(pearsonr(q, pt[te])[0]) if q.std() > 0 else 0.,
                                     "spearman": float(spearmanr(q, pt[te])[0]) if q.std() > 0 else 0.}
        glob_t = pt[te].mean(); wrong = tot = 0
        for dd in ("higher", "lower"):
            sel = te & (X["labeler_direction"].to_numpy() == dd)
            if sel.any():
                tot += sel.sum(); bad = pt[sel].mean() <= glob_t if dd == "higher" else pt[sel].mean() >= glob_t; wrong += sel.sum() * bad
        rec["labeler"] = {"test_rows_with_direction": int(tot), "test_rows_wrong_direction": int(wrong)}
    rec["null_false_claims"] = C["null_within_cell_permutation"]
    rows.append(rec)

agg = {}
for r_ in rows:
    A = agg.setdefault(r_["design"], {"replicates": 0, "methods": {}, "null": {}}); A["replicates"] += 1
    for k, e in r_["methods"].items():
        for meth, z in e.items():
            B = A["methods"].setdefault(f"{k}|{meth}", {"covered": 0, "n": 0, "tight": []}); B["n"] += 1; B["covered"] += z["covered"]
            if z.get("tightness") is not None:
                B["tight"].append(z["tightness"])
    K = r_["null_false_claims"]["K"]
    for k, d_ in r_["null_false_claims"]["false_claim_rate"].items():
        for meth, rate in d_.items():
            N = A["null"].setdefault(f"{k}|{meth}", {"claims": 0, "trials": 0}); N["claims"] += int(round(rate * K)); N["trials"] += K
for A in agg.values():
    for B in A["methods"].values():
        B["coverage"] = B["covered"] / B["n"]; B["coverage_CI"] = cp(B["covered"], B["n"]); B["mean_tightness"] = float(np.mean(B["tight"])) if B["tight"] else None; B.pop("tight")
    for N in A["null"].values():
        N["false_claim_rate"] = N["claims"] / N["trials"]; N["CI"] = cp(N["claims"], N["trials"])
OUTF.write_text(json.dumps({"per_replicate": rows, "aggregate": agg}, indent=1, default=float))
print(json.dumps(agg, indent=1, default=float)[:6000])
