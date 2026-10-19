"""WP4: temporal shift. Split-calibrated partition certificate (contrast relations), later cohort under
five shift scenarios. Measures (i) coverage of the LATER true region risk by the development interval,
(ii) the same after odds transport at declared Lambda, (iii) direction errors, (iv) anytime-valid
monitor alarms (harmful-shift detection) per region."""
import sys, json, time
import numpy as np
from sim_lib import *
from pythia.core.local_model_risk import build_support_partition
from pythia.core.local_transport import odds_transport, transported_relation, tipping_lambda, contrast_tipping_lambda, BettingRiskMonitor

FP2 = np.array([.08, .1, .12, .13, .14, .15, .15, .13])
SCEN = {"S0_none": {}, "S1_case_mix": {"family_prob": FP2},
        "S2_noise_up_15pct": {"sigma_mult": np.full(8, 1.15)},
        "S3_noise_up_40pct": {"sigma_mult": np.full(8, 1.4)},
        "S4_long_cases_plus30min": {"mean_shift": np.array([0, 0, 0, 0, 0, 0, 30, 30.])},
        "S5_covariate_drift_z+1": {"z_mean": 1.0},
        "S6_all_cases_plus15min": {"mean_shift": np.full(8, 15.)}}
LAMS = (1.0, 1.25, 1.5, 2.0)

def one(rep, n_dev=6000, n_later=2500, n_fresh=60000):
    rng = np.random.default_rng(50_000 + rep)
    dev = population(n_dev, rng)
    pats = np.unique(dev["pid"]); trp = set(rng.choice(pats, int(.6 * len(pats)), replace=False))
    tr = np.array([p in trp for p in dev["pid"]]); ca = ~tr
    mod = {a: gbm(rep + 7).fit(dev["X" + a][tr], dev["y"][tr]) for a in "AB"}
    pc = {a: mod[a].predict(dev["X" + a][ca]) for a in "AB"}
    pe = edges((pc["A"] + pc["B"]) / 2, 5); de = edges(np.abs(pc["A"] - pc["B"]), 3)
    lab_c = labels(dev["F"][ca], pc["A"], pc["B"], pe, de)
    part = build_support_partition(lab_c, dev["pid"][ca], 100)
    hc = {"region": part.labels(lab_c), "global": lab_c["global"]}
    lc = {a: (np.abs(dev["y"][ca] - pc[a]) > THR).astype(float) for a in "AB"}
    reg = registry(lc, hc, dev["pid"][ca], alpha_ledger="eligible", relation_test="contrast")
    out = {}
    for sname, sh in SCEN.items():
        lat = population(n_later, rng, shift=sh, pid0=10**6); fr = population(n_fresh, rng, shift=sh, pid0=2 * 10**6)
        res = {f"cover_L{l}": 0 for l in LAMS}; res.update(cells=0, wrong_dir_dev=0, wrong_dir_later=0, certified=0,
                                                             n=0, alarms=0, harmful=0, alarms_on_harmful=0, delays=[], tip=[])
        res.update({f"cert_L{l}": 0 for l in LAMS})
        for a in "AB":
            pl = mod[a].predict(lat["X" + a]); pf = mod[a].predict(fr["X" + a])
            lab_l = labels(lat["F"], *( [mod[b].predict(lat["X" + b]) for b in "AB"]), pe, de)
            lab_f = labels(fr["F"], *([mod[b].predict(fr["X" + b]) for b in "AB"]), pe, de)
            rl = part.labels(lab_l); rf = part.labels(lab_f)
            pfp = miss_prob(pf, fr["mu"], fr["s"]); gtrue = pfp.mean()
            g = reg.nodes[f"{a}|global|all"]; gint = (g.lower, g.upper)
            lat_loss = (np.abs(lat["y"] - pl) > THR).astype(float)
            first = np.r_[True, lat["pid"][1:] != lat["pid"][:-1]]      # first case per patient for the monitor
            for key in np.unique(rl):
                node = reg.nodes.get(f"{a}|region|{key}")
                if node is None or not node.eligible:
                    continue
                selF = rf == key; selL = rl == key
                if selF.sum() < 200:
                    continue
                truth = pfp[selF].mean(); nrow = int(selL.sum())
                res["cells"] += 1; res["n"] += nrow
                for l in LAMS:
                    lo, hi = odds_transport(node.lower, node.upper, l)
                    res[f"cover_L{l}"] += int(lo <= truth <= hi)
                    rel_l = transported_relation((node.lower, node.upper), gint, l)
                    res[f"cert_L{l}"] += nrow * rel_l.startswith("certified")
                rel = ("certified_higher_than_global" if node.contrast_lower > 0 else
                       "certified_lower_than_global" if node.contrast_upper < 0 else "local_shift_unresolved")
                if rel.startswith("certified"):
                    res["certified"] += nrow
                    res["tip"].append(contrast_tipping_lambda((node.lower, node.upper), (node.contrast_lower, node.contrast_upper), gint))
                    hi_true = truth > gtrue
                    if rel.startswith("certified_higher") != hi_true:
                        res["wrong_dir_later"] += nrow
                # monitor: harmful shift = later true risk above the certified upper
                harmful = truth > node.upper
                res["harmful"] += int(harmful)
                mon = BettingRiskMonitor(min(node.upper, .999), .05)
                for x in lat_loss[selL & first]:
                    mon.update(x)
                    if mon.alarm_at is not None:
                        break
                if mon.alarm_at is not None:
                    res["alarms"] += 1; res["alarms_on_harmful"] += int(harmful); res["delays"].append(mon.alarm_at)
        res["median_tip"] = float(np.median(res["tip"])) if res["tip"] else None
        res["median_delay"] = float(np.median(res["delays"])) if res["delays"] else None
        res.pop("tip"); res.pop("delays")
        out[sname] = res
    return out

if __name__ == "__main__":
    a, b = int(sys.argv[1]), int(sys.argv[2])
    for rep in range(a, b):
        t = time.time(); r = one(rep); r["rep"] = rep; r["sec"] = time.time() - t
        print(json.dumps(r, default=float), flush=True)
