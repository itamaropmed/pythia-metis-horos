"""WP1 + WP3: OOF (current Mayo replay) vs split calibration; alpha-ledger and contrast-test variants."""
import sys, json, time
import numpy as np
from sim_lib import *

def one(rep, n_dev=6000, n_later=1500, n_fresh=60000):
    rng = np.random.default_rng(10_000 + rep)
    dev = population(n_dev, rng); later = population(n_later, rng, pid0=10**6); fresh = population(n_fresh, rng, pid0=2 * 10**6)
    res = {}
    # ---------------- current pipeline: 5-fold OOF certification, final model deployed
    oof = {}; fin = {}
    for arm, X in (("A", "XA"), ("B", "XB")):
        o = np.zeros(len(dev["y"]))
        for k, (tr, va) in enumerate(GroupKFold(5).split(dev[X], dev["y"], dev["pid"])):
            o[va] = gbm(rep + k).fit(dev[X][tr], dev["y"][tr]).predict(dev[X][va])
        oof[arm] = o; fin[arm] = gbm(rep).fit(dev[X], dev["y"])
    pe = edges((oof["A"] + oof["B"]) / 2, 5); de = edges(np.abs(oof["A"] - oof["B"]), 3)
    lab_dev = labels(dev["F"], oof["A"], oof["B"], pe, de)
    pl = {a: fin[a].predict(later["X" + a]) for a in "AB"}; pf = {a: fin[a].predict(fresh["X" + a]) for a in "AB"}
    lab_l = labels(later["F"], pl["A"], pl["B"], pe, de); lab_f = labels(fresh["F"], pf["A"], pf["B"], pe, de)
    losses = {a: (np.abs(dev["y"] - oof[a]) > THR).astype(float) for a in "AB"}
    p_fresh = {a: miss_prob(pf[a], fresh["mu"], fresh["s"]) for a in "AB"}
    t0 = time.time(); reg = registry(losses, lab_dev, dev["pid"])
    res["oof_all_overlap"] = evaluate(reg, "AB", lab_l, lab_f, p_fresh); res["oof_all_overlap"]["sec"] = time.time() - t0
    res["mae_final"] = {a: float(np.abs(later["y"] - pl[a]).mean()) for a in "AB"}
    # ---------------- split calibration: train 60% patients, calibrate 40%
    pats = np.unique(dev["pid"]); trp = set(rng.choice(pats, int(.6 * len(pats)), replace=False))
    tr = np.array([p in trp for p in dev["pid"]]); ca = ~tr
    mod = {a: gbm(rep + 7).fit(dev["X" + a][tr], dev["y"][tr]) for a in "AB"}
    pc = {a: mod[a].predict(dev["X" + a][ca]) for a in "AB"}
    pe2 = edges((pc["A"] + pc["B"]) / 2, 5); de2 = edges(np.abs(pc["A"] - pc["B"]), 3)
    lab_c = labels(dev["F"][ca], pc["A"], pc["B"], pe2, de2)
    pl2 = {a: mod[a].predict(later["X" + a]) for a in "AB"}; pf2 = {a: mod[a].predict(fresh["X" + a]) for a in "AB"}
    lab_l2 = labels(later["F"], pl2["A"], pl2["B"], pe2, de2); lab_f2 = labels(fresh["F"], pf2["A"], pf2["B"], pe2, de2)
    lc = {a: (np.abs(dev["y"][ca] - pc[a]) > THR).astype(float) for a in "AB"}
    pf2p = {a: miss_prob(pf2[a], fresh["mu"], fresh["s"]) for a in "AB"}
    res["mae_split"] = {a: float(np.abs(later["y"] - pl2[a]).mean()) for a in "AB"}
    for name, kw in (("split_all_overlap", {}), ("split_eligible_overlap", dict(alpha_ledger="eligible")),
                     ("split_eligible_contrast", dict(alpha_ledger="eligible", relation_test="contrast"))):
        t0 = time.time(); reg = registry(lc, lab_c, dev["pid"][ca], **kw)
        res[name] = evaluate(reg, "AB", lab_l2, lab_f2, pf2p); res[name]["sec"] = time.time() - t0
        res[name]["n_tests"] = reg.n_charged_tests
    # outcome-blind backoff partition: certify exactly the routed regions
    from pythia.core.local_model_risk import build_support_partition
    part = build_support_partition(lab_c, dev["pid"][ca], 100)
    hier_c = {"region": part.labels(lab_c), "global": lab_c["global"]}
    hier_l = {"region": part.labels(lab_l2), "global": lab_l2["global"]}
    hier_f = {"region": part.labels(lab_f2), "global": lab_f2["global"]}
    for name, kw in (("split_partition_overlap", dict(alpha_ledger="eligible")),
                     ("split_partition_contrast", dict(alpha_ledger="eligible", relation_test="contrast"))):
        t0 = time.time(); reg = registry(lc, hier_c, dev["pid"][ca], **kw)
        res[name] = evaluate(reg, "AB", hier_l, hier_f, pf2p); res[name]["sec"] = time.time() - t0
        res[name]["n_tests"] = reg.n_charged_tests; res[name]["n_regions"] = len(part.regions)
    # same improved ledger on the full-data OOF registry, to separate ledger effect from split effect
    reg = registry(losses, lab_dev, dev["pid"], alpha_ledger="eligible", relation_test="contrast")
    res["oof_eligible_contrast"] = evaluate(reg, "AB", lab_l, lab_f, p_fresh)
    return res

if __name__ == "__main__":
    a, b = int(sys.argv[1]), int(sys.argv[2])
    for rep in range(a, b):
        t = time.time(); r = one(rep); r["rep"] = rep; r["sec_total"] = time.time() - t
        print(json.dumps(r), flush=True)
