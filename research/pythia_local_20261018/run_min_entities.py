"""Resolution vs support threshold for the recommended route (split + support partition + contrast)."""
import sys, json, time
import numpy as np
from sim_lib import *
from pythia.core.local_model_risk import build_support_partition

def one(rep):
    rng = np.random.default_rng(70_000 + rep)
    dev = population(6000, rng); later = population(1500, rng, pid0=10**6); fresh = population(60000, rng, pid0=2 * 10**6)
    pats = np.unique(dev["pid"]); trp = set(rng.choice(pats, int(.6 * len(pats)), replace=False))
    tr = np.array([p in trp for p in dev["pid"]]); ca = ~tr
    mod = {a: gbm(rep + 7).fit(dev["X" + a][tr], dev["y"][tr]) for a in "AB"}
    pc = {a: mod[a].predict(dev["X" + a][ca]) for a in "AB"}
    pe = edges((pc["A"] + pc["B"]) / 2, 5); de = edges(np.abs(pc["A"] - pc["B"]), 3)
    lab_c = labels(dev["F"][ca], pc["A"], pc["B"], pe, de)
    pl = {a: mod[a].predict(later["X" + a]) for a in "AB"}; pf = {a: mod[a].predict(fresh["X" + a]) for a in "AB"}
    lab_l = labels(later["F"], pl["A"], pl["B"], pe, de); lab_f = labels(fresh["F"], pf["A"], pf["B"], pe, de)
    lc = {a: (np.abs(dev["y"][ca] - pc[a]) > THR).astype(float) for a in "AB"}
    pfp = {a: miss_prob(pf[a], fresh["mu"], fresh["s"]) for a in "AB"}
    out = {}
    for me in (50, 100, 200, 400):
        part = build_support_partition(lab_c, dev["pid"][ca], me)
        hc = {"region": part.labels(lab_c), "global": lab_c["global"]}
        hl = {"region": part.labels(lab_l), "global": lab_l["global"]}; hf = {"region": part.labels(lab_f), "global": lab_f["global"]}
        reg = build_local_model_risk_registry(lc, hc, dev["pid"][ca], loss_cap=1., alpha=.05, min_entities=me,
                                              outcome_blind_hierarchy=True, independent_entities=True, loss_cap_attested=True,
                                              alpha_ledger="eligible", relation_test="contrast")
        out[str(me)] = evaluate(reg, "AB", hl, hf, pfp); out[str(me)]["n_regions"] = len(part.regions)
    return out

if __name__ == "__main__":
    for rep in range(int(sys.argv[1]), int(sys.argv[2])):
        r = one(rep); r["rep"] = rep; print(json.dumps(r), flush=True)
