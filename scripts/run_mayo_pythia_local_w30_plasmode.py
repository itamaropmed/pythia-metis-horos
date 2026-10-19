"""Plasmode validation of the Pythia local-W30 certificate on REAL Mayo covariates (local, aggregates only).

The real Mayo design (covariates, procedure families, patients, repeated cases, scheduled times, the
temporal split) is kept.  Only the outcome is re-simulated from a KNOWN law, so every certificate can be
checked against the exact truth:
    y* = m(x) + sqrt(1 - rho) * r + u_patient,    u_patient ~ N(0, rho * v_family)
where m is a fixed XGBoost mean fitted once on all real rows (arm B), r is drawn from the real residual pool
of the case's procedure family (variance v_family), and the patient effect makes repeated cases dependent.
For a frozen model f the true miss probability of case i is
    P_i = mean_j [ Phi((d_i - 30 - a r_j)/s) + 1 - Phi((d_i + 30 - a r_j)/s) ],   d_i = f(x_i) - m(x_i),
averaged over the family residual pool (a = sqrt(1 - rho), s = sd of u).
Per replicate it runs the v1 route (5-fold OOF + refit + backoff hierarchy) and the v2 route
(split + partition + eligible ledger + contrast) and records only aggregates:
  * family coverage of the stated estimand (v1: whole node; v2: routed region, calibration population)
  * wrong certified directions for the cases actually routed (vs true risk of the routed population)
  * later-cohort truth inside the interval (real temporal covariate drift, no outcome-law drift)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from pythia.core.local_model_risk import build_local_model_risk_registry, build_support_partition
from run_mayo_metis_case_risk import MODEL_KWARGS, procedure_family, split_indices
from run_mayo_pythia_local_w30_v2 import _edges, _labels, inner_split

NAMES = ("A", "B")


def true_miss(pred, mean, fam, pools, a, sd_u):
    out = np.empty(len(pred))
    for f, pool in pools.items():
        m = fam == f
        if not m.any():
            continue
        d = (pred[m] - mean[m])[:, None]; r = a * pool[None, :]
        s = sd_u[f]
        out[m] = (norm.cdf((d - 30 - r) / s) + norm.sf((d + 30 - r) / s)).mean(1)
    return out


def relation(node, gnode, contrast):
    if contrast and node.contrast_lower is not None:
        return 1 if node.contrast_lower > 0 else -1 if node.contrast_upper < 0 else 0
    return 1 if node.lower > gnode.upper else -1 if node.upper < gnode.lower else 0


def audit(reg, lab, idx_rows, truth, gtruth, contrast):
    """Route rows (vectorised), return coverage/direction stats against the true risk of routed rows."""
    st = dict(cells=0, node_miss=0, routed_miss=0, wrong_rows=0, cert_rows=0, rows=0)
    for m in NAMES:
        done = np.zeros(len(idx_rows), bool)
        for level in reg.level_order:
            keys = lab[level]
            for key in np.unique(keys[~done]):
                node = reg.nodes.get(f"{m}|{level}|{key}")
                if node is None or node.n_entities < reg.min_entities:
                    continue
                sel = (~done) & (keys == key); done |= sel
                tr_routed = truth[m][sel].mean(); tr_node = truth[m][keys == key].mean()
                st["cells"] += 1; st["rows"] += int(sel.sum())
                st["node_miss"] += int(not node.lower - 1e-9 <= tr_node <= node.upper + 1e-9)
                st["routed_miss"] += int(not node.lower - 1e-9 <= tr_routed <= node.upper + 1e-9)
                if level != "global":
                    rel = relation(node, reg.nodes[f"{m}|global|all"], contrast)
                    if rel != 0:
                        st["cert_rows"] += int(sel.sum())
                        if (rel > 0) != (tr_routed > gtruth[m]):
                            st["wrong_rows"] += int(sel.sum())
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="JSON-lines, one aggregate record per replicate")
    ap.add_argument("--reps", type=str, default="0:3")
    ap.add_argument("--rho", type=float, default=.3)
    ap.add_argument("--budget-sec", type=float, default=150)
    ap.add_argument("--later-noise", type=float, default=1.0, help="outcome-law drift: later residual scale")
    ap.add_argument("--v2-only", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    frame = pd.read_parquet(args.derived / "df.parquet")
    X = {"A": pd.read_parquet(args.derived / "XA.parquet").to_numpy(np.float32),
         "B": pd.read_parquet(args.derived / "XB.parquet").to_numpy(np.float32)}
    y = frame.case_actual_duration_time.to_numpy(float)
    pid = frame.patient_durable_id.astype(str).to_numpy()
    fam = frame.primary_procedure_name.map(procedure_family).to_numpy()
    kw = dict(MODEL_KWARGS); kw["n_jobs"] = 4
    mean = XGBRegressor(random_state=123, **kw).fit(X["B"], y).predict(X["B"])     # fixed "true" mean
    resid = y - mean
    pools = {f: resid[fam == f] for f in np.unique(fam)}
    var_f = {f: float(np.var(p)) for f, p in pools.items()}
    a = np.sqrt(1 - args.rho); sd_u = {f: np.sqrt(args.rho * v) + 1e-9 for f, v in var_f.items()}
    pools_mc = {f: np.random.default_rng(1).choice(p, min(300, len(p)), replace=False) for f, p in pools.items()}
    dev, later, _ = split_indices(frame); fit, cal = inner_split(frame, dev, .6)
    upid, pinv = np.unique(pid, return_inverse=True)
    lo, hi = (int(v) for v in args.reps.split(":"))
    for rep in range(lo, hi):
        if time.time() - t0 > args.budget_sec:
            break
        rng = np.random.default_rng(900 + rep)
        u = rng.normal(size=len(upid))[pinv]
        r = np.array([rng.choice(pools[f]) for f in fam])
        ys = mean + a * r + np.array([sd_u[f] for f in fam]) * u
        k = args.later_noise
        ys[later] = mean[later] + k * (ys[later] - mean[later])          # drift only in the later cohort
        rec = {"rep": rep, "later_noise": k}
        # ---------------- v1: OOF on dev, refit, backoff hierarchy, all-node ledger, overlap
        oof = {n: np.zeros(len(dev)) for n in NAMES}; final = {}
        for n in (() if args.v2_only else NAMES):
            for k, (tr, va) in enumerate(GroupKFold(5).split(X[n][dev], ys[dev], pid[dev])):
                oof[n][va] = XGBRegressor(random_state=k, **kw).fit(X[n][dev][tr], ys[dev][tr]).predict(X[n][dev][va])
            final[n] = XGBRegressor(random_state=0, **kw).fit(X[n][dev], ys[dev])
        if not args.v2_only:
            pe = _edges((oof["A"] + oof["B"]) / 2, 5); de = _edges(np.abs(oof["A"] - oof["B"]), 3)
            hd = _labels(fam[dev], oof["A"], oof["B"], pe, de)
            pl = {n: final[n].predict(X[n][later]) for n in NAMES}
            hl = _labels(fam[later], pl["A"], pl["B"], pe, de)
            reg1 = build_local_model_risk_registry({n: (np.abs(ys[dev] - oof[n]) > 30).astype(float) for n in NAMES}, hd, pid[dev],
                                                   loss_cap=1., alpha=.05, min_entities=100, outcome_blind_hierarchy=True,
                                                   independent_entities=True, loss_cap_attested=True)
            tl = {n: true_miss(pl[n], mean[later], fam[later], pools_mc, a * k, {f: v * k for f, v in sd_u.items()}) for n in NAMES}
            rec["v1_later"] = audit(reg1, hl, later, tl, {n: tl[n].mean() for n in NAMES}, False)
        # ---------------- v2: split, partition, eligible ledger, contrast
        mod = {n: XGBRegressor(random_state=7, **kw).fit(X[n][fit], ys[fit]) for n in NAMES}
        pc = {n: mod[n].predict(X[n][cal]) for n in NAMES}; pl2 = {n: mod[n].predict(X[n][later]) for n in NAMES}
        pe2 = _edges((pc["A"] + pc["B"]) / 2, 5); de2 = _edges(np.abs(pc["A"] - pc["B"]), 3)
        hc = _labels(fam[cal], pc["A"], pc["B"], pe2, de2); hl2 = _labels(fam[later], pl2["A"], pl2["B"], pe2, de2)
        part = build_support_partition(hc, pid[cal], 100)
        hcr = {"region": part.labels(hc), "global": hc["global"]}; hlr = {"region": part.labels(hl2), "global": hl2["global"]}
        reg2 = build_local_model_risk_registry({n: (np.abs(ys[cal] - pc[n]) > 30).astype(float) for n in NAMES}, hcr, pid[cal],
                                               loss_cap=1., alpha=.05, min_entities=100, outcome_blind_hierarchy=True,
                                               independent_entities=True, loss_cap_attested=True,
                                               alpha_ledger="eligible", relation_test="contrast")
        tc = {n: true_miss(pc[n], mean[cal], fam[cal], pools_mc, a, sd_u) for n in NAMES}
        tl2 = {n: true_miss(pl2[n], mean[later], fam[later], pools_mc, a * k, {f: v * k for f, v in sd_u.items()}) for n in NAMES}
        rec["v2_calibration"] = audit(reg2, hcr, cal, tc, {n: tc[n].mean() for n in NAMES}, True)   # the theorem's estimand
        rec["v2_later"] = audit(reg2, hlr, later, tl2, {n: tl2[n].mean() for n in NAMES}, True)     # transport to later
        from pythia.core.local_transport import odds_transport
        cov = {}
        for lam in (1.0, 1.25, 1.5):
            c = t = 0
            for n in NAMES:
                for key in np.unique(hlr["region"]):
                    node = reg2.nodes.get(f"{n}|region|{key}")
                    if node is None or not node.eligible:
                        continue
                    lo_, hi_ = odds_transport(node.lower, node.upper, lam)
                    tr_ = tl2[n][hlr["region"] == key].mean(); t += 1; c += int(lo_ <= tr_ <= hi_)
            cov[str(lam)] = [c, t]
        rec["v2_later_transport_coverage"] = cov
        rec["later_true_mae_proxy"] = {"v2": {n: float(np.abs(ys[later] - pl2[n]).mean()) for n in NAMES}}
        if not args.v2_only:
            rec["later_true_mae_proxy"]["v1"] = {n: float(np.abs(ys[later] - pl[n]).mean()) for n in NAMES}
        rec["sec"] = time.time() - t0
        with open(args.output, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps({k: v for k, v in rec.items()}), flush=True)


if __name__ == "__main__":
    main()
