"""Mayo local W30 certificate, v2 route (split calibration + support partition + contrast + transport).

Run LOCALLY on the machine that holds the derived Mayo files; writes AGGREGATES ONLY (no row-level output,
no identifiers, no note text).  Pythia trains nothing: the XGBoost arms below are Horos-side replay.

Pipeline (all patient-disjoint, time-ordered):
  development (before the 80th-percentile cutoff)
     -> FIT patients   (first development case earlier in time; `--fit-share` of development patients)
     -> CALIBRATION patients (the remaining development patients)
  later cohort (after the cutoff, patients never seen in development)
1. Fit each arm once on FIT rows; this frozen model is the deployed model.
2. On CALIBRATION rows: predictions -> hierarchy edges -> outcome-blind support partition -> losses.
3. Simultaneous eligible-ledger intervals + local-vs-global contrasts (one alpha ledger, both models).
4. Freeze routed cards for every later case BEFORE reading later outcomes.
5. Afterwards (audit only): later W30 per region, odds-transport coverage at Lambda in {1, 1.25, 1.5},
   tipping Lambda* per certified direction, anytime-valid monitor (first case per patient, time order).
Optional --compare-v1 re-runs the original OOF route for a side-by-side relation count.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from pythia.core.local_model_risk import build_local_model_risk_registry, build_support_partition
from pythia.core.local_transport import BettingRiskMonitor, contrast_tipping_lambda, odds_transport
from run_mayo_metis_case_risk import MODEL_KWARGS, procedure_family, split_indices

LAMBDAS = (1.0, 1.25, 1.5)
NAMES = ("basic_xgboost", "xgboost_plus_11_llm")


def _edges(values: np.ndarray, bins: int) -> np.ndarray:
    raw = np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1])
    return np.unique(raw[np.isfinite(raw)])


def _labels(family, pred_a, pred_b, pred_edges, dis_edges):
    pbin = np.digitize((pred_a + pred_b) / 2, pred_edges).astype(str)
    dbin = np.digitize(np.abs(pred_a - pred_b), dis_edges).astype(str)
    fam = np.asarray(family).astype(str)
    return {
        "family_prediction_disagreement": np.char.add(np.char.add(np.char.add(fam, "|"), pbin), np.char.add("|", dbin)),
        "family_prediction": np.char.add(np.char.add(fam, "|"), pbin),
        "prediction": pbin,
        "global": np.repeat("all", len(fam)),
    }


def inner_split(frame: pd.DataFrame, dev: np.ndarray, fit_share: float):
    """Patient-disjoint, time-ordered split of development rows into FIT and CALIBRATION."""
    t = pd.to_datetime(frame["scheduled_in_room"].iloc[dev], utc=True).to_numpy()
    pid = frame["patient_durable_id"].astype(str).to_numpy()[dev]
    first = pd.Series(t).groupby(pid).transform("min").to_numpy()
    order = np.sort(pd.Series(first).groupby(pid).first().to_numpy())
    cut = order[int(fit_share * (len(order) - 1))]
    fit = dev[first <= cut]; cal = dev[first > cut]
    assert not set(pid[first <= cut]) & set(pid[first > cut])
    return fit, cal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit-share", type=float, default=.6)
    ap.add_argument("--min-entities", type=int, default=100, help="primary, declared before outcomes")
    ap.add_argument("--sensitivity-min-entities", type=str, default="200,400")
    ap.add_argument("--alpha", type=float, default=.05)
    ap.add_argument("--compare-v1", action="store_true")
    args = ap.parse_args()

    frame = pd.read_parquet(args.derived / "df.parquet")
    X = {NAMES[0]: pd.read_parquet(args.derived / "XA.parquet").to_numpy(np.float32),
         NAMES[1]: pd.read_parquet(args.derived / "XB.parquet").to_numpy(np.float32)}
    y = frame.case_actual_duration_time.to_numpy(float)
    pid = frame.patient_durable_id.astype(str).to_numpy()
    fam_all = frame.primary_procedure_name.map(procedure_family).to_numpy()
    dev, later, cutoff = split_indices(frame)
    fit, cal = inner_split(frame, dev, args.fit_share)

    models = {n: XGBRegressor(random_state=args.seed, **MODEL_KWARGS).fit(X[n][fit], y[fit]) for n in NAMES}
    pcal = {n: models[n].predict(X[n][cal]) for n in NAMES}
    plat = {n: models[n].predict(X[n][later]) for n in NAMES}
    pe = _edges((pcal[NAMES[0]] + pcal[NAMES[1]]) / 2, 5)
    de = _edges(np.abs(pcal[NAMES[0]] - pcal[NAMES[1]]), 3)
    h_cal = _labels(fam_all[cal], pcal[NAMES[0]], pcal[NAMES[1]], pe, de)
    h_lat = _labels(fam_all[later], plat[NAMES[0]], plat[NAMES[1]], pe, de)
    loss_cal = {n: (np.abs(y[cal] - pcal[n]) > 30).astype(float) for n in NAMES}

    def certify(min_entities: int) -> dict:
        part = build_support_partition(h_cal, pid[cal], min_entities)
        hc = {"region": part.labels(h_cal), "global": h_cal["global"]}
        hl = {"region": part.labels(h_lat), "global": h_lat["global"]}
        reg = build_local_model_risk_registry(
            loss_cal, hc, pid[cal], loss_cap=1., alpha=args.alpha, min_entities=min_entities,
            outcome_blind_hierarchy=True, independent_entities=True, loss_cap_attested=True,
            alpha_ledger="eligible", relation_test="contrast")
        # ---- freeze cards for every later case before reading later outcomes
        cards = {n: [reg.route(n, {"region": hl["region"][i], "global": "all"}).to_dict() for i in range(len(later))]
                 for n in NAMES}
        # ---- audit (later outcomes revealed only now)
        out = {"regions": len(part.regions), "charged_tests": reg.n_charged_tests,
               "alpha_per_tail": reg.alpha_per_tail_node_model}
        t_lat = pd.to_datetime(frame["scheduled_in_room"].iloc[later], utc=True).to_numpy()
        order = np.argsort(t_lat, kind="stable")
        first = ~pd.Series(pid[later][order]).duplicated().to_numpy()
        for n in NAMES:
            loss_lat = (np.abs(y[later] - plat[n]) > 30).astype(float)
            g = reg.nodes[f"{n}|global|all"]
            rel = [c["local_relation"] for c in cards[n]]
            regions, cells = sorted(set(hl["region"])), []
            for key in regions:
                rows = np.flatnonzero(hl["region"] == key)
                node = reg.nodes.get(f"{n}|region|{key}")
                if node is None or not node.eligible:
                    continue
                obs = float(loss_lat[rows].mean())
                cell = {"later_rows": int(len(rows)), "dev_interval": [node.lower, node.upper],
                        "later_observed_w30": obs, "relation": cards[n][rows[0]]["local_relation"]}
                for lam in LAMBDAS:
                    lo, hi = odds_transport(node.lower, node.upper, lam)
                    cell[f"observed_inside_L{lam}"] = bool(lo <= obs <= hi)
                if cell["relation"].startswith("certified"):
                    cell["tipping_lambda"] = contrast_tipping_lambda(
                        (node.lower, node.upper), (node.contrast_lower, node.contrast_upper), (g.lower, g.upper))
                    cell["direction_agrees_later"] = bool((obs > loss_lat.mean()) == cell["relation"].endswith("higher_than_global"))
                mon = BettingRiskMonitor(min(node.upper, .999), args.alpha)
                stream = loss_lat[order][(hl["region"][order] == key) & first]
                for x in stream:
                    mon.update(x)
                    if mon.alarm_at is not None:
                        break
                cell["monitor_alarm_at_patient"] = mon.alarm_at
                cells.append(cell)
            big = [c for c in cells if c["later_rows"] >= 50]
            cert = [c for c in cells if "tipping_lambda" in c]
            out[n] = {
                "later_mae": float(np.abs(y[later] - plat[n]).mean()),
                "later_w30": float(loss_lat.mean()),
                "relation_counts": {r: rel.count(r) for r in sorted(set(rel))},
                "median_interval_width": float(np.median([c["upper"] - c["lower"] for c in cards[n] if c["upper"] is not None])),
                "cells_with_50plus_later_rows": len(big),
                **{f"observed_inside_L{lam}_50plus": int(sum(c[f"observed_inside_L{lam}"] for c in big)) for lam in LAMBDAS},
                "certified_cells": len(cert),
                "certified_cells_direction_agrees_later": int(sum(c["direction_agrees_later"] for c in cert)),
                "median_tipping_lambda": float(np.median([c["tipping_lambda"] for c in cert])) if cert else None,
                "monitor_alarms": int(sum(c["monitor_alarm_at_patient"] is not None for c in cells)),
                "cells": cells,
            }
        return out

    result = {
        "status": "retrospective temporal audit of the v2 route; not prospective confirmation",
        "estimand": "case-weighted W30-miss probability of the FROZEN deployed model in each routed region",
        "pythia_trains_models": False,
        "split": {"cutoff": cutoff, "fit_rows": int(len(fit)), "calibration_rows": int(len(cal)),
                  "later_rows": int(len(later)),
                  "fit_patients": int(len(set(pid[fit]))), "calibration_patients": int(len(set(pid[cal]))),
                  "later_patients": int(len(set(pid[later]))),
                  "overlap_fit_cal": int(len(set(pid[fit]) & set(pid[cal]))),
                  "overlap_dev_later": int(len(set(pid[dev]) & set(pid[later])))},
        "primary": {"min_entities": args.min_entities, **certify(args.min_entities)},
        "sensitivity_predeclared": {m: certify(int(m)) for m in args.sensitivity_min_entities.split(",") if m},
    }
    if args.compare_v1:
        from run_mayo_metis_case_risk import crossfit_and_holdout
        from run_mayo_pythia_local_w30 import _labels as v1_labels  # original hierarchy builder
        oof = {}; test = {}
        for n in NAMES:
            oof[n], test[n] = crossfit_and_holdout(X[n], y, pid, dev, later, args.seed)
        pe1 = _edges((oof[NAMES[0]] + oof[NAMES[1]]) / 2, 5); de1 = _edges(np.abs(oof[NAMES[0]] - oof[NAMES[1]]), 3)
        hd = v1_labels(frame, dev, oof[NAMES[0]], oof[NAMES[1]], pe1, de1)
        hl1 = v1_labels(frame, later, test[NAMES[0]], test[NAMES[1]], pe1, de1)
        reg1 = build_local_model_risk_registry(
            {n: (np.abs(y[dev] - oof[n]) > 30).astype(float) for n in NAMES}, hd, pid[dev], loss_cap=1.,
            alpha=args.alpha, min_entities=100, outcome_blind_hierarchy=True, independent_entities=True,
            loss_cap_attested=True)
        result["v1_original_route"] = {n: dict(pd.Series([reg1.route(n, {k: v[i] for k, v in hl1.items()}).local_relation
                                                           for i in range(len(later))]).value_counts()) for n in NAMES}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    brief = {"split": result["split"]}
    for tag, block in [("primary", result["primary"])] + list(result["sensitivity_predeclared"].items()):
        brief[str(tag)] = {n: {k: v for k, v in block[n].items() if k != "cells"} for n in NAMES}
        brief[str(tag)]["regions"] = block["regions"]
    if "v1_original_route" in result:
        brief["v1_original_route"] = result["v1_original_route"]
    print(json.dumps(brief, indent=2, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
