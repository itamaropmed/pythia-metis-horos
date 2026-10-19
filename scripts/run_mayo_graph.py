#!/usr/bin/env python3
"""PYTHIA-G on Mayo (aggregate only, exploratory): dev-split dependence audit, then graph-Chernoff
certificates on the temporal holdout and full cohort.  Usage:
  python scripts/run_mayo_graph.py --tabular .../mayo_hrs_tabular_features_and_durations.parquet --out result.json
"""
import argparse, json, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from pythia.folds import build_grouped_temporal_fold_manifest
from pythia.core.graph_dominated_saa import (graph_dominated_saa_lower_bound, dependence_audit,
                                            dependence_sensitivity_curve)
import run_mayo_clustered_family as M

ap = argparse.ArgumentParser(); ap.add_argument("--tabular", required=True); ap.add_argument("--out")
a = ap.parse_args()
tmp = Path(tempfile.mkdtemp()) / "folds.parquet"
build_grouped_temporal_fold_manifest(a.tabular, tmp)
man = pd.read_parquet(tmp)
cols = ["case_durable_id", "patient_durable_id", "surgeon_durable_id", "scheduled_in_room",
        "case_actual_duration_time", "primary_procedure_name"]
fr = pd.read_parquet(a.tabular, columns=cols).dropna(subset=["patient_durable_id", "case_actual_duration_time"])
fr["_t"] = pd.to_datetime(fr["scheduled_in_room"], errors="coerce")
fr = fr.sort_values(["_t", "case_durable_id"]).reset_index(drop=True)
role = man.set_index("case_durable_id")["outer_role"].reindex(fr["case_durable_id"]).to_numpy()
y = fr["case_actual_duration_time"].to_numpy(float)
proc = fr["primary_procedure_name"].fillna("missing").astype(str).to_numpy()
fam = np.asarray([M._family(v) for v in proc])
day = fr["_t"].dt.strftime("%Y-%m-%d").fillna("na").to_numpy()
week = fr["_t"].dt.strftime("%G-%V").fillna("na").to_numpy()
surg = fr["surgeon_durable_id"].astype(str).to_numpy()
units_all = {"patient": fr["patient_durable_id"].astype(str).to_numpy(),
             "surgeon_day": np.char.add(np.char.add(surg.astype(str), "|"), day.astype(str)),
             "day": day, "week": week, "surgeon": surg}
dev = role == "development"
audit = dependence_audit(y[dev], proc[dev], {k: v[dev] for k, v in units_all.items()}, n_perm=499, seed=1)
declared = ["patient"] + [k for k in ("surgeon_day", "day", "week") if audit[k]["p_value"] is not None and audit[k]["p_value"] <= .01]
out = {"audit": audit, "declared_units": declared, "nodes": {}}
CAPS = (10., 20., 40., 80., 160., 320.)
for scope, mask in (("holdout", role == "holdout"), ("full", np.ones(len(y), bool))):
    for name, lab in (("family", fam), ("procedure", proc)):
        res = {}
        for loss, kw in (("mae", {}), ("pinball", {"quantile": .9}), ("window_miss", {"window": 30.})):
            g = graph_dominated_saa_lower_bound(lab[mask], y[mask], {k: units_all[k][mask] for k in declared},
                    loss=loss, caps=CAPS, alpha=.05 / 4, retention_grid=(1, 2, None), retention_unit="patient",
                    representation_provenance=f"mayo-{name}", dependency_graph_attested=False,
                    outcome_blind_labels_attested=True, exploratory_override=True, **kw)
            best = max(g.candidates, key=lambda c: c["bound"])
            res[loss] = {"bound": g.bound, "K": best["colours"], "omega": best["max_clique"],
                         "retention": best["retention"], "cap": best["cap"],
                         "sensitivity": dependence_sensitivity_curve(best["statistic"] * len(y[mask]) / best["kept_rows"],
                                best["cap"], best["kept_rows"], (1, best["colours"], 2 * best["colours"], 4 * best["colours"]), .05 / 4 / 18)}
        out["nodes"][f"{scope}:{name}"] = res
text = json.dumps(out, indent=1, default=str)
if a.out: Path(a.out).write_text(text)
print(text)
