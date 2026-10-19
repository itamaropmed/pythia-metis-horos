#!/usr/bin/env python3
"""pythia-cert: run Pythia certificates on a local CSV and write AGGREGATE results only.

Subcommands
  local-risk   simultaneous local W30-style risk intervals for frozen models (v2 route:
               support partition + eligible ledger + contrast + odds transport)
  nli          note/text information certificate (NLI cohort lower bound) combined with the
               table-floor box -> certified upper bound on the joint (table + text) floor
  table-floor  table (cell) Brier floor box, marginal box and table information
  version      print plugin and engine versions

Every certificate is conditional on attestations the user must make explicitly (--attest-*).
A missing attestation gives status "refused" (exit code 2), never a silent weaker answer.
Outputs never contain row-level values or entity identifiers unless --row-cards is given.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLUGIN_ROOT = HERE.parent
for cand in (PLUGIN_ROOT / "vendor", PLUGIN_ROOT.parent.parent):   # vendored snapshot, else repo checkout
    if (cand / "pythia" / "core" / "unified.py").exists():
        sys.path.insert(0, str(cand))
        break

try:
    import numpy as np
    from pythia.core.local_model_risk import build_local_model_risk_registry, build_support_partition
    from pythia.core.local_transport import contrast_tipping_lambda, odds_transport
    from pythia.core.unified import marginal_box, model_ceiling, nli_cohort_bound, table_box
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(f"pythia-cert: missing dependency ({exc}). Install with: pip install numpy scipy\n")
    sys.exit(3)

MIN_ENTITIES_FLOOR = 50   # privacy + validity floor: no region/cell reported below this many entities


def plugin_version() -> str:
    try:
        return json.loads((PLUGIN_ROOT / ".claude-plugin" / "plugin.json").read_text())["version"]
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------------- io helpers
def read_csv(path: Path) -> dict[str, list[str]]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return {k: [r[k] for r in rows] for k in rows[0].keys()}


def need(cols: dict, names: list[str], path) -> None:
    miss = [n for n in names if n not in cols]
    if miss:
        raise SystemExit(f"{path}: missing columns {miss}; found {list(cols)}")


def refuse(args, kind: str, reasons: list[str]) -> None:
    res = {"status": "refused", "certificate": kind, "claim_tier": "none", "reasons": reasons,
           "plugin_version": plugin_version()}
    emit(args, res, f"# Pythia {kind}: REFUSED\n\n" + "\n".join(f"- {r}" for r in reasons) + "\n")
    sys.exit(2)


def emit(args, result: dict, md: str) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pythia_result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=_jd) + "\n")
    (out / "pythia_report.md").write_text(md)
    print(json.dumps({k: v for k, v in result.items() if k not in ("regions",)}, indent=2, sort_keys=True,
                     default=_jd))
    print(f"\nwrote {out / 'pythia_result.json'} and {out / 'pythia_report.md'}", file=sys.stderr)


def _jd(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


def fmt(x, d=4):
    return "—" if x is None else f"{x:.{d}f}"


# ----------------------------------------------------------------------------- local-risk
def cmd_local_risk(args) -> None:
    reasons = []
    if not args.attest_outcome_blind_levels:
        reasons.append("--attest-outcome-blind-levels missing: hierarchy levels must be built without outcomes "
                       "(bin edges from predictions/covariates only)")
    if not args.attest_independent_entities:
        reasons.append("--attest-independent-entities missing: different entities (patients) must be independent "
                       "given the design; repeated rows of one entity may be dependent")
    if not args.attest_loss_cap:
        reasons.append("--attest-loss-cap missing: every loss must lie in [0, loss_cap] almost surely")
    if not args.attest_frozen_model:
        reasons.append("--attest-frozen-model missing: the certified model must be the deployed model, fitted on "
                       "data disjoint (by entity) from the calibration file")
    if args.min_entities < MIN_ENTITIES_FLOOR:
        reasons.append(f"--min-entities {args.min_entities} < {MIN_ENTITIES_FLOOR} (privacy and validity floor)")
    if reasons:
        refuse(args, "local-risk", reasons)

    cal = read_csv(Path(args.cal))
    levels = [s for s in args.levels.split(",") if s]
    loss_cols = [s for s in args.loss_cols.split(",") if s]
    need(cal, [args.entity_col] + levels + loss_cols, args.cal)
    ent = np.asarray(cal[args.entity_col], dtype=object)
    hier = {lv: np.asarray(cal[lv], dtype=str) for lv in levels}
    hier["global"] = np.repeat("all", len(ent))
    losses = {c: np.asarray(cal[c], float) for c in loss_cols}
    for c, v in losses.items():
        if not np.isfinite(v).all() or v.min() < 0 or v.max() > args.loss_cap:
            refuse(args, "local-risk", [f"loss column {c} violates the attested range [0, {args.loss_cap}]"])

    part = build_support_partition(hier, ent, args.min_entities)
    h_cal = {"region": part.labels(hier), "global": hier["global"]}
    reg = build_local_model_risk_registry(
        losses, h_cal, ent, loss_cap=args.loss_cap, alpha=args.alpha, min_entities=args.min_entities,
        outcome_blind_hierarchy=True, independent_entities=True, loss_cap_attested=True,
        alpha_ledger="eligible", relation_test="contrast")
    lams = [float(s) for s in args.lambdas.split(",") if s]

    regions = []
    for key, node in sorted(reg.nodes.items()):
        if not node.eligible:
            continue
        model = node.model
        g = reg.nodes[f"{model}|global|all"]
        row = {"model": model, "region": node.key, "level": node.level, "n_rows": node.n_rows,
               "n_entities": node.n_entities, "estimate": node.estimate, "lower": node.lower, "upper": node.upper,
               "contrast": [node.contrast_lower, node.contrast_upper]}
        if node.level != "global":
            if node.contrast_lower is not None and node.contrast_lower > 0:
                row["relation"] = "certified_higher_than_global"
            elif node.contrast_upper is not None and node.contrast_upper < 0:
                row["relation"] = "certified_lower_than_global"
            else:
                row["relation"] = "local_shift_unresolved"
            if row["relation"].startswith("certified") and args.loss_cap == 1.0:
                row["tipping_lambda"] = contrast_tipping_lambda((node.lower, node.upper),
                                                                (node.contrast_lower, node.contrast_upper),
                                                                (g.lower, g.upper))
        if args.loss_cap == 1.0:
            row["transported"] = {str(l): list(odds_transport(node.lower, node.upper, l)) for l in lams}
        regions.append(row)

    result = {
        "status": "certified", "certificate": "local-risk", "route": "v2 (support partition, eligible ledger, contrast)",
        "claim_tier": "conditional (C1): CAL population of each routed region; C2 transported intervals are "
                      "conditional on the declared odds-shift budget Lambda",
        "plugin_version": plugin_version(),
        "alpha_family": args.alpha, "alpha_per_tail": reg.alpha_per_tail_node_model,
        "charged_tests": reg.n_charged_tests, "min_entities": args.min_entities, "loss_cap": args.loss_cap,
        "n_rows": int(len(ent)), "n_entities": int(len(set(ent.tolist()))), "n_regions": len(part.regions),
        "models": loss_cols, "lambdas": lams,
        "assumptions": list(reg.assumptions) + ["certified model frozen and fitted on entities disjoint from CAL"],
        "not_claimed": ["individual-case outcome guarantees", "any threshold on risk/error correlation"],
        "regions": regions,
    }

    if args.route:
        new = read_csv(Path(args.route))
        need(new, levels, args.route)
        hn = {lv: np.asarray(new[lv], dtype=str) for lv in levels}
        lab = part.labels(hn)
        counts, cards = {}, []
        for m in loss_cols:
            rel = []
            for i in range(len(lab)):
                c = reg.route(m, {"region": lab[i], "global": "all"}).to_dict()
                rel.append(c["local_relation"])
                if args.row_cards:
                    cards.append({"row": i, "model": m, **{k: c[k] for k in ("level", "key", "estimate", "lower",
                                                                         "upper", "local_relation")}})
            counts[m] = {r: rel.count(r) for r in sorted(set(rel))}
        result["routed"] = {"n_rows": int(len(lab)), "relation_counts": counts}
        if args.row_cards:
            p = Path(args.row_cards)
            p.write_text("\n".join(json.dumps(c, default=_jd) for c in cards) + "\n")
            result["routed"]["row_cards_written_locally"] = str(p)

    md = [f"# Pythia local-risk certificate (plugin {plugin_version()})", "",
          f"- Claim tier: **{result['claim_tier']}**",
          f"- Simultaneous level: 1 − {args.alpha} over {reg.n_charged_tests} charged tests "
          f"(per-tail α = {fmt(reg.alpha_per_tail_node_model, 6)})",
          f"- Calibration: {result['n_rows']} rows, {result['n_entities']} entities, "
          f"{result['n_regions']} regions (min {args.min_entities} entities each)", "",
          "| Model | Region | Entities | Estimate | Interval | Relation | Λ* |", "|---|---|---|---|---|---|---|"]
    for r in regions:
        md.append(f"| {r['model']} | {r['region']} | {r['n_entities']} | {fmt(r['estimate'])} | "
                  f"[{fmt(r['lower'])}, {fmt(r['upper'])}] | {r.get('relation', 'global')} | "
                  f"{fmt(r.get('tipping_lambda'), 2)} |")
    if "routed" in result:
        md += ["", "## Routed cases", "", "```json", json.dumps(result["routed"]["relation_counts"], indent=2), "```"]
    md += ["", "## Assumptions", ""] + [f"- {a}" for a in result["assumptions"]]
    md += ["", "Not claimed: individual-case guarantees; correlation thresholds.", ""]
    emit(args, result, "\n".join(md))


# ----------------------------------------------------------------------------- nli / table-floor
def _labels(cols, args, path):
    need(cols, [args.cell_col, args.y_col], path)
    cells = np.asarray(cols[args.cell_col], dtype=str)
    y_raw = np.asarray(cols[args.y_col], dtype=str)
    classes = sorted(set(y_raw.tolist()))
    y = np.searchsorted(np.asarray(classes), y_raw)
    return cells, y, classes


def _cell_guard(args, cells, kind):
    _, cnt = np.unique(cells, return_counts=True)
    if len(cells) < 2 * MIN_ENTITIES_FLOOR:
        refuse(args, kind, [f"only {len(cells)} rows; need at least {2 * MIN_ENTITIES_FLOOR}"])
    keep = np.isin(cells, np.unique(cells)[cnt >= 2])
    return keep


def cmd_table_floor(args) -> None:
    if not args.attest_independent_rows:
        refuse(args, "table-floor", ["--attest-independent-rows missing: labels must be independent given the "
                                     "table design (one row per entity, or entities independent)"])
    cols = read_csv(Path(args.data))
    cells, y, classes = _labels(cols, args, args.data)
    keep = _cell_guard(args, cells, "table-floor")
    cells, y = cells[keep], y[keep]
    K = len(classes)
    tL, tU, tQ = table_box(cells, y, K, args.alpha / 2)
    mL, mU, mQ = marginal_box(y, K, args.alpha / 2)
    result = {"status": "certified", "certificate": "table-floor", "plugin_version": plugin_version(),
              "claim_tier": "conditional on the design (in-sample floor of the observed rows)",
              "units": "Brier/Gini g = 1 - |p|^2 (multiply by 1/2 for half-Brier)",
              "n_rows_used": int(len(y)), "n_cells": int(len(np.unique(cells))), "classes": len(classes),
              "alpha": args.alpha, "table_floor": [tL, tU], "table_point": tQ, "marginal_floor": [mL, mU],
              "table_information": [max(mL - tU, 0.), max(mU - tL, 0.)],
              "assumptions": ["labels independent given the design", "cells fixed without labels"]}
    md = (f"# Pythia table floor (plugin {plugin_version()})\n\n- Claim tier: **{result['claim_tier']}**\n"
          f"- Rows used: {len(y)} in {result['n_cells']} cells (singleton cells excluded)\n\n"
          f"| Quantity | Interval (level 1 − {args.alpha}) |\n|---|---|\n"
          f"| Table floor G+V | [{fmt(tL)}, {fmt(tU)}] |\n| Marginal floor | [{fmt(mL)}, {fmt(mU)}] |\n"
          f"| Table information | [{fmt(result['table_information'][0])}, {fmt(result['table_information'][1])}] |\n")
    emit(args, result, md)


def cmd_nli(args) -> None:
    reasons = []
    if not args.attest_heldout_probs:
        reasons.append("--attest-heldout-probs missing: probability columns must come from a model fitted "
                       "without these rows' labels (cross-fitted or frozen)")
    if not args.attest_independent_rows:
        reasons.append("--attest-independent-rows missing: labels independent given the design")
    if reasons:
        refuse(args, "nli", reasons)
    cols = read_csv(Path(args.data))
    cells, y, classes = _labels(cols, args, args.data)
    pcols = [c for c in cols if c.startswith(args.prob_prefix)]
    if len(pcols) != len(classes):
        refuse(args, "nli", [f"found {len(pcols)} probability columns with prefix '{args.prob_prefix}' but "
                             f"{len(classes)} classes {classes}; order them to match the sorted class labels"])
    Q = np.column_stack([np.asarray(cols[c], float) for c in pcols])
    if not np.isfinite(Q).all() or (Q < -1e-9).any() or np.abs(Q.sum(1) - 1).max() > 1e-3:
        refuse(args, "nli", ["probability columns must be finite, non-negative and sum to 1 per row"])
    keep = _cell_guard(args, cells, "nli")
    cells, y, Q = cells[keep], y[keep], Q[keep]
    K = len(classes)
    a = args.alpha / 4                                       # table, marginal, NLI, model ceiling
    tL, tU, tQ = table_box(cells, y, K, a)
    mL, mU, _ = marginal_box(y, K, a)
    nliL, info = nli_cohort_bound(cells, Q, y, a, concentration=args.concentration)
    cU, cpt = model_ceiling(Q, y, K, a)
    joint_upper = min(tU - nliL, cU)
    result = {"status": "certified", "certificate": "nli", "plugin_version": plugin_version(),
              "claim_tier": "conditional on the design; all intervals simultaneous at level 1 - alpha",
              "units": "Brier/Gini g = 1 - |p|^2", "alpha": args.alpha, "concentration": args.concentration,
              "n_rows_used": int(len(y)), "n_cells": int(len(np.unique(cells))), "classes": K,
              "nli_lower": nliL, "nli_rho": info.get("rho"), "table_floor": [tL, tU], "marginal_floor": [mL, mU],
              "model_ceiling": cU, "joint_floor_upper": joint_upper,
              "binding": "table − NLI" if tU - nliL <= cU else "model ceiling",
              "text_cond_MI_lower_nats": nliL / 2,
              "assumptions": ["labels independent given (table cell, text) design",
                              "probabilities fitted without these labels",
                              "NLI is a lower bound only: a zero lower bound is not evidence of no information"]}
    md = (f"# Pythia NLI certificate (plugin {plugin_version()})\n\n- Claim tier: **{result['claim_tier']}**\n"
          f"- Rows used: {len(y)} in {result['n_cells']} cells; concentration: {args.concentration}\n\n"
          f"| Quantity | Value |\n|---|---|\n| NLI lower bound (text information beyond table) | {fmt(nliL)} |\n"
          f"| Table floor G+V | [{fmt(tL)}, {fmt(tU)}] |\n| Model ceiling (held-out Brier) | {fmt(cU)} |\n"
          f"| **Joint (table+text) floor upper bound** | **{fmt(joint_upper)}** ({result['binding']}) |\n"
          f"| Conditional MI lower bound (nats, Brier proxy) | {fmt(nliL / 2)} |\n\n"
          "NLI is lower-only: it can certify that text adds information, never that it adds none.\n")
    emit(args, result, md)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(prog="pythia-cert", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", default="pythia_out", help="output folder (aggregate JSON + markdown)")
        p.add_argument("--alpha", type=float, default=.05)

    p = sub.add_parser("local-risk", help="local risk intervals for frozen models")
    common(p)
    p.add_argument("--cal", required=True, help="calibration CSV (entities disjoint from model fitting)")
    p.add_argument("--entity-col", required=True)
    p.add_argument("--levels", required=True, help="hierarchy columns, finest first, comma separated")
    p.add_argument("--loss-cols", required=True, help="one loss column per model, comma separated")
    p.add_argument("--loss-cap", type=float, default=1.0)
    p.add_argument("--min-entities", type=int, default=100)
    p.add_argument("--lambdas", default="1,1.25,1.5")
    p.add_argument("--route", help="optional CSV of new cases (same level columns) to route")
    p.add_argument("--row-cards", help="optional LOCAL path for per-row cards (row-level; keep private)")
    p.add_argument("--attest-outcome-blind-levels", action="store_true")
    p.add_argument("--attest-independent-entities", action="store_true")
    p.add_argument("--attest-loss-cap", action="store_true")
    p.add_argument("--attest-frozen-model", action="store_true")
    p.set_defaults(fn=cmd_local_risk)

    for name, fn in (("nli", cmd_nli), ("table-floor", cmd_table_floor)):
        p = sub.add_parser(name)
        common(p)
        p.add_argument("--data", required=True)
        p.add_argument("--cell-col", required=True, help="table cell (discretised table covariates)")
        p.add_argument("--y-col", required=True, help="class label")
        p.add_argument("--attest-independent-rows", action="store_true")
        if name == "nli":
            p.add_argument("--prob-prefix", default="p_", help="held-out class-probability columns prefix")
            p.add_argument("--concentration", default="safe_portfolio",
                           choices=["hoeffding", "bentkus", "empirical_bernstein", "heterogeneous_chernoff",
                                    "stratified_bentkus", "safe_portfolio"])
            p.add_argument("--attest-heldout-probs", action="store_true")
        p.set_defaults(fn=fn)

    p = sub.add_parser("version")
    p.set_defaults(fn=lambda a: print(json.dumps({"plugin": plugin_version(),
                                                  "engine_path": sys.modules["pythia"].__file__})))
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
