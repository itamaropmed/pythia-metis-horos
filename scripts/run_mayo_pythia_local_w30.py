"""Retrospective Pythia local-W30 certificate for two frozen Mayo arms.

The bounded loss is 1{|Y-f(X)| > 30}; Pythia trains no prediction or risk model.
XGBoost fitting below belongs to the Mayo/Horos replay and only recreates honest
development OOF and patient-disjoint later predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pythia.core.local_model_risk import build_local_model_risk_registry
from run_mayo_metis_case_risk import crossfit_and_holdout, procedure_family, split_indices


def _edges(values: np.ndarray, bins: int) -> np.ndarray:
    raw = np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1])
    return np.unique(raw[np.isfinite(raw)])


def _labels(frame: pd.DataFrame, indices: np.ndarray, pred_a: np.ndarray,
            pred_b: np.ndarray, pred_edges: np.ndarray,
            disagreement_edges: np.ndarray) -> dict[str, np.ndarray]:
    family = frame.iloc[indices].primary_procedure_name.map(procedure_family).to_numpy()
    pbin = np.digitize((pred_a + pred_b) / 2, pred_edges).astype(str)
    dbin = np.digitize(np.abs(pred_a - pred_b), disagreement_edges).astype(str)
    return {
        "family_prediction_disagreement": np.char.add(
            np.char.add(np.char.add(family.astype(str), "|"), pbin), np.char.add("|", dbin)
        ),
        "family_prediction": np.char.add(np.char.add(family.astype(str), "|"), pbin),
        "prediction": pbin,
        "global": np.repeat("all", len(indices)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frame = pd.read_parquet(args.derived / "df.parquet")
    xa = pd.read_parquet(args.derived / "XA.parquet").to_numpy(np.float32)
    xb = pd.read_parquet(args.derived / "XB.parquet").to_numpy(np.float32)
    y = frame.case_actual_duration_time.to_numpy(float)
    groups = frame.patient_durable_id.astype(str).to_numpy()
    train, heldout, cutoff = split_indices(frame)
    oof_a, test_a = crossfit_and_holdout(xa, y, groups, train, heldout, args.seed)
    oof_b, test_b = crossfit_and_holdout(xb, y, groups, train, heldout, args.seed)
    names = ("basic_xgboost", "xgboost_plus_11_llm")
    dev_pred = {names[0]: oof_a, names[1]: oof_b}
    later_pred = {names[0]: test_a, names[1]: test_b}

    mean_dev = (oof_a + oof_b) / 2
    disagreement_dev = np.abs(oof_a - oof_b)
    pred_edges = _edges(mean_dev, 5)
    disagreement_edges = _edges(disagreement_dev, 3)
    dev_hierarchy = _labels(
        frame, train, oof_a, oof_b, pred_edges, disagreement_edges,
    )
    later_hierarchy = _labels(
        frame, heldout, test_a, test_b, pred_edges, disagreement_edges,
    )
    dev_loss = {
        name: (np.abs(y[train] - prediction) > 30).astype(float)
        for name, prediction in dev_pred.items()
    }
    registry = build_local_model_risk_registry(
        dev_loss, dev_hierarchy, groups[train], loss_cap=1., alpha=.05,
        min_entities=100, outcome_blind_hierarchy=True,
        independent_entities=True, loss_cap_attested=True,
    )

    # Freeze all routed certificates before revealing later outcomes.
    cards = {name: [] for name in names}
    for row in range(len(heldout)):
        keys = {level: values[row] for level, values in later_hierarchy.items()}
        for name in names:
            cards[name].append(registry.route(name, keys).to_dict())

    # Retrospective temporal transport audit.  These outcomes do not alter cards.
    audit = {}
    for name in names:
        later_loss = (np.abs(y[heldout] - later_pred[name]) > 30).astype(float)
        grouped: dict[tuple[str, str], list[int]] = {}
        for row, card in enumerate(cards[name]):
            grouped.setdefault((str(card["level"]), str(card["key"])), []).append(row)
        cells = []
        for (level, key), rows in grouped.items():
            card = cards[name][rows[0]]
            cells.append({
                "level": level, "key": key, "later_rows": len(rows),
                "development_estimate": card["estimate"],
                "development_interval": [card["lower"], card["upper"]],
                "later_observed_w30_miss": float(later_loss[rows].mean()),
                "relation": card["local_relation"],
            })
        audit[name] = {
            "later_observed_w30_miss": float(later_loss.mean()),
            "relation_counts": {
                relation: sum(c["local_relation"] == relation for c in cards[name])
                for relation in sorted({c["local_relation"] for c in cards[name]})
            },
            "support_level_counts": {
                level: sum(c["level"] == level for c in cards[name])
                for level in registry.level_order
            },
            "median_interval_width": float(np.median([
                c["upper"] - c["lower"] for c in cards[name]
            ])),
            "cell_transport_diagnostic": cells,
        }
    result = {
        "status": "retrospective temporal transport audit; not prospective confirmation",
        "claim_tier": registry.claim_tier,
        "estimand": "conditional case-weighted probability that frozen-model absolute error exceeds 30 minutes",
        "pythia_trains_models": False,
        "natural_loss_cap": 1.0,
        "split": {
            "cutoff": str(cutoff), "development_rows": len(train),
            "later_rows": len(heldout),
            "patient_overlap": int(len(set(groups[train]) & set(groups[heldout]))),
        },
        "alpha_family": registry.alpha_family,
        "alpha_per_tail_node_model": registry.alpha_per_tail_node_model,
        "models": audit,
        "limitations": [
            "transport from development to the later temporal cohort requires stability",
            "intervals target cell-average risk, not an individual outcome guarantee",
            "the XGBoost arms are reconstructed by Horos; Pythia only consumes frozen losses",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"], "split": result["split"],
        "models": {name: {key: value for key, value in row.items()
                           if key != "cell_transport_diagnostic"}
                   for name, row in audit.items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
