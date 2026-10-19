"""Command-line interface for deterministic certificate generation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .api import certify
from .contracts import FoldMatrix, Representation, TaskCard
from .core.guarded import load_guard
from .folds import build_grouped_temporal_fold_manifest, build_mayo_fold_manifest
from .incumbent import audit_incumbent
from .io import load_certificate_config
from .ladder import evaluate_ladder
from .provenance import audit_note_provenance, recut_note_bundles
from horos.pipeline import prepare_study


def _hash(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode())
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def _load_mayo(config_path: Path, arm: str, *, bootstrap: bool = False) -> tuple[TaskCard, FoldMatrix, Representation, object, dict]:
    cfg = yaml.safe_load(config_path.read_text())
    root = config_path.parent
    der = root / cfg["paths"]["derived_mayo"]
    frame_path = der / "df.parquet"
    pred_path = der / f"P{arm}.npy"
    feature_path = der / cfg["mayo"]["arms"][arm]
    df = pd.read_parquet(frame_path)
    p = np.load(pred_path)
    y = df[cfg["mayo"]["target"]].to_numpy(float)
    groups = df[cfg["mayo"]["group"]].astype(str).to_numpy()
    best = p[int(np.mean(np.abs(y[None, :] - p), axis=1).argmin())]
    labels = np.asarray(pd.qcut(best, 5, labels=False, duplicates="drop"))
    fold_matrix = FoldMatrix(
        y=y, P_oof=p, names=tuple(f"m{i}" for i in range(p.shape[0])),
        families=("hgb", "xgb", "extra_trees", "hgb", "xgb", "extra_trees"),
        fold_id=np.full(y.size, -1), groups=groups,
        row_ids=df.get("case_durable_id", pd.Series(np.arange(y.size))).to_numpy(),
        meta={
            "source": "frozen_mayo_oof", "arm": arm, "n": int(y.size),
            "cohort_hash": _hash([frame_path, pred_path]),
            "bootstrap_reps": int(cfg["pythia"]["bootstrap"]["reps"]) if bootstrap else 0,
            "bootstrap_seed": int(cfg["pythia"]["bootstrap"].get("seed", 0)),
        },
    )
    task = TaskCard(
        target=cfg["mayo"]["target"], loss=cfg["pythia"]["loss"],
        group=cfg["mayo"]["group"], time=cfg["mayo"]["time"],
        as_of_rule="features available before scheduled_in_room",
        operational_unit="minutes MAE",
    )
    representation = Representation(
        name=f"Mayo arm {arm}", feature_names=(),
        provenance={"arm_file": feature_path.name},
        source_hash=_hash([feature_path]),
    )
    guard = load_guard(der)
    return task, fold_matrix, representation, guard, {"predicted_duration_quintile": labels}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pythia")
    sub = parser.add_subparsers(dest="command", required=True)
    cert = sub.add_parser("certify", help="emit a JSON certificate")
    cert.add_argument("config", type=Path)
    cert.add_argument("--arm", choices=("A", "B", "C"), default="B")
    cert.add_argument("--output", type=Path)
    ladder = sub.add_parser("mayo-ladder", help="compare frozen Mayo information sets A/B/C")
    ladder.add_argument("config", type=Path)
    ladder.add_argument("--output", type=Path)
    audit = sub.add_parser("audit-incumbent", help="verify an incumbent prediction artifact")
    audit.add_argument("predictions", type=Path)
    audit.add_argument("--target-column", required=True)
    audit.add_argument("--prediction-column", required=True)
    audit.add_argument("--row-id-column", default="case_durable_id")
    audit.add_argument("--group-column", default="patient_durable_id")
    audit.add_argument("--fold-column", default="fold_id")
    audit.add_argument("--expected-mae", type=float, default=31.2623)
    audit.add_argument("--tolerance", type=float, default=1e-4)
    audit.add_argument("--manifest", type=Path)
    audit.add_argument("--manifest-role", default="development")
    notes = sub.add_parser("audit-notes", help="verify note cutoff and patient-fold isolation")
    notes.add_argument("bundles", type=Path)
    notes.add_argument("cases", type=Path)
    notes.add_argument("--case-id-column", default="case_durable_id")
    notes.add_argument("--patient-column", default="patient_durable_id")
    notes.add_argument("--scheduled-column", default="scheduled_in_room")
    notes.add_argument("--fold-column", default="fold_id")
    notes.add_argument("--cutoff-days", type=int, default=2)
    notes.add_argument("--cutoff-only", action="store_true", help="do not require fold evidence")
    recut = sub.add_parser("recut-notes", help="apply the strict cutoff to a capped note bundle")
    recut.add_argument("bundles", type=Path)
    recut.add_argument("cases", type=Path)
    recut.add_argument("output", type=Path)
    recut.add_argument("--case-id-column", default="case_durable_id")
    recut.add_argument("--scheduled-column", default="scheduled_in_room")
    recut.add_argument("--cutoff-days", type=int, default=2)
    recut.add_argument("--history-floor-days", type=int, default=548)
    folds = sub.add_parser("build-mayo-folds", help="build the canonical Mayo split/fold manifest")
    folds.add_argument("cases", type=Path)
    folds.add_argument("output", type=Path)
    folds.add_argument("--test-fraction", type=float, default=0.20)
    folds.add_argument("--n-folds", type=int, default=5)
    generic_folds = sub.add_parser("build-folds", help="build a generic grouped temporal manifest")
    generic_folds.add_argument("data", type=Path)
    generic_folds.add_argument("output", type=Path)
    generic_folds.add_argument("--row-id", required=True)
    generic_folds.add_argument("--group", required=True)
    generic_folds.add_argument("--time", required=True)
    generic_folds.add_argument("--target", required=True)
    generic_folds.add_argument("--test-fraction", type=float, default=0.20)
    generic_folds.add_argument("--n-folds", type=int, default=5)
    prepare = sub.add_parser("prepare", help="prepare a dataset-agnostic study from YAML")
    prepare.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_study(args.config)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["status"] == "prepared" else 2
        if args.command == "build-folds":
            result = build_grouped_temporal_fold_manifest(
                args.data, args.output, case_id_column=args.row_id,
                patient_column=args.group, time_column=args.time,
                target_column=args.target, test_fraction=args.test_fraction,
                n_folds=args.n_folds,
            )
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
            return 0 if result.accepted else 2
        if args.command == "build-mayo-folds":
            result = build_mayo_fold_manifest(
                args.cases, args.output,
                test_fraction=args.test_fraction, n_folds=args.n_folds,
            )
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
            return 0 if result.accepted else 2
        if args.command == "recut-notes":
            result = recut_note_bundles(
                args.bundles, args.cases, args.output,
                case_id_column=args.case_id_column,
                scheduled_column=args.scheduled_column,
                cutoff_days=args.cutoff_days,
                history_floor_days=args.history_floor_days,
            )
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
            return 0
        if args.command == "audit-notes":
            result = audit_note_provenance(
                args.bundles, args.cases,
                case_id_column=args.case_id_column,
                patient_column=args.patient_column,
                scheduled_column=args.scheduled_column,
                fold_column=args.fold_column,
                cutoff_days=args.cutoff_days,
                require_cluster_isolation=not args.cutoff_only,
            )
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
            return 0 if result.accepted else 2
        if args.command == "audit-incumbent":
            result = audit_incumbent(
                args.predictions, target_column=args.target_column,
                prediction_column=args.prediction_column,
                row_id_column=args.row_id_column, group_column=args.group_column,
                fold_column=args.fold_column, expected_mae=args.expected_mae,
                tolerance=args.tolerance,
                manifest_path=args.manifest, manifest_role=args.manifest_role,
            )
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
            return 0 if result.accepted else 2
        if args.command == "mayo-ladder":
            loaded = [_load_mayo(args.config.resolve(), arm) for arm in "ABC"]
            task, _, _, guard, _ = loaded[0]
            entries = [(representation, folds) for _, folds, representation, _, _ in loaded]
            slice_sets = [slices for _, _, _, _, slices in loaded]
            result = evaluate_ladder(task, entries, guard, slice_sets=slice_sets)
            payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
            if args.output:
                args.output.write_text(payload)
            else:
                print(payload, end="")
            return 0
        raw_config = yaml.safe_load(args.config.resolve().read_text())
        if isinstance(raw_config, dict) and "certificate" in raw_config:
            task, folds, representation, guard = load_certificate_config(args.config)
            slices = None
        else:
            task, folds, representation, guard, slices = _load_mayo(
                args.config.resolve(), args.arm, bootstrap=True
            )
        result = certify(task, folds, representation, guard, slices)
        payload = json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(payload)
        else:
            print(payload, end="")
        return 0 if result.status == "complete" else 2
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
