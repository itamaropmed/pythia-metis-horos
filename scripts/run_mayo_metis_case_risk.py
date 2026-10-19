"""Retrospective, privacy-preserving Mayo new-case risk simulation.

Two duration predictors are trained in Horos territory: tabular XGBoost and the
same XGBoost with the frozen 11 LLM concepts.  Metis is calibrated only from
development OOF residuals.  Held-out outcomes are revealed only after every
case risk card has been emitted in memory.

No row-level output is written unless --examples-output is supplied.  That
file contains salted case pseudonyms and no patient identifier or note text.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from metis.case_risk import fit_case_risk_map


MODEL_KWARGS = dict(
    objective="reg:absoluteerror", n_estimators=400, max_depth=6,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=8, reg_lambda=1.0, tree_method="hist", n_jobs=-1,
)


def procedure_family(value: object) -> str:
    text = str(value).lower()
    patterns = (
        ("extraction", ("extract", "removal", "explant")),
        ("revision", ("revision", "generator", "upgrade", "pocket")),
        ("af_pvi", ("af ", "fibrillation", "pvi", "pulmonary vein", "pfa")),
        ("vt_pvc", ("vt", "ventricular tach", "pvc")),
        ("other_ablation", ("ablat",)),
        ("implant", ("implant", "insert", "pacemaker", "icd", "crt", "leadless", "loop recorder")),
        ("diagnostic", ("study", "test", "diagnostic", "tilt", "cardiovers")),
    )
    return next((name for name, terms in patterns if any(term in text for term in terms)), "other")


def split_indices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:
    time = pd.to_datetime(frame["scheduled_in_room"], utc=True)
    patient = frame["patient_durable_id"].astype(str)
    cutoff = time.quantile(0.8)
    heldout_patients = set(patient[time >= cutoff])
    train = np.where((time < cutoff) & ~patient.isin(heldout_patients))[0]
    heldout = np.where((time >= cutoff) & patient.isin(heldout_patients))[0]
    if set(patient.iloc[train]) & set(patient.iloc[heldout]):
        raise RuntimeError("patient overlap in outer split")
    return train, heldout, str(cutoff)


def crossfit_and_holdout(
    features: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train: np.ndarray,
    heldout: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.empty(len(train), dtype=float)
    for fold, (fit_local, val_local) in enumerate(
        GroupKFold(5).split(features[train], y[train], groups[train])
    ):
        model = XGBRegressor(random_state=seed + fold, **MODEL_KWARGS)
        model.fit(features[train][fit_local], y[train][fit_local])
        oof[val_local] = model.predict(features[train][val_local])
    final = XGBRegressor(random_state=seed, **MODEL_KWARGS)
    final.fit(features[train], y[train])
    return oof, final.predict(features[heldout])


def category_summary(cards: list[dict], actual: np.ndarray, predictions: dict[str, np.ndarray]) -> dict:
    rows = []
    for i, card in enumerate(cards):
        for risk in card["model_risks"]:
            name = risk["model"]
            rows.append({
                **risk,
                "actual_abs_error": abs(float(actual[i]) - float(predictions[name][i])),
            })
    table = pd.DataFrame(rows)
    out = {}
    for name, model_rows in table.groupby("model", sort=True):
        order = [label for label in ("low", "moderate", "high", "very_high")
                 if label in set(model_rows.risk_category)]
        categories = []
        for label in order:
            part = model_rows[model_rows.risk_category == label]
            categories.append({
                "risk_category": label, "n": int(len(part)),
                "mean_predicted_absolute_error": float(part.risk_value.mean()),
                "observed_mae_after_reveal": float(part.actual_abs_error.mean()),
                "observed_probability_over_30": float((part.actual_abs_error > 30).mean()),
            })
        rho = spearmanr(model_rows.risk_value, model_rows.actual_abs_error).statistic
        out[name] = {
            "spearman_risk_vs_absolute_error": float(rho),
            "mean_predicted_absolute_error": float(model_rows.risk_value.mean()),
            "observed_mae": float(model_rows.actual_abs_error.mean()),
            "mean_probability_over_30": float(model_rows.probability_over_tolerance.mean()),
            "observed_probability_over_30": float((model_rows.actual_abs_error > 30).mean()),
            "support_level_counts": model_rows.support_level.value_counts().to_dict(),
            "categories": categories,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived", type=Path, required=True,
                        help="directory containing df.parquet, XA.parquet, XB.parquet")
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--examples-output", type=Path)
    parser.add_argument("--id-salt", required=True, help="private salt; never commit this value")
    parser.add_argument("--seed", type=int, default=0,
                        help="historical A/B XGBoost seed")
    args = parser.parse_args()

    frame = pd.read_parquet(args.derived / "df.parquet")
    xa = pd.read_parquet(args.derived / "XA.parquet").to_numpy(np.float32)
    xb = pd.read_parquet(args.derived / "XB.parquet").to_numpy(np.float32)
    y = frame["case_actual_duration_time"].to_numpy(float)
    groups = frame["patient_durable_id"].astype(str).to_numpy()
    train, heldout, cutoff = split_indices(frame)
    oof_a, test_a = crossfit_and_holdout(xa, y, groups, train, heldout, args.seed)
    oof_b, test_b = crossfit_and_holdout(xb, y, groups, train, heldout, args.seed)
    names = ("basic_xgboost", "xgboost_plus_11_llm")
    train_pred = {names[0]: oof_a, names[1]: oof_b}
    test_pred = {names[0]: test_a, names[1]: test_b}
    train_strata = frame.iloc[train].primary_procedure_name.map(procedure_family).to_numpy()
    test_strata = frame.iloc[heldout].primary_procedure_name.map(procedure_family).to_numpy()

    risk_map = fit_case_risk_map(
        y[train], train_pred, strata=train_strata, entity_ids=groups[train],
        min_entities=100, provenance={
            "calibration": "patient-grouped development OOF only",
            "outer_split": "patient-disjoint temporal 80th percentile",
            "features": "frozen XA/XB; XB adds 11 pre-existing LLM concepts",
        },
    )
    information_set_floors = {
        names[0]: (29.29, "calibrated"),
        names[1]: (28.84959960434628, "calibrated"),
    }

    # Arrival phase: no held-out outcome is passed into score().
    cards = []
    case_ids = frame.iloc[heldout].case_durable_id.to_numpy()
    for i, case_id in enumerate(case_ids):
        result = risk_map.score(
            case_id, {name: test_pred[name][i] for name in names},
            stratum=test_strata[i], id_salt=args.id_salt,
            information_set_floors=information_set_floors,
        )
        cards.append(result.to_dict())

    # Retrospective reveal phase, kept structurally after all cards are frozen.
    evaluation = category_summary(cards, y[heldout], test_pred)
    delta = np.abs(y[heldout] - test_a) - np.abs(y[heldout] - test_b)
    summary = {
        "status": "retrospective_poc_not_clinical_certification",
        "split": {"cutoff": cutoff, "development_rows": int(len(train)),
                  "heldout_rows": int(len(heldout)),
                  "development_patients": int(pd.Series(groups[train]).nunique()),
                  "heldout_patients": int(pd.Series(groups[heldout]).nunique()),
                  "patient_overlap": 0},
        "pythia_reference_floors": {
            "basic_xgboost_information_set": {"lower_bound_minutes": 29.29, "tier": "calibrated"},
            "xgboost_plus_11_llm_information_set": {"lower_bound_minutes": 28.84959960434628,
                                                      "tier": "calibrated"},
            "warning": "historical full-cohort information-set floors; not individual risk scores",
        },
        "metis_risk_map": {"version": risk_map.version,
                           "calibration_rows": risk_map.calibration_rows,
                           "calibration_entities": risk_map.calibration_entities,
                           "tolerance_minutes": risk_map.tolerance},
        "heldout_evaluation_after_reveal": evaluation,
        "paired_model_comparison": {
            "mean_mae_gain_plus_11_over_basic": float(delta.mean()),
            "plus_11_better_case_fraction": float((delta > 0).mean()),
        },
        "privacy": "aggregate summary only; no identifiers or notes",
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True))

    if args.examples_output:
        examples = []
        for i, card in enumerate(cards[:20]):
            row = {"anonymous_case_id": card["anonymous_case_id"], "models": []}
            for risk in card["model_risks"]:
                name = risk["model"]
                row["models"].append({
                    **risk,
                    "actual_absolute_error_after_reveal": float(
                        abs(float(y[heldout][i]) - float(test_pred[name][i]))
                    ),
                })
            examples.append(row)
        args.examples_output.parent.mkdir(parents=True, exist_ok=True)
        args.examples_output.write_text(json.dumps(examples, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
