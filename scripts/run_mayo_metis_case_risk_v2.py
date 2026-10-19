"""Leakage-controlled comparison of Metis cell-map v1 and learned-risk v2.

All v2 candidate selection, isotonic calibration, and fitting use development
OOF errors only.  Held-out outcomes are touched only after both v1 and v2 cards
have been created for every later patient-disjoint case.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from horos.models.case_risk import fit_risk_forecasters
from metis.case_risk import fit_case_risk_map
from run_mayo_metis_case_risk import (
    category_summary, crossfit_and_holdout, procedure_family, split_indices,
)


FAMILIES = (
    "extraction", "revision", "af_pvi", "vt_pvc", "other_ablation",
    "implant", "diagnostic", "other",
)
SAFE_CONTEXT = (
    "scheduled_room_minutes", "scheduled_setup_lead_minutes",
    "scheduled_cleanup_minutes", "case_num_procedures", "case_num_panels",
    "case_num_providers", "patient_age", "bmi", "is_redo",
)


def risk_features(
    frame: pd.DataFrame,
    indices: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    llm: pd.DataFrame,
    *,
    medians: np.ndarray | None = None,
    auxiliary: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a fixed pre-outcome risk representation; fit imputation on dev."""
    part = frame.iloc[indices]
    family = part.primary_procedure_name.map(procedure_family).to_numpy()
    numeric = []
    for column in SAFE_CONTEXT:
        if column in part:
            numeric.append(pd.to_numeric(part[column], errors="coerce").to_numpy(float))
        else:
            numeric.append(np.full(len(part), np.nan))
    base = np.column_stack([
        pred_a, pred_b, (pred_a + pred_b) / 2., np.abs(pred_a - pred_b),
        np.minimum(pred_a, pred_b), np.maximum(pred_a, pred_b), *numeric,
    ])
    llm_values = llm.iloc[indices].to_numpy(float)
    onehot = np.column_stack([(family == value).astype(float) for value in FAMILIES])
    extra = [] if not auxiliary else [np.asarray(value, float) for value in auxiliary.values()]
    x = np.column_stack([base, *extra, llm_values, onehot])
    if medians is None:
        medians = np.nanmedian(x, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.)
    x = np.where(np.isfinite(x), x, medians)
    return x, medians


def auxiliary_predictions(features: np.ndarray, y: np.ndarray, groups: np.ndarray,
                          train: np.ndarray, heldout: np.ndarray,
                          seed: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Diverse honest predictions used only as arrival-time uncertainty evidence."""
    designs = {
        "hgb_absolute": HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=240, learning_rate=.05,
            max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=2.,
            random_state=seed,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=180, min_samples_leaf=8, max_features=.6,
            n_jobs=-1, random_state=seed,
        ),
    }
    oof, test = {}, {}
    fold_spec = list(GroupKFold(3).split(features[train], y[train], groups[train]))
    for name, estimator in designs.items():
        pred = np.empty(len(train), float)
        for fit, val in fold_spec:
            model = clone(estimator).fit(features[train][fit], y[train][fit])
            pred[val] = model.predict(features[train][val])
        final = clone(estimator).fit(features[train], y[train])
        oof[name] = pred
        test[name] = final.predict(features[heldout])
    return oof, test


def detailed_metrics(cards: list[dict], actual: np.ndarray,
                     predictions: dict[str, np.ndarray]) -> dict:
    rows = []
    for i, card in enumerate(cards):
        for risk in card["model_risks"]:
            name = risk["model"]
            error = abs(float(actual[i]) - float(predictions[name][i]))
            rows.append({
                "model": name, "risk": float(risk["risk_value"]),
                "probability": float(risk["probability_over_tolerance"]),
                "error": error, "miss": float(error > 30.),
            })
    table = pd.DataFrame(rows)
    return {
        name: {
            "risk_mae": float(np.mean(np.abs(part.risk - part.error))),
            "risk_rmse": float(np.sqrt(np.mean((part.risk - part.error) ** 2))),
            "probability_brier": float(np.mean((part.probability - part.miss) ** 2)),
            "risk_mean_bias": float(np.mean(part.risk - part.error)),
        }
        for name, part in table.groupby("model", sort=True)
    }


def paired_cluster_bootstrap(
    v1_cards: list[dict],
    v2_cards: list[dict],
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
    patients: np.ndarray,
    *,
    reps: int = 1000,
    seed: int = 20261018,
) -> dict:
    """Paired intervals resampling later patients, preserving repeated cases."""
    patients = np.asarray(patients).astype(str)
    unique = np.unique(patients)
    blocks = [np.where(patients == patient)[0] for patient in unique]
    rng = np.random.default_rng(seed)
    output = {}
    for model_position, name in enumerate(predictions):
        risk1 = np.asarray([card["model_risks"][model_position]["risk_value"]
                            for card in v1_cards], float)
        risk2 = np.asarray([card["model_risks"][model_position]["risk_value"]
                            for card in v2_cards], float)
        prob1 = np.asarray([card["model_risks"][model_position]["probability_over_tolerance"]
                            for card in v1_cards], float)
        prob2 = np.asarray([card["model_risks"][model_position]["probability_over_tolerance"]
                            for card in v2_cards], float)
        error = np.abs(np.asarray(actual, float) - np.asarray(predictions[name], float))
        miss = (error > 30.).astype(float)
        draws = {"spearman_change": [], "risk_mae_change": [], "brier_change": []}
        for _ in range(reps):
            ix = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
            rho1 = spearmanr(risk1[ix], error[ix]).statistic
            rho2 = spearmanr(risk2[ix], error[ix]).statistic
            draws["spearman_change"].append(float(rho2 - rho1))
            draws["risk_mae_change"].append(float(
                np.mean(np.abs(risk2[ix] - error[ix]))
                - np.mean(np.abs(risk1[ix] - error[ix]))
            ))
            draws["brier_change"].append(float(
                np.mean((prob2[ix] - miss[ix]) ** 2)
                - np.mean((prob1[ix] - miss[ix]) ** 2)
            ))
        output[name] = {
            metric: {
                "point": float(np.mean(values)),
                "ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))],
                "improvement_probability": float(
                    np.mean(np.asarray(values) > 0) if metric == "spearman_change"
                    else np.mean(np.asarray(values) < 0)
                ),
            }
            for metric, values in draws.items()
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--id-salt", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--auxiliary-evidence", action="store_true",
                        help="add diverse Horos predictions as risk-only evidence")
    args = parser.parse_args()

    frame = pd.read_parquet(args.derived / "df.parquet")
    xa_frame = pd.read_parquet(args.derived / "XA.parquet")
    xb_frame = pd.read_parquet(args.derived / "XB.parquet")
    xa = xa_frame.to_numpy(np.float32)
    xb = xb_frame.to_numpy(np.float32)
    llm_columns = [column for column in xb_frame if column.startswith("llm_")]
    if len(llm_columns) != 11:
        raise RuntimeError(f"expected 11 LLM features, found {len(llm_columns)}")
    llm = xb_frame[llm_columns]
    y = frame.case_actual_duration_time.to_numpy(float)
    groups = frame.patient_durable_id.astype(str).to_numpy()
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
        min_entities=100, version="metis-case-risk-v2-support-map",
        provenance={"selection": "development-only", "heldout_outcomes": "not read"},
    )

    aux_train: dict[str, np.ndarray] = {}
    aux_test: dict[str, np.ndarray] = {}
    if args.auxiliary_evidence:
        for arm_name, values in (("tabular", xa), ("plus11", xb)):
            arm_oof, arm_test = auxiliary_predictions(
                values, y, groups, train, heldout, args.seed + len(aux_train) * 31,
            )
            for method in arm_oof:
                aux_train[f"{arm_name}_{method}"] = arm_oof[method]
                aux_test[f"{arm_name}_{method}"] = arm_test[method]
        a_stack = np.column_stack([oof_a, aux_train["tabular_hgb_absolute"],
                                   aux_train["tabular_extra_trees"]])
        b_stack = np.column_stack([oof_b, aux_train["plus11_hgb_absolute"],
                                   aux_train["plus11_extra_trees"]])
        at_stack = np.column_stack([test_a, aux_test["tabular_hgb_absolute"],
                                    aux_test["tabular_extra_trees"]])
        bt_stack = np.column_stack([test_b, aux_test["plus11_hgb_absolute"],
                                    aux_test["plus11_extra_trees"]])
        aux_train.update({"tabular_spread": a_stack.std(axis=1),
                          "plus11_spread": b_stack.std(axis=1),
                          "all_spread": np.column_stack([a_stack, b_stack]).std(axis=1)})
        aux_test.update({"tabular_spread": at_stack.std(axis=1),
                         "plus11_spread": bt_stack.std(axis=1),
                         "all_spread": np.column_stack([at_stack, bt_stack]).std(axis=1)})
    x_train, medians = risk_features(
        frame, train, oof_a, oof_b, llm, auxiliary=aux_train,
    )
    x_test, _ = risk_features(
        frame, heldout, test_a, test_b, llm, medians=medians, auxiliary=aux_test,
    )
    errors = {name: np.abs(y[train] - train_pred[name]) for name in names}
    forecasters = fit_risk_forecasters(
        x_train, errors, groups[train], tolerance=30., folds=5, seed=args.seed,
    )
    v2_test = {name: forecasters[name].predict(x_test) for name in names}
    floors = {names[0]: (29.29, "calibrated"), names[1]: (28.84959960434628, "calibrated")}

    # Freeze every risk card before the reveal block below.
    case_ids = frame.iloc[heldout].case_durable_id.to_numpy()
    v1_cards, v2_cards = [], []
    for i, case_id in enumerate(case_ids):
        predictions = {name: test_pred[name][i] for name in names}
        common = dict(stratum=test_strata[i], id_salt=args.id_salt,
                      information_set_floors=floors)
        v1_cards.append(risk_map.score(case_id, predictions, **common).to_dict())
        v2_cards.append(risk_map.score(
            case_id, predictions,
            point_risk_overrides={name: v2_test[name][0][i] for name in names},
            probability_overrides={name: v2_test[name][1][i] for name in names},
            risk_method="crossfitted_horos_forecaster_v2", **common,
        ).to_dict())

    selection = {
        name: {
            "selected_error_candidate": model.selected_error_candidate,
            "selected_probability_candidate": model.selected_probability_candidate,
            "development_oof_candidates": [
                {
                    "name": row.name,
                    "spearman": row.spearman,
                    "mean_absolute_calibration_error": row.mean_absolute_calibration_error,
                    "mean_bias": row.mean_bias,
                    "brier": None if not np.isfinite(row.brier) else row.brier,
                }
                for row in model.candidate_results
            ],
        }
        for name, model in forecasters.items()
    }

    # Outcome reveal begins here. Nothing above uses y[heldout].
    actual = y[heldout]
    v1_summary = category_summary(v1_cards, actual, test_pred)
    v2_summary = category_summary(v2_cards, actual, test_pred)
    v1_detail = detailed_metrics(v1_cards, actual, test_pred)
    v2_detail = detailed_metrics(v2_cards, actual, test_pred)
    comparison = {}
    for name in names:
        comparison[name] = {
            "spearman_change": (
                v2_summary[name]["spearman_risk_vs_absolute_error"]
                - v1_summary[name]["spearman_risk_vs_absolute_error"]
            ),
            "risk_mae_change": v2_detail[name]["risk_mae"] - v1_detail[name]["risk_mae"],
            "brier_change": (
                v2_detail[name]["probability_brier"] - v1_detail[name]["probability_brier"]
            ),
        }
    bootstrap = paired_cluster_bootstrap(
        v1_cards, v2_cards, actual, test_pred, groups[heldout], seed=args.seed + 20261018,
    )
    output = {
        "status": "retrospective_replay; v2 selected_without_heldout_outcomes",
        "split": {"cutoff": cutoff, "development_rows": len(train),
                  "heldout_rows": len(heldout), "patient_overlap": 0},
        "risk_features": {
            "count": int(x_train.shape[1]), "llm_features": len(llm_columns),
            "auxiliary_evidence": list(aux_train),
            "imputation": "development medians only",
            "categorical_encoding": "fixed predeclared procedure-family vocabulary",
        },
        "development_only_selection": selection,
        "v1": {"category_evaluation": v1_summary, "forecast_metrics": v1_detail},
        "v2": {"category_evaluation": v2_summary, "forecast_metrics": v2_detail},
        "v2_minus_v1": comparison,
        "patient_clustered_paired_bootstrap": bootstrap,
        "claim": "diagnostic comparison, not fresh confirmation",
        "privacy": "aggregate output only",
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
