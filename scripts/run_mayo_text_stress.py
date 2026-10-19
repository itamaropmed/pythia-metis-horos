"""Run content-free Mayo note perturbation controls through Pythia.

This is a retrospective/exploratory migration study.  It reproduces the
historical label-free TF-IDF/SVD construction on the full development matrix;
cross-representation claims therefore remain exploratory until the transform
is moved inside each fold.  No note text or row-level prediction is persisted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

from horos.represent import TextPerturbationSpec, perturb_texts
from pythia import FoldMatrix, Representation, RiskInterval, TaskCard, certify
from pythia.core.guarded import load_guard
from pythia.information_order import close_information_order, decompose_achieved_gain


DEFAULT_CONTROLS = (
    TextPerturbationSpec("identity", 0.0, 20260911),
    TextPerturbationSpec("character_shuffle", 1.0, 20260911),
    TextPerturbationSpec("token_shuffle", 1.0, 20260911),
    TextPerturbationSpec("span_mask", 0.5, 20260911),
    TextPerturbationSpec("case_dropout", 0.5, 20260911),
    TextPerturbationSpec("segment_dropout", 0.5, 20260911),
    TextPerturbationSpec("segment_duplicate", 0.5, 20260911),
)


def _models(seed: int):
    return (
        HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=250, learning_rate=0.05,
            max_depth=6, min_samples_leaf=20, l2_regularization=1.0,
            random_state=seed,
        ),
        ExtraTreesRegressor(
            n_estimators=180, min_samples_leaf=5, n_jobs=2, random_state=seed,
        ),
    )


def _oof(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    predictions = []
    for seed in (11, 112):
        codes = pd.factorize(groups)[0]
        permuted = np.random.default_rng(seed).permutation(codes.max() + 1)[codes]
        splits = list(GroupKFold(5).split(x, y, groups=permuted))
        for model in _models(seed):
            pred = np.empty(len(y), float)
            for train, test in splits:
                fitted = model.__class__(**model.get_params())
                fitted.fit(x[train], y[train])
                pred[test] = fitted.predict(x[test])
            predictions.append(pred)
    return np.vstack(predictions)


def _text_matrix(texts: np.ndarray) -> tuple[np.ndarray, dict]:
    vectorizer = TfidfVectorizer(
        max_features=30000, min_df=10, max_df=0.9, ngram_range=(1, 2),
        sublinear_tf=True, strip_accents="unicode",
    )
    z = vectorizer.fit_transform(texts)
    components = min(64, max(1, z.shape[1] - 1))
    svd = TruncatedSVD(n_components=components, random_state=0)
    matrix = svd.fit_transform(z)
    return matrix.astype(np.float32), {
        "tfidf_features": int(z.shape[1]), "svd_components": components,
        "svd_explained_variance": float(svd.explained_variance_ratio_.sum()),
        "fit_scope": "full retrospective matrix; label-free but transductive",
    }


def _certificate(
    name: str, x: np.ndarray, frame: pd.DataFrame, guard, provenance: dict,
    bootstrap_reps: int,
) -> tuple[dict, np.ndarray]:
    y = frame["case_actual_duration_time"].to_numpy(float)
    groups = frame["patient_durable_id"].astype(str).to_numpy()
    predictions = _oof(x, y, groups)
    best = predictions[np.mean(np.abs(y[None, :] - predictions), axis=1).argmin()]
    strata = np.asarray(pd.qcut(best, 5, labels=False, duplicates="drop"))
    folds = FoldMatrix(
        y=y, P_oof=predictions,
        names=("hgb_11", "extra_trees_11", "hgb_112", "extra_trees_112"),
        families=("hgb", "extra_trees", "hgb", "extra_trees"),
        fold_id=np.full(len(y), -1), groups=groups,
        row_ids=frame["case_durable_id"].to_numpy(),
        meta={"bootstrap_reps": bootstrap_reps, "bootstrap_seed": 20260911,
              "study_tier": "exploratory"},
    )
    result = certify(
        TaskCard("case_actual_duration_time", "mae", group="patient_durable_id",
                 operational_unit="minutes MAE"),
        folds,
        Representation(name=name, version="mayo-text-stress-v1", provenance=provenance),
        guard, {"predicted_duration_quintile": strata},
    )
    return result.to_dict(), predictions


def _risk_interval(certificate: dict) -> RiskInterval:
    sampling = certificate.get("intervals", {}).get("sampling_uncertainty", {})
    lower = sampling.get("bound", {}).get("lo", certificate["bound"])
    upper = sampling.get("L", {}).get("hi", certificate["achieved_loss"])
    estimate = float(np.clip(certificate["estimator"], lower, upper))
    return RiskInterval(
        float(lower), estimate, float(upper), "exploratory", simultaneous=False,
        source=certificate["representation"]["name"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("outputs/mayo_text_stress.json"))
    parser.add_argument("--bootstrap-reps", type=int, default=100)
    args = parser.parse_args()
    derived = args.root / "data/derived/mayo"
    frame = pd.read_parquet(derived / "df.parquet")
    tabular = pd.read_parquet(derived / "XA.parquet").to_numpy(np.float32)
    bundles = pd.read_parquet(derived / "note_bundles_recut.parquet")
    bundles = bundles.drop_duplicates("case_durable_id").set_index("case_durable_id")
    case_ids = frame["case_durable_id"].to_numpy()
    texts = bundles["note_text"].reindex(case_ids).fillna("").astype(str).to_numpy()
    counts = np.column_stack([
        bundles.get("note_chars", pd.Series(dtype=float)).reindex(case_ids).fillna(0).to_numpy(),
        bundles.get("n_pre_cutoff_notes_used", pd.Series(dtype=float)).reindex(case_ids).fillna(0).to_numpy(),
    ]).astype(np.float32)
    guard = load_guard(derived)
    results: dict[str, dict] = {}

    started = time.time()
    baseline, _ = _certificate(
        "tabular", tabular, frame, guard,
        {"information_set": "tabular", "deployable": True}, args.bootstrap_reps,
    )
    results["tabular"] = {"certificate": baseline}
    print(f"tabular L={baseline['achieved_loss']:.3f} B={baseline['bound']:.3f}", flush=True)

    for spec in DEFAULT_CONTROLS:
        perturbed, audit = perturb_texts(texts, case_ids, spec)
        text_matrix, text_meta = _text_matrix(perturbed)
        x = np.column_stack([tabular, text_matrix, counts]).astype(np.float32)
        key = "intact" if spec.kind == "identity" else f"{spec.kind}_{spec.fraction:g}"
        certificate, _ = _certificate(
            key, x, frame, guard,
            {"information_set": "tabular+notes", "deployable": spec.kind == "identity",
             "perturbation": audit.to_dict(), "text_transform": text_meta},
            args.bootstrap_reps,
        )
        results[key] = {"certificate": certificate, "perturbation": audit.to_dict(),
                        "text_transform": text_meta}
        print(f"{key} L={certificate['achieved_loss']:.3f} B={certificate['bound']:.3f}", flush=True)

    intervals = {name: _risk_interval(value["certificate"]) for name, value in results.items()}
    edges = [("intact", name) for name in results if name not in {"intact", "tabular"}]
    edges += [(name, "tabular") for name in results if name not in {"intact", "tabular"}]
    edges.append(("intact", "tabular"))
    order = close_information_order(intervals, edges)
    decompositions = {}
    if order.status == "complete":
        for name in results:
            if name in {"intact", "tabular"}:
                continue
            decompositions[name] = decompose_achieved_gain(
                order, richer="intact", poorer=name,
                achieved_richer=results["intact"]["certificate"]["achieved_loss"],
                achieved_poorer=results[name]["certificate"]["achieved_loss"],
            )
    payload = {
        "status": "complete", "study_tier": "exploratory",
        "n": len(frame), "n_with_text": int(np.sum(texts != "")),
        "elapsed_seconds": time.time() - started,
        "limitations": [
            "TF-IDF/SVD is fitted on the full retrospective feature matrix without labels",
            "bootstrap intervals are pointwise, not simultaneous",
            "Foundation/transformer representation is not yet connected",
        ],
        "results": results, "information_order": order.to_dict(),
        "perturbation_decompositions": decompositions,
    }
    output = args.output if args.output.is_absolute() else args.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote aggregate-only report to {output}", flush=True)


if __name__ == "__main__":
    main()
