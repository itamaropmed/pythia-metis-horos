"""Synthetic regressions for split isolation in the strict Mayo evaluator.

These tests never load clinical records. One test intentionally reproduces the
remaining cohort-wide rare-category preprocessing defect in the legacy builder;
it passes when that vulnerability is present and must be converted to a
split-safe invariance assertion when the builder is repaired.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
STRICT = ROOT / "research/mayo_strict_20261024/mayo_strict.py"
LEGACY_BUILDER = ROOT / "scripts/legacy_mayo/build.py"


def _load_helpers():
    tree = ast.parse(STRICT.read_text(), filename=str(STRICT))
    chosen = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"centre_from_training", "ridge_dense", "strict_tabular_matrix",
                          "validate_feature_alignment", "paired_cluster_bootstrap"}
    ]
    ns = {"np": np, "pd": pd}
    exec(compile(ast.Module(body=chosen, type_ignores=[]), str(STRICT), "exec"), ns)
    return ns


def test_heldout_outcomes_do_not_change_fit_group_center_or_reader():
    helpers = _load_helpers()
    rng = np.random.default_rng(70)
    X = rng.normal(size=(40, 6))
    groups = np.array(["A"] * 20 + ["B"] * 20)
    train = np.array([True] * 12 + [False] * 8 + [True] * 12 + [False] * 8)
    y0 = rng.normal(size=40)
    y1 = y0.copy()
    y1[~train] += 500.0

    center0 = helpers["centre_from_training"](y0, groups, train)
    center1 = helpers["centre_from_training"](y1, groups, train)
    pred0 = helpers["ridge_dense"](X, y0, groups, train, lam=10.0)
    pred1 = helpers["ridge_dense"](X, y1, groups, train, lam=10.0)

    np.testing.assert_array_equal(center0[train], center1[train])
    np.testing.assert_allclose(pred0, pred1, atol=1e-12, rtol=0.0)


def test_text_pipeline_fits_vocabulary_and_idf_on_fit_only():
    source = STRICT.read_text()
    assert "V.fit([T[i] for i in idx]).transform(T)" in source
    assert "Vq.fit_transform([T[i] for i in train_idx])" in source
    assert "H_train[inner_valid, j] = fit_predict(key, inner_train, inner_valid)" in source

    fit_docs = ["common shared"] * 35 + ["common other"] * 5
    cal_doc, test_doc = "common other", "delta"
    changed_test_doc = "common delta"
    kwargs = dict(min_df=3, max_df=0.95, ngram_range=(1, 1), use_idf=True)
    v0 = TfidfVectorizer(**kwargs).fit(fit_docs)
    v1 = TfidfVectorizer(**kwargs).fit(fit_docs)
    fit0 = v0.transform(fit_docs).toarray()
    fit1 = v1.transform(fit_docs).toarray()
    model0 = Ridge(alpha=3.0).fit(fit0, np.linspace(-1, 1, len(fit_docs)))
    model1 = Ridge(alpha=3.0).fit(fit1, np.linspace(-1, 1, len(fit_docs)))
    np.testing.assert_array_equal(fit0, fit1)
    np.testing.assert_allclose(model0.coef_, model1.coef_, atol=0.0, rtol=0.0)
    cal0 = model0.predict(v0.transform([cal_doc]))
    cal1 = model1.predict(v1.transform([cal_doc]))
    np.testing.assert_allclose(cal0, cal1, atol=0.0, rtol=0.0)
    # The future-case representation can change when its own text changes;
    # only FIT/CAL artifacts are required to remain fixed.
    assert v0.transform([test_doc]).shape == v0.transform([changed_test_doc]).shape


def test_agent_diagnostics_center_outcomes_on_fit_only():
    source = STRICT.read_text()
    assert "centre_from_training(rcap, fam, fit) - o" in source
    assert "rc = centre(rcap, fam)" not in source


def test_strict_mayo_default_derived_path_matches_local_project():
    source = STRICT.read_text()
    assert "~/PycharmProjects/peras/data/derived/mayo" in source
    assert "~/mnt/PycharmProjects/peras/data/derived/mayo" not in source


def test_legacy_builder_attack_heldout_frequency_recodes_fit_rows():
    spec = importlib.util.spec_from_file_location("legacy_mayo_builder_test", LEGACY_BUILDER)
    builder = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(builder)

    fit = pd.DataFrame({
        "primary_procedure_name": ["rare-level"] * 24 + ["common-level"] * 30,
        "numeric_control": range(54),
    })
    held = pd.DataFrame({
        "primary_procedure_name": ["rare-level"],
        "numeric_control": [54],
    })
    fit_only, _ = builder.frame(fit, rare_min=25)
    full, _ = builder.frame(pd.concat([fit, held], ignore_index=True), rare_min=25)
    assert set(fit_only.iloc[:24]["primary_procedure_name"]) == {"__rare__"}
    assert set(full.iloc[:24]["primary_procedure_name"]) == {"rare-level"}


def test_strict_encoder_fit_rows_are_invariant_to_heldout_category_frequency():
    encoder = _load_helpers()["strict_tabular_matrix"]
    fit = pd.DataFrame({
        "primary_procedure_name": ["rare-level"] * 24 + ["common-level"] * 30,
        "numeric_control": range(54),
    })
    held = pd.DataFrame({
        "primary_procedure_name": ["rare-level"],
        "numeric_control": [54],
    })
    all_rows = pd.concat([fit, held], ignore_index=True)
    fit_mask = np.r_[np.ones(len(fit), dtype=bool), False]
    fit_only_matrix, fit_only_names = encoder(fit, np.ones(len(fit), dtype=bool), rare_min=25)
    mixed_matrix, mixed_names = encoder(all_rows, fit_mask, rare_min=25)
    changed = all_rows.copy()
    changed.loc[len(fit), "primary_procedure_name"] = "novel-heldout-category"
    changed_matrix, changed_names = encoder(changed, fit_mask, rare_min=25)

    assert fit_only_names == mixed_names == changed_names
    np.testing.assert_array_equal(fit_only_matrix.toarray(), mixed_matrix[:len(fit)].toarray())
    np.testing.assert_array_equal(mixed_matrix[:len(fit)].toarray(), changed_matrix[:len(fit)].toarray())


def test_feature_alignment_guard_rejects_shuffled_artifact_rows():
    validate = _load_helpers()["validate_feature_alignment"]
    df = pd.DataFrame({f"x{i}": np.arange(40, dtype=float) + i for i in range(24)})
    xa = df.copy()
    xb = df.copy()
    assert validate(df, xa, xb)["direct_df_feature_columns_checked"] == 24
    with pytest.raises(ValueError, match="disagree"):
        validate(df, xa, xb.iloc[::-1].reset_index(drop=True))


def test_paired_bootstrap_resamples_whole_patients_and_keeps_duplicate_draws():
    bootstrap = _load_helpers()["paired_cluster_bootstrap"]
    # Patient A contributes two cases; patient B contributes one. Resampling
    # two patients with replacement must preserve both A cases on an A draw.
    result = bootstrap([1.0, 1.0, 10.0], ["A", "A", "B"], reps=500, seed=7)
    assert result["point_gain_minutes"] == 4.0
    assert result["test_patients"] == 2
    assert result["test_cases"] == 3
    assert result["ci95_case_weighted_patient_cluster"] == [1.0, 10.0]
    assert result["familywise95_bonferroni_ci_across_5_arms"] == [1.0, 10.0]
