"""Selection-safe multiscale residual text-information certificates.

The table cells are fixed first.  Each supplied text geometry is an
outcome-blind representation of the notes (for example a concept vector, a
TF--IDF geometry, or an embedding geometry).  The existing anchored witness
is evaluated separately at ``alpha / number_of_scales`` and the largest lower
endpoint is returned.  The Bonferroni ledger makes the max selection safe;
the function never trains a predictor and never selects a geometry using Y.

This is a composition theorem, not a new concentration inequality.  Its
practical purpose is to avoid making a rare text signal disappear merely
because one representation has poor neighborhood resolution.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
from scipy.linalg import helmert

from .anchored_information import anchored_information
from .unified import nli_cohort_bound


def _resolution_score(cells: np.ndarray, geometry: np.ndarray) -> dict:
    """Outcome-blind support diagnostics used only for alpha allocation."""
    counts = np.unique(cells, return_counts=True)[1]
    pairs = int(np.sum(counts // 2))
    singletons = int(np.sum(counts == 1))
    d = max(int(geometry.shape[1]), 1)
    # Dimension is a fixed complexity penalty; no outcomes are inspected.
    score = max(float(pairs) / np.sqrt(d) * (1.0 - singletons / max(len(cells), 1)), 1e-12)
    return {
        "n_pairs": pairs,
        "singleton_fraction": singletons / max(len(cells), 1),
        "dimension": d,
        "resolution_score": score,
    }


def _resolution_status(result: Mapping[str, object], min_pairs: int) -> str:
    if int(result["n_used"]) < 2 or int(result["n_pairs"]) < min_pairs:
        return "insufficient_resolution"
    # A zero lower endpoint is not evidence for a null.  A near-zero observed
    # witness can be compatible with zero; a positive witness with a zero LCB
    # is simply unresolved at this sample size.
    if float(result["dI_lower"]) > 0:
        return "positive_certificate"
    if abs(float(result["dI_point"])) < 1e-12:
        return "compatible_with_zero"
    return "insufficient_resolution"


def multiscale_residual_information(
    table_cells: np.ndarray,
    text_geometries: Iterable[np.ndarray],
    y: np.ndarray,
    K: int,
    *,
    alpha: float = 0.05,
    min_cell: int = 4,
    seed: int = 0,
    names: Iterable[str] | None = None,
    provenance: Mapping[str, object] | None = None,
    alpha_weighting: str = "equal",
    min_pairs: int = 20,
) -> dict:
    """Return a simultaneous lower bound for text information beyond a table.

    ``text_geometries`` must be frozen before outcomes are inspected.  Every
    geometry must have one row per outcome and must be outcome-blind.  The
    implementation records these assumptions but cannot independently verify
    their provenance.  Empty or malformed ladders are refused rather than
    silently returning a numerical result.
    """
    cells = np.asarray(table_cells)
    outcomes = np.asarray(y)
    geometries = [np.asarray(g, dtype=float) for g in text_geometries]
    if cells.ndim != 1 or outcomes.ndim != 1 or len(cells) != len(outcomes):
        return {"status": "refused", "claim_tier": "refused",
                "refusal": {"reason": "aligned table cells and outcomes are required"}}
    if not geometries:
        return {"status": "refused", "claim_tier": "refused",
                "refusal": {"reason": "at least one frozen text geometry is required"}}
    if not np.isfinite(outcomes).all() or not (0 < float(alpha) < 1):
        return {"status": "refused", "claim_tier": "refused",
                "refusal": {"reason": "finite outcomes and alpha in (0,1) are required"}}
    for i, geometry in enumerate(geometries):
        if geometry.ndim != 2 or geometry.shape[0] != len(outcomes) or not np.isfinite(geometry).all():
            return {"status": "refused", "claim_tier": "refused",
                    "refusal": {"reason": f"geometry {i} is not a finite n-by-d matrix"}}
    if names is None:
        labels = [f"scale_{i}" for i in range(len(geometries))]
    else:
        labels = [str(x) for x in names]
        if len(labels) != len(geometries) or len(set(labels)) != len(labels):
            return {"status": "refused", "claim_tier": "refused",
                    "refusal": {"reason": "geometry names must be unique and aligned"}}
    if not provenance:
        return {"status": "refused", "claim_tier": "refused",
                "refusal": {"reason": "frozen geometry provenance is required"}}

    m = len(geometries)
    if alpha_weighting not in {"equal", "resolution"}:
        return {"status": "refused", "claim_tier": "refused",
                "refusal": {"reason": "alpha_weighting must be equal or resolution"}}
    support = [_resolution_score(cells, geometry) for geometry in geometries]
    raw_weights = np.ones(m, dtype=float)
    if alpha_weighting == "resolution":
        raw_weights = np.asarray([item["resolution_score"] for item in support])
    alpha_weights = raw_weights / raw_weights.sum()
    results = []
    for i, (label, geometry) in enumerate(zip(labels, geometries)):
        scale_alpha = float(alpha) * float(alpha_weights[i])
        result = anchored_information(
            cells, geometry, outcomes, K,
            alpha=scale_alpha, min_cell=min_cell, seed=int(seed) + i,
        )
        enriched = {"name": label, "alpha": scale_alpha,
                    **support[i], **result}
        enriched["resolution_status"] = _resolution_status(enriched, min_pairs)
        results.append(enriched)
    selected = int(np.argmax([r["dI_lower"] for r in results]))
    return {
        "status": "complete",
        "claim_tier": "conditional",
        "dI_lower": float(results[selected]["dI_lower"]),
        "dI_point": float(results[selected]["dI_point"]),
        "selected_scale": labels[selected],
        "scales": results,
        "n_scales": m,
        "alpha": float(alpha),
        "alpha_split": [float(alpha * x) for x in alpha_weights],
        "alpha_weights": [float(x) for x in alpha_weights],
        "alpha_weighting": alpha_weighting,
        "selection_simultaneous": True,
        "n_used": int(results[selected]["n_used"]),
        "cells_used": int(results[selected]["cells_used"]),
        "provenance": dict(provenance),
        "assumptions": (
            "table cells and all text geometries were frozen before outcomes",
            "geometries are outcome-blind and available at prediction time",
            "distinct entities are conditionally independent given table and text",
            "the alpha ledger covers every registered geometry",
        ),
        "refusal": None,
    }


def hierarchical_residual_information(
    table_cells: np.ndarray,
    parent_cells: np.ndarray,
    text_geometry: np.ndarray,
    y: np.ndarray,
    K: int,
    *,
    variation_penalty: float | None,
    alpha: float = 0.05,
    min_cell: int = 4,
    min_pairs: int = 20,
    seed: int = 0,
    provenance: Mapping[str, object] | None = None,
) -> dict:
    """Pool sparse child cells through a parent route with a declared penalty.

    The parent anchored witness targets text information beyond the parent
    representation.  If ``variation_penalty`` upper-bounds the Brier-scale
    information attributable to the child-table refinement within each parent,
    subtracting it gives a valid conditional lower bound for text beyond the
    child table.  The penalty is never estimated from the evaluation outcomes.
    """
    child = np.asarray(table_cells)
    parent = np.asarray(parent_cells)
    geometry = np.asarray(text_geometry, dtype=float)
    outcomes = np.asarray(y)
    if (child.ndim != 1 or parent.ndim != 1 or outcomes.ndim != 1
            or len(child) != len(parent) or len(child) != len(outcomes)
            or geometry.ndim != 2 or geometry.shape[0] != len(outcomes)):
        return {"status": "refused", "claim_tier": "refused",
                "refusal": {"reason": "aligned child, parent, geometry and outcomes are required"}}
    if variation_penalty is None or not np.isfinite(variation_penalty) or variation_penalty < 0:
        return {"status": "refused", "claim_tier": "refused",
                "refusal": {"reason": "a nonnegative externally declared variation penalty is required"}}
    if not provenance:
        return {"status": "refused", "claim_tier": "refused",
                "refusal": {"reason": "parent/child support provenance is required"}}
    base = anchored_information(
        parent, geometry, outcomes, K, alpha=alpha, min_cell=min_cell, seed=seed,
    )
    lower = max(float(base["dI_lower"]) - float(variation_penalty), 0.0)
    status = "positive_certificate" if lower > 0 else (
        "insufficient_resolution" if base["dI_lower"] <= 0 and base["dI_point"] > 0
        else "compatible_with_zero")
    return {
        "status": "complete", "claim_tier": "conditional",
        "dI_lower": lower, "parent_dI_lower": float(base["dI_lower"]),
        "variation_penalty": float(variation_penalty),
        "resolution_status": status,
        "child_cells": int(np.unique(child).size),
        "parent_cells": int(np.unique(parent).size),
        "support": _resolution_score(parent, geometry),
        "base": base, "alpha": float(alpha), "seed": int(seed),
        "provenance": dict(provenance),
        "assumptions": (
            "parent and child labels were frozen before outcomes",
            "text geometry is outcome-blind and available at prediction time",
            "variation_penalty upper-bounds child-table information within parents",
            "distinct entities are conditionally independent given parent and text",
        ), "refusal": None,
    }


def support_residual_information(
    table_cells: np.ndarray,
    support_labels: np.ndarray,
    y: np.ndarray,
    K: int,
    *,
    alpha: float = 0.05,
    concentration: str = "heterogeneous_chernoff",
    min_support: int = 8,
    provenance: Mapping[str, object] | None = None,
) -> dict:
    """Certify residual information carried by a frozen finite text support.

    The most common support value is an outcome-blind baseline.  For every
    other registered value and each of ``K-1`` outcome coordinates, a fixed
    support-by-class contrast is evaluated.  The variational NLI inequality
    makes each endpoint a lower bound on Brier information beyond the table;
    Bonferroni makes selecting their maximum safe.  No model is fitted.
    """
    cells = np.asarray(table_cells)
    support = np.asarray(support_labels).astype(str)
    outcomes = np.asarray(y)
    n = len(outcomes)
    if (cells.ndim != 1 or support.ndim != 1 or outcomes.ndim != 1
            or len(cells) != n or len(support) != n or K < 2
            or not 0 < alpha < 1):
        return {"status": "refused", "claim_tier": "refused",
                "refusal": {"reason": "aligned cells/support/outcomes, K>=2, and valid alpha required"}}
    if not provenance:
        return {"status": "refused", "claim_tier": "refused",
                "refusal": {"reason": "frozen support provenance is required"}}
    values, counts = np.unique(support, return_counts=True)
    baseline = str(values[int(np.argmax(counts))])
    eligible = [str(v) for v, count in zip(values, counts)
                if str(v) != baseline and int(count) >= int(min_support)]
    if not eligible:
        return {
            "status": "complete", "claim_tier": "conditional",
            "dI_lower": 0.0, "resolution_status": "insufficient_resolution",
            "baseline": baseline, "registered_supports": values.tolist(),
            "eligible_supports": [], "candidates": [], "alpha": float(alpha),
            "provenance": dict(provenance), "refusal": None,
        }
    outcome_basis = helmert(K, full=False)
    specs = [(value, axis, sign) for value in eligible
             for axis in range(K - 1) for sign in (-1.0, 1.0)]
    per_alpha = alpha / len(specs)
    candidates = []
    for value, axis, sign in specs:
        q = np.zeros((n, K), float)
        q[support == value] = sign * outcome_basis[axis]
        bound, info = nli_cohort_bound(
            cells, q, outcomes, per_alpha, concentration=concentration,
        )
        candidates.append({
            "support": value, "outcome_axis": int(axis),
            "direction_sign": int(sign),
            "alpha": float(per_alpha), "dI_lower": float(bound),
            "support_count": int(np.sum(support == value)), **info,
        })
    aggregates = []
    for value in eligible:
        axes = []
        for axis in range(K - 1):
            direction = max(
                (item for item in candidates
                 if item["support"] == value and item["outcome_axis"] == axis),
                key=lambda item: item["dI_lower"],
            )
            axes.append(direction)
        # Helmert axes are orthonormal, so their variational projections add.
        aggregates.append({"support": value, "dI_lower": float(sum(
            item["dI_lower"] for item in axes)), "axes": axes})
    selected = max(aggregates, key=lambda item: item["dI_lower"])
    selected_point = max((item["C"] for item in selected["axes"]), default=0.0)
    if selected["dI_lower"] > 0:
        resolution = "positive_certificate"
    elif selected_point > 0:
        resolution = "insufficient_resolution"
    else:
        resolution = "compatible_with_zero"
    return {
        "status": "complete", "claim_tier": "conditional",
        "dI_lower": float(selected["dI_lower"]),
        "resolution_status": resolution, "selected": selected,
        "baseline": baseline, "registered_supports": values.tolist(),
        "eligible_supports": eligible, "candidates": candidates,
        "support_aggregates": aggregates,
        "alpha": float(alpha), "concentration": concentration,
        "selection_simultaneous": True, "provenance": dict(provenance),
        "assumptions": (
            "support labels and their finite registry were frozen before outcomes",
            "support extraction is outcome-blind and available at prediction time",
            "rows are conditionally independent given table and text",
            "all support-by-class contrasts are covered by the alpha ledger",
        ), "refusal": None,
    }


def hierarchical_residual_ladder(
    table_cells: np.ndarray,
    parent_partitions: Iterable[np.ndarray],
    text_geometry: np.ndarray,
    y: np.ndarray,
    K: int,
    *,
    variation_penalties: Iterable[float],
    names: Iterable[str] | None = None,
    alpha: float = 0.05,
    min_cell: int = 4,
    min_pairs: int = 20,
    seed: int = 0,
    alpha_weighting: str = "resolution",
    provenance: Mapping[str, object] | None = None,
) -> dict:
    """Selection-safe pooling over predeclared neighboring-cell hierarchies."""
    parents = [np.asarray(item) for item in parent_partitions]
    penalties = [float(item) for item in variation_penalties]
    labels = ([f"parent_{i}" for i in range(len(parents))] if names is None
              else [str(item) for item in names])
    if (not parents or len(parents) != len(penalties) or len(parents) != len(labels)
            or len(set(labels)) != len(labels) or not provenance):
        return {"status": "refused", "claim_tier": "refused",
                "refusal": {"reason": "aligned nonempty hierarchy, penalties, unique names and provenance required"}}
    if alpha_weighting not in {"equal", "resolution"}:
        return {"status": "refused", "claim_tier": "refused",
                "refusal": {"reason": "alpha_weighting must be equal or resolution"}}
    geometry = np.asarray(text_geometry, float)
    support = [_resolution_score(parent, geometry) for parent in parents]
    weights = np.ones(len(parents)) if alpha_weighting == "equal" else np.array(
        [item["resolution_score"] for item in support], float)
    weights /= weights.sum()
    routes = []
    for i, (label, parent, penalty) in enumerate(zip(labels, parents, penalties)):
        route = hierarchical_residual_information(
            table_cells, parent, geometry, y, K,
            variation_penalty=penalty, alpha=float(alpha * weights[i]),
            min_cell=min_cell, min_pairs=min_pairs, seed=seed + i,
            provenance={**dict(provenance), "hierarchy_name": label},
        )
        if route["status"] == "refused":
            return route
        routes.append({"name": label, "alpha": float(alpha * weights[i]),
                       **support[i], **route})
    selected = max(routes, key=lambda item: item["dI_lower"])
    return {
        "status": "complete", "claim_tier": "conditional",
        "dI_lower": float(selected["dI_lower"]),
        "resolution_status": selected["resolution_status"],
        "selected_hierarchy": selected["name"], "routes": routes,
        "alpha": float(alpha), "alpha_split": [float(alpha * w) for w in weights],
        "alpha_weighting": alpha_weighting, "selection_simultaneous": True,
        "provenance": dict(provenance), "refusal": None,
    }
