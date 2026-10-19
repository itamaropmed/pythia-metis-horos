"""Support-free block-SAA certificates over independent entity clusters.

The endpoint truncates independent *witness scores*, never outcomes.  For a
frozen block of clusters, the empirical cellwise optimum has expectation no
larger than the corresponding fixed-design Bayes risk.  Scaling disjoint block
scores by their contribution to the full case-weighted target makes the mean
of the independent scores a lower witness for the full risk.  Safe score
truncation then licenses bounded-mean betting without a population outcome
range.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import math
import numpy as np

from .finite_cell_saa import _cell_score
from .truncated_saa import TruncatedSAACertificate, truncated_saa_portfolio


def _refusal(loss: str, alpha: float, reason: str, diagnostics=None):
    return TruncatedSAACertificate(
        status="refused", claim_tier="refused", loss=loss, bound=None,
        alpha_family=float(alpha), alpha_per_candidate=None,
        selected_route=None, selected_score_cap=None, retained_fraction=None,
        candidates=(), assumptions=(), diagnostics=diagnostics or {},
        refusal={"reason": reason})


def clustered_block_saa_portfolio(
        cell_labels, outcomes, clusters, *, loss: str,
        block_cluster_sizes: Sequence[int], score_caps: Sequence[float],
        alpha: float = .05, quantile: float | None = None,
        window: float | None = None, partition_seed: int,
        representation_provenance: str, partition_provenance: str,
        independent_clusters_attested: bool,
        outcome_blind_labels_attested: bool,
        partition_frozen_before_outcomes: bool,
        exploratory_cluster_override: bool = False,
        ) -> TruncatedSAACertificate:
    """Lower-bound case-weighted Bayes risk without an outcome support bound.

    Different registered block sizes may reuse clusters; dependence across
    routes is handled by the portfolio's Bonferroni ledger.  Within each route,
    blocks are disjoint, so their scores are independent when the input
    clusters are independent.  Leftover clusters are omitted and contribute
    zero to the lower witness.
    """
    labels = np.asarray(cell_labels).astype(str).reshape(-1)
    y = np.asarray(outcomes, dtype=float).reshape(-1)
    group = np.asarray(clusters).astype(str).reshape(-1)
    selected = str(loss).lower()
    n = len(y)
    if (n < 2 or len(labels) != n or len(group) != n
            or not np.isfinite(y).all() or not 0 < alpha < 1):
        raise ValueError("aligned finite rows, labels, clusters and valid alpha required")
    if selected not in {"mae", "pinball", "window_miss"}:
        raise ValueError("loss must be mae, pinball, or window_miss")
    if selected == "pinball" and (
            quantile is None or not math.isfinite(quantile) or not 0 < quantile < 1):
        raise ValueError("pinball requires a quantile strictly between zero and one")
    if selected == "window_miss" and (
            window is None or not math.isfinite(window) or window <= 0):
        raise ValueError("window miss requires a positive finite half-window")
    if not representation_provenance or not partition_provenance:
        raise ValueError("representation and partition provenance are required")
    missing = []
    if not independent_clusters_attested and not exploratory_cluster_override:
        missing.append("independent clusters")
    if not outcome_blind_labels_attested:
        missing.append("outcome-blind representation labels")
    if not partition_frozen_before_outcomes:
        missing.append("outcome-blind frozen partitions")
    if missing:
        return _refusal(
            selected, alpha, "missing required contract: " + ", ".join(missing),
            {"n_rows": n, "n_clusters": int(len(np.unique(group)))})

    sizes = tuple(int(value) for value in block_cluster_sizes)
    if (not sizes or len(sizes) != len(set(sizes))
            or any(value < 2 for value in sizes)):
        raise ValueError("unique block cluster sizes of at least two are required")
    unique, group_inverse = np.unique(group, return_inverse=True)
    if any(value > len(unique) for value in sizes):
        raise ValueError("block size cannot exceed the number of clusters")

    scores_by_route = {}
    route_details = {}
    for route_index, block_size in enumerate(sizes):
        # A separate deterministic shuffle for each registered route.  It reads
        # cluster membership only, never outcomes or representation labels.
        rng = np.random.default_rng(int(partition_seed) + 104729 * route_index)
        cluster_order = rng.permutation(len(unique))
        n_blocks = len(unique) // block_size
        used_positions = cluster_order[:n_blocks * block_size]
        block_by_cluster = np.full(len(unique), -1, dtype=int)
        block_by_cluster[used_positions] = np.repeat(np.arange(n_blocks), block_size)
        row_block = block_by_cluster[group_inverse]
        included = np.flatnonzero(row_block >= 0)
        row_order = included[np.argsort(row_block[included], kind="stable")]
        boundaries = np.searchsorted(row_block[row_order], np.arange(n_blocks + 1))
        scores = []
        for block_index in range(n_blocks):
            row_index = row_order[boundaries[block_index]:boundaries[block_index + 1]]
            block_total = 0.
            for cell in np.unique(labels[row_index]):
                values = y[row_index[labels[row_index] == cell]]
                block_total += len(values) * _cell_score(
                    values, selected, quantile, window)
            # The route mean equals the sum of included block witnesses divided
            # by the full target row count.  Omitted rows therefore contribute
            # a safe zero rather than changing the estimand.
            scores.append(n_blocks * block_total / n)
        name = f"clusters_per_block={block_size}"
        scores_by_route[name] = np.asarray(scores, dtype=float)
        route_details[name] = {
            "clusters_per_block": block_size, "n_blocks": n_blocks,
            "used_clusters": int(len(used_positions)),
            "omitted_clusters": int(len(unique) - len(used_positions)),
            "used_rows": int(len(included)), "full_target_rows": n,
        }

    result = truncated_saa_portfolio(
        scores_by_route, loss=selected, score_caps=score_caps,
        alpha_family=alpha,
        route_provenance={
            "representation": representation_provenance,
            "partition": partition_provenance,
            "partition_seed": int(partition_seed),
            "routes": route_details,
            "score_scaling": "n_blocks * block empirical optimum / full target rows",
        }, portfolio_frozen_before_evaluation=True,
        outcome_scale="original untruncated loss units")
    tier = "conditional" if independent_clusters_attested else "exploratory"
    independence_statement = (
        "entity clusters are independent; dependence within each cluster is arbitrary"
        if independent_clusters_attested else
        "cluster independence is not attested; numerical endpoints have no coverage claim")
    return replace(
        result,
        status=tier, claim_tier=tier,
        assumptions=result.assumptions + (
            independence_statement,
            "representation cells and block partitions were frozen without outcomes",
            "every route uses disjoint blocks; dependence across routes is Bonferroni-protected",
            "no population outcome or loss support is required because witness scores are truncated",
        ),
        diagnostics={
            **dict(result.diagnostics), "n_rows": n,
            "n_clusters": int(len(unique)), "routes": route_details,
            "model_training_performed": False,
            "population_outcome_support_required": False,
            "cluster_independence_attested": bool(independent_clusters_attested),
        })
