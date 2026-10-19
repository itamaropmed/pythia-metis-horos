"""Outcome-free construction of randomized-block near-replicate designs."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Any

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class BlockedPairDesign:
    left_indices: np.ndarray
    right_indices: np.ndarray
    distances: np.ndarray
    pair_blocks: np.ndarray
    meta: Mapping[str, Any]


def _design_digest(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        digest.update(np.ascontiguousarray(value).view(np.uint8))
    return digest.hexdigest()


def global_nearest_pairs(
    representation: np.ndarray,
    entities: np.ndarray,
) -> BlockedPairDesign:
    """Greedily form disjoint global L1-nearest pairs without outcomes.

    Repeated nearest-neighbor passes avoid a quadratic distance matrix. The
    construction is deterministic and requires an even number of unique
    entities so no outcome-dependent retention decision can be hidden.
    """
    x = np.atleast_2d(np.asarray(representation, float))
    entity = np.asarray(entities).astype(str).ravel()
    if x.shape[0] != entity.size or x.shape[0] < 4 or x.shape[0] % 2:
        raise ValueError("global pairing requires an even number of at least four aligned rows")
    if not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1):
        raise ValueError("pairing representation must be finite and contained in [0,1]")
    if np.unique(entity).size != entity.size:
        raise ValueError("pair construction requires one row per sampling entity")

    remaining = np.arange(x.shape[0])
    left: list[int] = []
    right: list[int] = []
    distances: list[float] = []
    while remaining.size:
        tree = cKDTree(x[remaining])
        # With duplicate coordinates, cKDTree need not put the query row in
        # slot zero. Select an explicitly different row or a self-edge can
        # leave an odd number of unmatched entities.
        nearest_distance, nearest_position = tree.query(
            x[remaining], k=min(8, remaining.size), p=1
        )
        edges = {}
        for local_i, candidates in enumerate(np.atleast_2d(nearest_position)):
            rank = next(rank for rank, candidate in enumerate(candidates) if candidate != local_i)
            local_j = int(candidates[rank])
            i, j = sorted((int(remaining[local_i]), int(remaining[local_j])))
            edges[(i, j)] = float(nearest_distance[local_i, rank] / x.shape[1])
        used: set[int] = set()
        for (i, j), distance in sorted(edges.items(), key=lambda item: (item[1], *item[0])):
            if i not in used and j not in used:
                left.append(i); right.append(j); distances.append(distance)
                used.update((i, j))
        remaining = remaining[~np.isin(remaining, np.fromiter(used, dtype=int))]

    arrays = [
        np.asarray(left, int), np.asarray(right, int), np.asarray(distances, float),
        np.arange(len(left), dtype=int),
    ]
    return BlockedPairDesign(
        left_indices=arrays[0], right_indices=arrays[1], distances=arrays[2],
        pair_blocks=arrays[3],
        meta={
            "pairing_function_outcome_free": True, "blocks_randomized": False,
            "pairing_outcome_blind": False, "iid_randomized_blocks": False,
            "conditionally_independent_entities": False,
            "independent_blocks": False,
            "algorithm": "global-repeated-nearest-mean-l1-v1",
            "n_input_entities": int(x.shape[0]), "n_used_entities": int(x.shape[0]),
            "design_hash": _design_digest(arrays),
            "audit_required": (
                "an external provenance audit must establish an outcome-blind representation "
                "and conditionally independent entity outcomes before certification"
            ),
        },
    )


def distance_refined_pairs(
    representation: np.ndarray, entities: np.ndarray, *, max_passes: int = 3,
) -> BlockedPairDesign:
    """Outcome-blind 2-opt distance refinement of the global greedy matching.

    The swaps can reduce total pair distance but need not improve a realized
    risk witness. They never inspect outcome values and preserve disjointness.
    """
    if max_passes < 1:
        raise ValueError("max_passes must be positive")
    initial = global_nearest_pairs(representation, entities)
    x = np.asarray(representation, float)
    left = initial.left_indices.copy()
    right = initial.right_indices.copy()
    distance = initial.distances.copy()
    n_swaps = 0
    for _ in range(max_passes):
        changed = False
        for i in range(len(left)):
            for j in range(i + 1, len(left)):
                a, b, c, d = left[i], right[i], left[j], right[j]
                old = distance[i] + distance[j]
                alternatives = (
                    ((a, c), (b, d)),
                    ((a, d), (b, c)),
                )
                choices = []
                for first, second in alternatives:
                    first_distance = float(np.mean(np.abs(x[first[0]] - x[first[1]])))
                    second_distance = float(np.mean(np.abs(x[second[0]] - x[second[1]])))
                    choices.append((first_distance + second_distance,
                                    first, second, first_distance, second_distance))
                best = min(choices, key=lambda item: item[0])
                if best[0] < old - 1e-12:
                    left[i], right[i] = best[1]
                    left[j], right[j] = best[2]
                    distance[i], distance[j] = best[3], best[4]
                    n_swaps += 1
                    changed = True
        if not changed:
            break
    blocks = np.arange(len(left), dtype=int)
    arrays = [left, right, distance, blocks]
    return BlockedPairDesign(
        left_indices=left, right_indices=right, distances=distance,
        pair_blocks=blocks,
        meta={
            **initial.meta,
            "algorithm": "global-greedy-distance-2opt-v1",
            "max_passes": int(max_passes), "n_distance_reducing_swaps": n_swaps,
            "initial_mean_pair_distance": float(np.mean(initial.distances)),
            "refined_mean_pair_distance": float(np.mean(distance)),
            "design_hash": _design_digest(arrays),
        },
    )


def clustered_nearest_pairs(
    representation: np.ndarray,
    entities: np.ndarray,
    clusters: np.ndarray,
) -> BlockedPairDesign:
    """Form outcome-blind nearest pairs separately within declared clusters."""
    x = np.atleast_2d(np.asarray(representation, float))
    entity = np.asarray(entities).astype(str).ravel()
    cluster = np.asarray(clusters).astype(str).ravel()
    if x.shape[0] != entity.size or cluster.size != entity.size:
        raise ValueError("representation, entities, and clusters must align")
    if not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1):
        raise ValueError("pairing representation must be finite and contained in [0,1]")
    if np.unique(entity).size != entity.size:
        raise ValueError("pair construction requires one row per sampling entity")

    left: list[int] = []
    right: list[int] = []
    distances: list[float] = []
    pair_blocks: list[str] = []
    dropped: list[int] = []
    for label in sorted(np.unique(cluster)):
        rows = np.flatnonzero(cluster == label)
        if rows.size % 2:
            # Entity-ID ordering is deterministic and cannot inspect outcomes.
            order = np.argsort(entity[rows], kind="stable")
            dropped.append(int(rows[order[-1]]))
            rows = rows[order[:-1]]
        if rows.size < 2:
            continue
        if rows.size == 2:
            local_left, local_right = rows[:1], rows[1:]
            local_distance = np.asarray([np.mean(np.abs(x[rows[0]] - x[rows[1]]))])
        else:
            local = global_nearest_pairs(x[rows], entity[rows])
            local_left, local_right = rows[local.left_indices], rows[local.right_indices]
            local_distance = local.distances
        left.extend(local_left.tolist()); right.extend(local_right.tolist())
        distances.extend(local_distance.tolist()); pair_blocks.extend([label] * len(local_left))
    if len(left) < 2:
        raise ValueError("clustered pairing produced fewer than two pairs")
    arrays = [
        np.asarray(left, int), np.asarray(right, int), np.asarray(distances, float),
        np.asarray(pair_blocks),
    ]
    return BlockedPairDesign(
        left_indices=arrays[0], right_indices=arrays[1], distances=arrays[2],
        pair_blocks=arrays[3],
        meta={
            "pairing_function_outcome_free": True, "blocks_randomized": False,
            "pairing_outcome_blind": False, "iid_randomized_blocks": False,
            "conditionally_independent_entities": False, "independent_blocks": False,
            "algorithm": "clustered-global-nearest-mean-l1-v1",
            "n_input_entities": int(x.shape[0]), "n_used_entities": int(2 * len(left)),
            "n_dropped_entities": len(dropped), "n_blocks": len(np.unique(arrays[3])),
            "design_hash": _design_digest(arrays),
            "audit_required": (
                "an external provenance audit must establish outcome-blind representation, "
                "cluster construction, and cross-cluster independence"
            ),
        },
    )


def randomized_block_pairs(
    representation: np.ndarray,
    entities: np.ndarray,
    *,
    block_size: int = 8,
    seed: int = 0,
) -> BlockedPairDesign:
    """Pair nearby rows inside random blocks without accepting outcomes.

    Input features must already be mapped to ``[0,1]`` under a frozen metric
    contract. Greedy matching is deterministic within each randomized block.
    """
    x = np.atleast_2d(np.asarray(representation, float))
    entity = np.asarray(entities).astype(str).ravel()
    if x.shape[0] != entity.size or x.shape[0] < block_size:
        raise ValueError("representation rows and entities must align")
    if not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1):
        raise ValueError("pairing representation must be finite and contained in [0,1]")
    if np.unique(entity).size != entity.size:
        raise ValueError("pair construction requires one row per sampling entity")
    if block_size < 2 or block_size % 2:
        raise ValueError("block_size must be an even integer of at least two")

    rng = np.random.default_rng(seed)
    usable = (x.shape[0] // block_size) * block_size
    block_rows = rng.permutation(x.shape[0])[:usable].reshape(-1, block_size)
    left, right, distances, blocks = [], [], [], []
    for block_id, rows in enumerate(block_rows):
        available = list(range(block_size))
        while available:
            best = min(
                (
                    (float(np.mean(np.abs(x[rows[i]] - x[rows[j]]))), i, j)
                    for position, i in enumerate(available)
                    for j in available[position + 1:]
                ),
                key=lambda value: (value[0], value[1], value[2]),
            )
            distance, i, j = best
            left.append(rows[i]); right.append(rows[j]); distances.append(distance)
            blocks.append(block_id)
            available.remove(i); available.remove(j)

    arrays = [
        np.asarray(left, int), np.asarray(right, int), np.asarray(distances, float),
        np.asarray(blocks, int),
    ]
    return BlockedPairDesign(
        left_indices=arrays[0], right_indices=arrays[1], distances=arrays[2],
        pair_blocks=arrays[3],
        meta={
            "pairing_function_outcome_free": True, "blocks_randomized": True,
            "pairing_outcome_blind": False, "iid_randomized_blocks": False,
            "conditionally_independent_entities": False,
            "independent_blocks": False,
            "algorithm": "random-block-greedy-mean-l1-v1", "seed": int(seed),
            "block_size": int(block_size), "n_input_entities": int(x.shape[0]),
            "n_used_entities": int(2 * arrays[0].size),
            "design_hash": _design_digest(arrays),
            "audit_required": (
                "an external provenance audit must establish outcome-blind representation "
                "and iid sampling before certification"
            ),
        },
    )
