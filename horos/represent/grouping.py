"""Outcome-free hierarchy of disjoint near-replicate groups."""
from __future__ import annotations

import numpy as np

from .pairing import global_nearest_pairs


def nearest_group_hierarchy(
    representation: np.ndarray, entities: np.ndarray,
    *, max_group_size: int = 16,
) -> dict[int, np.ndarray]:
    """Pair group centroids recursively; no response value is ever read."""
    x = np.asarray(representation, float)
    entity = np.asarray(entities).astype(str).ravel()
    if x.ndim != 2 or x.shape[0] != len(entity) or len(entity) < 4:
        raise ValueError("aligned two-dimensional representation with >=4 rows required")
    if len(entity) % 2 or np.unique(entity).size != len(entity):
        raise ValueError("grouping requires an even number of unique entities")
    if max_group_size < 2 or max_group_size & (max_group_size - 1):
        raise ValueError("max_group_size must be a power of two >=2")
    groups = np.arange(len(entity))[:, None]
    hierarchy = {}
    while groups.shape[1] < max_group_size and len(groups) % 2 == 0 and len(groups) >= 4:
        centroids = np.mean(x[groups], axis=1)
        design = global_nearest_pairs(centroids, np.arange(len(groups)))
        groups = np.concatenate([
            groups[design.left_indices], groups[design.right_indices],
        ], axis=1)
        hierarchy[groups.shape[1]] = groups.copy()
    return hierarchy
