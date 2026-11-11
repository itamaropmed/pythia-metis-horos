"""Outcome-blind forward-time folds with whole-entity exclusion."""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_group_splits(
    times,
    entity_ids,
    *,
    n_splits: int = 5,
    min_train_fraction: float = 0.5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return expanding-window (train, validation) row indices.

    Validation uses successive blocks of unique timestamps after a warm-up
    prefix. Training is restricted to strictly earlier timestamps and excludes
    every entity appearing in that validation block. Warm-up rows receive no
    OOF prediction by design. Inputs are only timestamps and entity IDs; labels
    are neither accepted nor consulted.
    """
    time_ns = pd.DatetimeIndex(pd.to_datetime(times, utc=True, errors="coerce")).asi8
    raw_entities = np.asarray(entity_ids).reshape(-1)
    if pd.Series(raw_entities).isna().any():
        raise ValueError("entity_ids must be complete and nonmissing")
    entities = pd.Series(raw_entities).astype(str).str.strip().to_numpy()
    n = len(time_ns)
    if n == 0 or len(entities) != n:
        raise ValueError("times and entity_ids must be nonempty and row-aligned")
    if np.any(entities == ""):
        raise ValueError("entity_ids must be complete and nonempty")
    nat = np.iinfo(np.int64).min
    if np.any(time_ns == nat):
        raise ValueError("times must be valid and nonmissing")
    if n_splits < 1 or not 0 < min_train_fraction < 1:
        raise ValueError("n_splits and min_train_fraction are invalid")
    unique_times = np.unique(time_ns)
    warmup = max(1, int(np.ceil(len(unique_times) * min_train_fraction)))
    later_times = unique_times[warmup:]
    if later_times.size == 0:
        raise ValueError("not enough distinct timestamps after the warm-up period")
    blocks = [block for block in np.array_split(later_times, min(n_splits, len(later_times))) if block.size]
    splits = []
    for block in blocks:
        val = np.flatnonzero(np.isin(time_ns, block))
        start = int(block[0])
        val_entities = np.unique(entities[val])
        train_mask = (time_ns < start) & ~np.isin(entities, val_entities)
        train = np.flatnonzero(train_mask)
        if not len(train) or not len(val):
            continue
        if np.intersect1d(entities[train], entities[val]).size:
            raise RuntimeError("internal error: entity overlap in forward split")
        if time_ns[train].max() >= time_ns[val].min():
            raise RuntimeError("internal error: non-forward temporal split")
        splits.append((train, val))
    if not splits:
        raise ValueError("no valid forward entity-disjoint folds could be formed")
    return splits
