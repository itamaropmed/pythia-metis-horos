"""Split-fitted group centering for residual-reader experiments."""
from __future__ import annotations

import numpy as np


def center_by_training_groups(values, groups, train_rows):
    """Center rows using group means estimated only from ``train_rows``.

    Groups unseen in training use the global training mean. Changing held-out
    outcomes therefore cannot change transformed training outcomes.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    g = np.asarray(groups).reshape(-1)
    train = np.asarray(train_rows, dtype=bool).reshape(-1)
    if len(x) != len(g) or len(x) != len(train):
        raise ValueError("values, groups, and train_rows must have equal length")
    if not train.any() or not np.isfinite(x).all():
        raise ValueError("training rows must be nonempty and values finite")
    fallback = x[train].mean(axis=0)
    out = x.copy()
    for group in np.unique(g):
        rows = g == group
        reference = rows & train
        center = x[reference].mean(axis=0) if reference.any() else fallback
        out[rows] -= center
    return out
