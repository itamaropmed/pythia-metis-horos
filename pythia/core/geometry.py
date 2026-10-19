"""Outcome-blind geometric primitives with no estimator training lifecycle."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def knn_query(reference, query=None, k=1):
    """Return Euclidean neighbor distances/indices using a fixed spatial index."""
    ref = np.asarray(reference, float)
    qry = ref if query is None else np.asarray(query, float)
    kk = min(int(k), len(ref))
    d, idx = cKDTree(ref).query(qry, k=kk)
    if kk == 1:
        d, idx = d[:, None], idx[:, None]
    return np.asarray(d), np.asarray(idx)
