"""Horos-side score learning for Pythia's projection certificate (Pythia itself trains nothing).

Pythia's bound for any label-free score H centred within table cells is
    V(cells) >= (C_lower)_+^2 / E|H|^2,      C = mean H[i, y_i],
so the bound is (alignment cos^2 between H and p - pbar) x V, minus a deviation.  The optimal score is
h* = P(Y | X, R) - P(Y | X): the *conditional information direction*.  This module learns that direction
directly, on FIT rows only, instead of going through a classifier's probabilities:

  variational_scores  multi-output ridge of the cell-centred one-hot outcome on cell-centred reader
                      features.  This is the empirical maximiser of the variational form
                      NLI = sup_{E[h|X]=0} (E<e_Y - pbar, h>)^2 / E|h|^2 over linear h.
  stack_scores        optimal linear combination of several readers' score matrices; weights maximise the
                      Rayleigh quotient (w'c)^2 / (w'Gw) on out-of-fold FIT scores (closed form w ~ G^-1 c).
                      Unlike 'take the best reader' it adds complementary readers; unlike feature
                      concatenation it fits only J weights, so it does not over-fit small FIT sets.

All weights and models depend on FIT outcomes only, so the scores are label-free for the evaluation rows
and the Pythia certificate stays valid.
"""
from __future__ import annotations

import numpy as np


def _centre(M: np.ndarray, g: np.ndarray) -> np.ndarray:
    out = M.astype(float).copy()
    for c in np.unique(g):
        I = g == c
        out[I] -= out[I].mean(0)
    return out


def _onehot(y, K):
    E = np.zeros((len(y), K)); E[np.arange(len(y)), y] = 1; return E


def _entity_fold_labels(entity_ids, n_rows: int, folds: int, seed: int) -> np.ndarray:
    entities = np.asarray(entity_ids).astype(str).reshape(-1)
    if len(entities) != n_rows:
        raise ValueError("entity_ids must align with rows")
    _, inverse = np.unique(entities, return_inverse=True)
    n_entities = int(inverse.max()) + 1
    if folds < 2 or n_entities < folds:
        raise ValueError("folds must be >= 2 and no greater than the number of entities")
    order = np.random.default_rng(seed).permutation(n_entities)
    entity_fold = np.empty(n_entities, dtype=int)
    entity_fold[order] = np.arange(n_entities) % folds
    return entity_fold[inverse]


def variational_scores(X_fit, y_fit, g_fit, X_eval, g_eval, K: int, ridge: float = 10.0,
                       model: str = "ridge") -> np.ndarray:
    """Score matrix for the evaluation rows (to be centred within evaluation cells by the certificate).
    model="gbm": gradient-boosted regression of each centred outcome column on the raw reader features
    (captures interactions; still label-free for the evaluation rows)."""
    if model == "gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor
        Yf = _centre(_onehot(y_fit, K), g_fit); Xf = np.asarray(X_fit, float); Xe = np.asarray(X_eval, float)
        cols = range(K) if K > 2 else [1]
        H = np.zeros((len(Xe), K))
        for k in cols:
            m = HistGradientBoostingRegressor(max_iter=150, learning_rate=.05, max_leaf_nodes=15,
                                              min_samples_leaf=40).fit(Xf, Yf[:, k])
            H[:, k] = m.predict(Xe)
        if K == 2:
            H[:, 0] = -H[:, 1]
        return H
    Xf = _centre(np.asarray(X_fit, float), g_fit); Yf = _centre(_onehot(y_fit, K), g_fit)
    sd = Xf.std(0) + 1e-9; Xf = Xf / sd
    B = np.linalg.solve(Xf.T @ Xf + ridge * np.eye(Xf.shape[1]), Xf.T @ Yf)
    Xe = _centre(np.asarray(X_eval, float), g_eval) / sd
    return Xe @ B


def oof_scores(fn, X, y, g, K: int, *, entity_ids, folds: int = 2, seed: int = 0, **kw) -> np.ndarray:
    """Entity-disjoint OOF scores used only to fit stacking weights.

    ``entity_ids`` identifies the independent sampling units (e.g. patients),
    not the conditioning cells ``g``. Omitting it is an error: row-random OOF
    can memorize repeated entities and contaminate stack weights.
    """
    y = np.asarray(y)
    # Randomize entity-to-fold assignment while keeping every row of an entity
    # together. This avoids GroupKFold's potentially severe imbalance for
    # uneven longitudinal group sizes.
    r = _entity_fold_labels(entity_ids, len(y), folds, seed)
    H = np.zeros((len(y), K))
    for f in range(folds):
        tr, te = r != f, r == f
        H[te] = fn(X[tr], y[tr], g[tr], X[te], g[te], K, **kw)
    return H


def stack_weights(H_list_oof, y, g, ridge: float = 1e-6) -> np.ndarray:
    """w = argmax (w'c)^2 / (w'Gw): closed form G^-1 c, on cell-centred out-of-fold scores."""
    Hs = [_centre(H, g) for H in H_list_oof]
    c = np.array([H[np.arange(len(y)), y].mean() for H in Hs])
    G = np.array([[np.mean((A * B).sum(1)) for B in Hs] for A in Hs])
    w = np.linalg.solve(G + ridge * np.trace(G) / len(G) * np.eye(len(G)), c)
    return w / (np.abs(w).sum() + 1e-12)


def stack_scores(H_list_eval, w) -> np.ndarray:
    return sum(wj * H for wj, H in zip(w, H_list_eval))


def variational_scores_cv(X_fit, y_fit, g_fit, X_eval, g_eval, K: int, grid=(1, 10, 100, 1000, 10000),
                          folds: int = 3, seed: int = 0) -> np.ndarray:
    """Ridge variational scores with the penalty chosen on FIT only, by out-of-fold certificate signal C^2/E|H|^2."""
    r = np.random.default_rng(seed).permutation(len(y_fit)) % folds; best, best_lam = -1, grid[0]
    for lam in grid:
        H = np.zeros((len(y_fit), K))
        for f in range(folds):
            tr, te = r != f, r == f
            H[te] = variational_scores(X_fit[tr], y_fit[tr], g_fit[tr], X_fit[te], g_fit[te], K, ridge=lam)
        Hc = _centre(H, g_fit); C = Hc[np.arange(len(y_fit)), y_fit].mean()
        sig = max(C, 0) ** 2 / max((Hc ** 2).sum(1).mean(), 1e-12)
        if sig > best:
            best, best_lam = sig, lam
    return variational_scores(X_fit, y_fit, g_fit, X_eval, g_eval, K, ridge=best_lam)



def variational_scores_scalar(X_fit, z_fit, g_fit, X_eval, g_eval, ridge: float = 10.0) -> np.ndarray:
    """Ridge estimate of E[z | X, R] - E[z | X] for a scalar bounded outcome (e.g. capped absolute error)."""
    Xf = _centre(np.asarray(X_fit, float), g_fit); zf = _centre(np.asarray(z_fit, float)[:, None], g_fit)
    sd = Xf.std(0) + 1e-9; Xf = Xf / sd
    b = np.linalg.solve(Xf.T @ Xf + ridge * np.eye(Xf.shape[1]), Xf.T @ zf)
    return (_centre(np.asarray(X_eval, float), g_eval) / sd @ b)[:, 0]
