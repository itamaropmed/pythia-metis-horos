"""Dynamic geometric embeddings for Agreement-Field Information certificates.

Validity of every AFI certificate requires only that the pairing of rows (i, j) does not look at (Y_i, Y_j).
That leaves the GEOMETRY free to move, as long as it is PREDICTABLE: at the moment a batch is paired, the geometry may
depend on the covariates (tables, texts) of everyone and on the outcomes of rows certified EARLIER, never on the
outcomes of the batch itself.  Three dynamic geometries:

1. Diffusion-time geometry (unsupervised): diffusion-map coordinates psi_t = lambda^t phi of the text kNN graph; the
   diffusion time t moves the geometry from local (t small) to manifold-global (t large).  Portfolio over t.
2. Predictable learned geometry (sequential): rows are randomly ordered into batches 1..B; the geometry for batch b is
   learned from batches < b.  The information-optimal geometry is the outcome law itself (pairs with p_i = p_j have
   zero drift), so the learner targets p(Y | X, T): either the predicted posterior of a classifier (posterior
   geometry) or a Neighbourhood-Components-Analysis projection, whose objective -- maximise the probability that soft
   neighbours share the label -- is exactly 'minimise the matched-pair disagreement' that enters the certificate.
3. Fibre geometry: text embedding with the table-predictable directions projected out (learned from X only),
   so that within a table cell pairing resolves only what the table cannot.

Theorem (predictable geometry).  Let F_{b-1} contain all covariates and the outcomes of batches < b.  If the batch-b
design is F_{b-1}-measurable, then with Z_b = Q_cell,b - Q_text,b (within-batch, within-cell),
   E[Z_b | F_{b-1}] <= (m*/(m*-1)) DeltaI_b ,
Z_b - E[Z_b | F_{b-1}] is conditionally sub-Gaussian with an F_{b-1}-measurable proxy, hence by Azuma-Hoeffding
   sum_b w_b DeltaI_b >= ((m*-1)/m*) [ sum_b w_b Z_b - sqrt(log(1/alpha) * sum_b w_b^2 proxy_b / 2) ]  w.p. >= 1-alpha,
and sum_b w_b DeltaI_b <= DeltaI(T | X) (within-batch cell means only remove variance).  No assumption on the
geometry, the learner or its convergence.
"""
from __future__ import annotations
import math
import numpy as np
from .anchored_information import _greedy_pairs
from .geometry import knn_query


def diffusion_coords(E, k=15, n_comp=20, ts=(1, 4, 16), seed=0):
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    n = len(E); d, idx = knn_query(E, k=k + 1)
    sig = np.median(d[:, 1:]) + 1e-12
    rows = np.repeat(np.arange(n), k); cols = idx[:, 1:].ravel(); w = np.exp(-(d[:, 1:].ravel() / sig) ** 2)
    W = csr_matrix((w, (rows, cols)), shape=(n, n)); W = W.maximum(W.T)
    deg = np.asarray(W.sum(1)).ravel(); Dm = csr_matrix((1 / np.sqrt(deg), (np.arange(n), np.arange(n))), shape=(n, n))
    S = Dm @ W @ Dm
    vals, vecs = eigsh(S, k=n_comp + 1, which="LA", v0=np.random.default_rng(seed).normal(size=n))
    o = np.argsort(-vals); vals, vecs = vals[o][1:], vecs[:, o][:, 1:]
    phi = vecs / np.sqrt(deg)[:, None]
    return {t: phi * (np.clip(vals, 0, None) ** t)[None, :] for t in ts}


def _within_cell_stats(cells, G, y, K, rng):
    """Q_cell (within-cell all-pairs), Q_text (within-cell greedy matching on geometry G), proxies, min cell size."""
    ids, wsum, Qc, Zt, pr_c, mstar = [], 0, 0.0, [], 0.0, 10 ** 9
    groups = [np.flatnonzero(cells == c) for c in np.unique(cells)]
    groups = [g for g in groups if len(g) >= 4]
    n = sum(len(g) for g in groups)
    for I in groups:
        I = I[rng.permutation(len(I))]; m = len(I); w = m / n
        cnt = np.bincount(y[I], minlength=K).astype(float)
        Qc += w * (1 - (cnt * (cnt - 1)).sum() / (m * (m - 1))); pr_c += w ** 2 / (m // 2); mstar = min(mstar, m)
        GI = G[I] + 1e-9 * rng.normal(size=G[I].shape)                       # random tie-breaking
        for a, b in _greedy_pairs(GI):
            Zt.append(float(y[I[a]] != y[I[b]]))
    Zt = np.array(Zt)
    return Qc, float(Zt.mean()), pr_c, 1 / len(Zt), mstar, n


def dynamic_certificate(cells, E0, y, K, alpha=.05, B=3, learner="static", F=None, seed=0, fracs=None, certify_from=0,
                        geometry_builder=None):
    """Sequential predictable-geometry certificate of DeltaI(T | X).

    Pythia never fits a representation or predictor.  ``geometry_builder`` is
    an optional Horos-supplied callable ``(past, current, K, seed) -> G``.  It
    may use only past outcomes; the certificate treats its output as frozen.
    ``learner='static'`` needs no builder and uses ``E0`` directly.
    """
    rng = np.random.default_rng(seed); cells = np.asarray(cells); y = np.asarray(y); n = len(y)
    order = rng.permutation(n)
    if fracs is None:
        batches = np.array_split(order, B)
    else:                                   # e.g. (.2, .4, .4): a small warm-up batch, then larger learned batches
        cuts = (np.cumsum(fracs)[:-1] * n).astype(int); batches = np.split(order, cuts)
    F = E0 if F is None else F
    onehot = np.eye(cells.max() + 1)[cells]
    Z, prox, mstar, points = 0.0, 0.0, 10 ** 9, []
    for b, Ib in enumerate(batches):
        if b < certify_from:                 # warm-up batches only feed the learner (their rows contribute >= 0)
            continue
        if b == 0 or learner == "static":
            G = E0[Ib]
        else:
            past = np.concatenate(batches[:b])
            if geometry_builder is None:
                raise ValueError("non-static geometry must be built outside Pythia and supplied as geometry_builder")
            G = np.asarray(geometry_builder(past, Ib, K, seed + b), float)
            if len(G) != len(Ib) or not np.isfinite(G).all():
                raise ValueError("geometry_builder returned invalid frozen geometry")
        Qc, Qt, pc, pt, ms, nb = _within_cell_stats(cells[Ib], G, y[Ib], K, rng)
        w = nb / n; Z += w * (Qc - Qt); prox += w ** 2 * (math.sqrt(pc) + math.sqrt(pt)) ** 2; mstar = min(mstar, ms)
        points.append(Qc - Qt)
    wtot = sum(len(b) for b in batches) / n
    dev = math.sqrt(math.log(1 / alpha) * prox / 2)
    return dict(dI_lower=max(((mstar - 1) / mstar) * (Z - dev), 0.0), batch_contrasts=points, point=Z)
