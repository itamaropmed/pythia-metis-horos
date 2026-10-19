"""The Bregman branch -- exact, guard-free.

The identity
----------------------------------------------------------------------------
For a Bregman divergence ``D`` with Bayes act ``m(X) = E[Y | X]``, every
predictor ``f`` satisfies

    E[D(Y, f(X))]  =  E[D(Y, m(X))]  +  E[D(m(X), f(X))]
                      \\_____________/    \\_________________/
                          R*                 excess risk

exactly -- no approximation, no shape assumption, no envelope. So

    R*  =  Risk(f)  -  E[D(m, f)] .

The observable
----------------------------------------------------------------------------
``E[D(m, f)]`` involves the unknown ``m``. What *is* observable is the
disagreement between two cross-fitted predictors. Writing the symmetrised
divergence ``Dsym(a, b) = D(a, b) + D(b, a)``, a second-order expansion gives
``Dsym(a, b) ~ phi''(.) (a - b)^2`` and ``D(m, f) ~ (1/2) phi''(.) (m - f)^2``,
so under **fold-error orthogonality** -- E[(f1 - m)(f2 - m)] = 0, the A-fold
assumption --

    E[D(m, f)]  =  (1/4) E[Dsym(f1, f2)] .

For squared loss this is an identity, not an approximation:
``Dsym(f1, f2) = 2 (f1 - f2)^2`` and ``(1/4) * 2 E[(f1-f2)^2] = E[(f1-m)^2]``.

Bound vs estimator
----------------------------------------------------------------------------
``B = L - Delta`` is a **lower** bound on R* exactly when ``Delta`` **over**-states
the best predictor's excess risk. So:

    bound      Delta_max   -- the largest pairwise disagreement  (conservative)
    estimator  Delta_mean  -- the mean pairwise disagreement     (central)

What this branch does NOT escape
----------------------------------------------------------------------------
A-fold. Two predictors that share a systematic bias disagree very little while
both being far from ``m``; the bound is then too high. There is no
assumption-free repair -- two identical models have zero disagreement and
arbitrary excess risk -- so the honest move is to make A-fold explicit, report a
diagnostic for it, and keep heterogeneous learners in the pool.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .losses import Loss, get_loss


def _pairwise_delta(P: np.ndarray, loss: Loss) -> np.ndarray:
    """Delta estimate from every ordered pair of predictors, shape (n_pairs,)."""
    D = loss.divergence
    out = []
    for i, j in combinations(range(P.shape[0]), 2):
        dsym = D(P[i], P[j]) + D(P[j], P[i])
        out.append(0.25 * float(np.mean(dsym)))
    return np.asarray(out, float)


def _family_of(names: list[str] | None, n: int) -> list[str]:
    if names is None:
        return [f"m{i}" for i in range(n)]
    return list(names)


def fold_orthogonality_proxy(P: np.ndarray, loss: Loss,
                             families: list[str] | None) -> float | None:
    """A cheap read on A-fold.

    Predictors from the *same* family share inductive bias; predictors from
    *different* families share less of it. If within-family pairs disagree far
    less than cross-family pairs, some of the shared error is invisible to the
    disagreement statistic and the bound is optimistic.

    Returns ``1 - mean(within) / mean(cross)`` in [0, 1): 0 is reassuring,
    values approaching 1 mean the disagreement statistic is mostly measuring
    family differences rather than estimation error. ``None`` when there is not
    at least one within-family and one cross-family pair.
    """
    fam = _family_of(families, P.shape[0])
    D = loss.divergence
    within, cross = [], []
    for i, j in combinations(range(P.shape[0]), 2):
        d = 0.25 * float(np.mean(D(P[i], P[j]) + D(P[j], P[i])))
        (within if fam[i] == fam[j] else cross).append(d)
    if not within or not cross:
        return None
    mc = float(np.mean(cross))
    if mc <= 0:
        return None
    return float(max(0.0, 1.0 - float(np.mean(within)) / mc))


def bregman_bound(y: np.ndarray,
                  P: np.ndarray,
                  loss: str | Loss = "mse",
                  families: list[str] | None = None,
                  predictor_names: list[str] | None = None) -> dict:
    """Certified floor and attainability estimate for a Bregman loss.

    Parameters
    ----------
    y : (n,)            observed target
    P : (B, n)          out-of-fold predictions, one row per predictor.
                        Every entry must come from a model that never saw that row.
    loss :              a name from ``pythia.core.losses`` in the Bregman family
    families :          per-predictor family label (e.g. 'gbm', 'forest', 'linear');
                        used only for the A-fold diagnostic
    """
    loss = loss if isinstance(loss, Loss) else get_loss(loss)
    if not loss.is_bregman:
        raise ValueError(f"{loss.name!r} is not in the Bregman family -- use the guarded branch")

    y = np.asarray(y, float).ravel()
    P = np.atleast_2d(np.asarray(P, float))
    if P.shape[1] != y.shape[0]:
        raise ValueError(f"P has {P.shape[1]} columns but y has {y.shape[0]} rows")
    if P.shape[0] < 2:
        raise ValueError("need at least two cross-fitted predictors to observe disagreement")

    risks = np.array([loss.risk(y, P[b]) for b in range(P.shape[0])])
    k = int(np.argmin(risks))
    L = float(risks[k])

    deltas = _pairwise_delta(P, loss)
    delta_max = float(deltas.max())
    delta_mean = float(deltas.mean())

    B = max(L - delta_max, 0.0)
    E = max(L - delta_mean, 0.0)

    # pool health: the bound uses the LARGEST pairwise disagreement, so a predictor
    # far off the pace inflates Delta and can drive B to zero. Competitive pools only.
    pool_health = float(risks.max() / risks.min()) if risks.min() > 0 else float("inf")
    warnings = []
    if pool_health > 2.0:
        warnings.append(
            f"pool health {pool_health:.1f}x: the worst predictor's risk is far above the "
            f"best. The bound uses the largest pairwise disagreement, so an off-the-pace "
            f"predictor makes it vacuous. Drop it or improve it."
        )

    zero_info = loss.zero_info(y)
    mp = loss.monotone_map

    def rep(v: float) -> float:
        return float(mp(v)) if mp is not None else float(v)

    # experimental: identify (Delta, lambda) from risks + pairwise disagreement
    try:
        from .folddep import pair_delta_map, solve_delta_lambda
        sol = solve_delta_lambda(risks, pair_delta_map(P, loss.divergence))
    except Exception:                                       # pragma: no cover
        sol = {"identified": False}

    names = predictor_names or [f"m{i}" for i in range(P.shape[0])]
    return {
        "loss": loss.name,
        "regime": "bregman",
        "n": int(y.size),
        "n_predictors": int(P.shape[0]),
        "best_predictor": names[k],
        "L": rep(L),
        "bound": rep(B),
        "estimator": rep(E),
        "gamma": rep(E) - rep(B),
        "zero_info": rep(zero_info),
        "dispersion_removed": (1.0 - L / zero_info) if zero_info > 0 else float("nan"),
        "delta_max": delta_max,
        "delta_mean": delta_mean,
        "per_predictor_risk": {n: rep(r) for n, r in zip(names, risks)},
        "assumptions": ["A-fold: fold-error orthogonality"],
        "diagnostics": {
            "fold_orthogonality_proxy": fold_orthogonality_proxy(P, loss, families),
            "delta_spread": float(deltas.max() - deltas.min()),
            "exact": loss.name in ("mse", "rmse"),
            # EXPERIMENTAL -- see docs/experiments/M0_FINDINGS.md. The (Delta, lambda) fit is
            # only weakly identified; reported for research, never used for B.
            "lambda_hat": sol.get("lambda"),
            "delta_fit": sol.get("delta"),
            "delta_conservative": sol.get("delta_conservative"),
            "folddep_rmse": sol.get("rmse"),
            "folddep_identified": sol.get("identified", False),
            "pool_health": pool_health,
        },
        "warnings": warnings,
        "rung": "orthogonal",
        "guard": None,
        "coverage_guarantee": None,
    }


def cluster_bootstrap(y: np.ndarray,
                      P: np.ndarray,
                      groups: np.ndarray | None,
                      loss: str | Loss = "mse",
                      reps: int = 400,
                      seed: int = 0,
                      alpha: float = 0.05) -> dict:
    """Percentile intervals for L, the bound and the estimator.

    Resamples the *grouping key* (patient, session, entity) with replacement --
    never rows -- because rows sharing a group have correlated errors.
    """
    y = np.asarray(y, float).ravel()
    P = np.atleast_2d(np.asarray(P, float))
    groups = np.arange(y.size) if groups is None else np.asarray(groups)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.default_rng(seed)

    keys = ("L", "bound", "estimator")
    draws = {k: np.empty(reps) for k in keys}
    for r in range(reps):
        pick = rng.choice(ug, size=ug.size, replace=True)
        sel = np.concatenate([idx[g] for g in pick])
        res = bregman_bound(y[sel], P[:, sel], loss)
        for k in keys:
            draws[k][r] = res[k]

    point = bregman_bound(y, P, loss)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        k: {
            "point": point[k],
            "lo": float(np.percentile(draws[k], lo)),
            "hi": float(np.percentile(draws[k], hi)),
            "q05": float(np.percentile(draws[k], 5)),
        }
        for k in keys
    }
