"""Legacy estimator-based cross-fitting, owned by Horos rather than Pythia.

Pythia consumes the resulting frozen :class:`pythia.contracts.FoldMatrix`; it
never calls estimators or trains models itself.

S3 -- cross-fitting.

Two things come out of this stage and nothing else in the system produces them:

  * ``L``  -- the best honest out-of-fold risk, computed downstream from ``P_oof``
  * the **disagreement object** -- the whole basis of the bound

Folds are grouped by the entity key by default. Rows sharing an entity (a
patient, a session, a user) share history, so ungrouped folds leak that history
across the split and both L and the disagreement are then optimistic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold

from pythia.contracts import FoldMatrix


@dataclass
class Learner:
    name: str
    estimator: object
    family: str = "misc"          # 'gbm' | 'forest' | 'linear' | ...
    predict: str = "predict"      # or 'predict_proba' for probability targets


def _predict(est, X, how: str) -> np.ndarray:
    if how == "predict_proba":
        p = est.predict_proba(X)
        return np.asarray(p)[:, 1]
    return np.asarray(est.predict(X)).ravel()


def preflight(learners: list[Learner], bayes_act: str) -> list[str]:
    """Check C1 -- the learner's internal objective must match the loss.

    A squared-error learner estimates the conditional *mean*. Used under an
    absolute loss, whose Bayes act is the conditional *median*, its mean-median
    gap enters the disagreement statistic as if it were estimation error, and on
    a skewed target that gap can be large. Returns a list of warnings.
    """
    warn = []
    for lr in learners:
        crit = (getattr(lr.estimator, "loss", None)
                or getattr(lr.estimator, "criterion", None)
                or getattr(lr.estimator, "objective", None))
        crit = str(crit).lower() if crit is not None else ""
        if bayes_act == "median" and ("squared" in crit or "friedman" in crit or crit == ""):
            warn.append(
                f"C1: {lr.name!r} appears to optimise squared error (criterion={crit!r}) "
                f"while the loss targets the conditional median."
            )
        if bayes_act == "mean" and "absolute" in crit:
            warn.append(
                f"C1: {lr.name!r} optimises absolute error while the loss targets the "
                f"conditional mean."
            )
    return warn


def cross_fit(X, y, learners: list[Learner], groups=None,
              n_splits: int = 3, seed: int = 0) -> FoldMatrix:
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n = y.size

    if groups is not None:
        groups = np.asarray(groups)
        n_splits = int(min(n_splits, len(np.unique(groups))))
        splitter = GroupKFold(n_splits=n_splits)
        folds = list(splitter.split(X, y, groups=groups))
        grouped = True
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(X, y))
        grouped = False

    fold_id = np.empty(n, dtype=int)
    for k, (_, te) in enumerate(folds):
        fold_id[te] = k

    P = np.zeros((len(learners), n))
    for b, lr in enumerate(learners):
        for tr, te in folds:
            est = clone(lr.estimator)
            est.fit(X[tr], y[tr])
            P[b, te] = _predict(est, X[te], lr.predict)

    return FoldMatrix(
        y=y,
        P_oof=P,
        names=tuple(lr.name for lr in learners),
        families=tuple(lr.family for lr in learners),
        fold_id=fold_id,
        groups=groups,
        meta={"seed": seed, "n": n, "p": X.shape[1],
              "n_splits": n_splits, "grouped": grouped},
    )
