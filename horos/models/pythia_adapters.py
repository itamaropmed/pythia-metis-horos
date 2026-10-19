"""Training-side adapters that feed frozen predictions/geometries to Pythia."""
from __future__ import annotations

import numpy as np
from scipy.sparse import hstack, csr_matrix, issparse
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NeighborhoodComponentsAnalysis

from pythia.core.dynamic_geometry import dynamic_certificate as _dynamic_certificate
from pythia.core.unified import unified_certificate as _unified_certificate
from pythia.core.unified import stratified_certificate as _stratified_certificate


def _prediction_builder(features, labels, factory=None):
    def build(train_rows, test_rows, classes):
        q = np.zeros((len(test_rows), classes))
        if len(np.unique(labels[train_rows])) < 2:
            q[:] = ((np.bincount(labels[train_rows], minlength=classes) + .5)
                    / (len(train_rows) + .5 * classes))
            return q
        model = (factory() if factory else LogisticRegression(C=2, max_iter=3000))
        model.fit(features[train_rows], labels[train_rows])
        q[:, model.classes_] = model.predict_proba(features[test_rows])
        return q
    return build


def unified_certificate(cells, E, F, y, K, alpha=.05, gamma=.5, levels=3,
                        seed=0, model_factory=None, population=True,
                        models=None, nli_concentration="hoeffding"):
    """Horos fits cross-fold scores; Pythia receives only frozen builders."""
    specs = models or [(F, model_factory)]
    builders = [_prediction_builder(features, np.asarray(y), factory)
                for features, factory in specs]
    return _unified_certificate(
        cells, E, None, y, K, alpha, gamma, levels, seed,
        population=population, nli_concentration=nli_concentration,
        prediction_builders=builders,
    )


def stratified_certificate(strata, cells, E, F, y, K, alpha=.05, gamma=.5,
                           levels=3, seed=0, min_rows=128,
                           model_factory=None, models=None,
                           nli_concentration="hoeffding"):
    specs = models or [(F, model_factory)]
    builders = [_prediction_builder(features, np.asarray(y), factory)
                for features, factory in specs]
    return _stratified_certificate(
        strata, cells, E, None, y, K, alpha, gamma, levels, seed,
        min_rows=min_rows, nli_concentration=nli_concentration,
        prediction_builders=builders,
    )


def dynamic_certificate(cells, E0, y, K, alpha=.05, B=3,
                        learner="posterior", F=None, seed=0, fracs=None,
                        certify_from=0):
    """Build predictable learned geometry in Horos, certify it in Pythia."""
    if learner == "static":
        return _dynamic_certificate(cells, E0, y, K, alpha, B, "static",
                                    seed=seed, fracs=fracs,
                                    certify_from=certify_from)
    features = E0 if F is None else F
    cells_arr, labels = np.asarray(cells), np.asarray(y)
    onehot = np.eye(cells_arr.max() + 1)[cells_arr]

    def build(past, current, classes, local_seed):
        if learner == "posterior":
            fx = (lambda rows: hstack([features[rows], csr_matrix(onehot[rows])]).tocsr()) \
                if issparse(features) else (lambda rows: np.hstack([features[rows], onehot[rows]]))
            model = LogisticRegression(C=2, max_iter=3000)
            model.fit(fx(past), labels[past])
            q = np.zeros((len(current), classes))
            q[:, model.classes_] = model.predict_proba(fx(current))
            return q
        if learner == "nca":
            rng = np.random.default_rng(local_seed)
            sub = past if len(past) <= 2500 else rng.choice(past, 2500, replace=False)
            model = NeighborhoodComponentsAnalysis(
                n_components=min(16, E0.shape[1]), max_iter=60,
                random_state=local_seed,
            )
            model.fit(E0[sub], labels[sub])
            return model.transform(E0[current])
        raise ValueError("learner must be static, posterior, or nca")

    return _dynamic_certificate(
        cells, E0, y, K, alpha, B, learner, seed=seed, fracs=fracs,
        certify_from=certify_from, geometry_builder=build,
    )
