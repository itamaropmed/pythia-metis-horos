"""Jev reader: response -> probability features (no network)."""
import numpy as np
from horos.represent.jev_reader import SURGICAL_QUESTIONS, feature_names, to_features


def test_to_features_shapes_and_missing():
    ans = {"anesthesia": {"type": "choice", "choice": "general",
                          "probabilities": {"general": .9, "regional": .05, "local_mac": .05, "not_stated": 0}},
           "complication": {"type": "noul", "noul": .1},
           "blood_loss": {"type": "score", "probabilities": {"0": .7, "1": .3, "2": 0, "3": 0}}}
    x = to_features(ans)
    assert len(x) == len(feature_names()) == 4 + 4 + 1 + 1 + 4
    assert x[0] == .9 and x[feature_names().index("complication:P(true)")] == .1
    assert np.allclose(x[feature_names().index("emergency:P(true)")], .5)        # missing -> uninformative
    assert np.allclose(to_features(None)[:4], .25)
