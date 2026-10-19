import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "research", "peras_v5", "nli_20261017"))
from methods import _loo_centred, nli_projection
from generator_ood import make_ood
from sklearn.feature_extraction.text import TfidfVectorizer


def _data(beta, seed, n=2000):
    d = make_ood("O1_grammar_severity", n, beta, seed)
    Xs = TfidfVectorizer(token_pattern=r"\S+", min_df=2).fit_transform(d["notes"])
    return d, Xs


def test_loo_centring_sums_to_zero():
    rng = np.random.default_rng(0); cell = rng.integers(0, 5, 200); q = rng.random(200); h = _loo_centred(cell, q)
    for c in range(5):
        assert abs(h[cell == c].sum()) < 1e-9


def test_projection_null_and_signal():
    d, Xs = _data(0.0, 1); o = nli_projection(d["cell"], Xs, d["y"], .05, 1, B=99)
    assert o["nli_proj_cert"] <= d["true_nli"] + 1e-12           # truth is 0 under the null
    d, Xs = _data(2.0, 2); o = nli_projection(d["cell"], Xs, d["y"], .05, 2, B=99)
    assert 0 < o["nli_proj_cert"] <= d["true_nli"] and o["nli_proj_perm_p"] <= .05
