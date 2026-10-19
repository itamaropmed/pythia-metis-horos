import sys, os, numpy as np
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "research", "peras_v5", "afi_20261017"))
from pythia.core.unified import nli_cohort_bound, loo_centre, table_box
from horos.models.pythia_adapters import unified_certificate
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def test_vector_nli_matches_binary_identity():
    rng = np.random.default_rng(0); n = 400; cell = rng.integers(0, 4, n); q = rng.random(n); y = (rng.random(n) < q).astype(int)
    Q = np.column_stack([1 - q, q]); H = loo_centre(cell, Q)
    for c in range(4):
        assert np.allclose(H[cell == c].sum(0), 0)
    b, info = nli_cohort_bound(cell, Q, y, .05)
    assert b >= 0 and abs(info["C"] - 2 * (y * H[:, 1]).mean()) < 1e-12


def test_table_box_exact_mean_and_unified_validity():
    from generator import make
    d = make("D02_novel_findings", 3000, 11); y, cell, P, K = d["y"], d["cell"], d["P"], d["K"]
    Xs = TfidfVectorizer(token_pattern=r"\S+", min_df=2).fit_transform(d["notes"])
    E = TruncatedSVD(32, random_state=0).fit_transform(Xs); E /= np.linalg.norm(E, axis=1, keepdims=True)
    F = hstack([csr_matrix(np.eye(cell.max() + 1)[cell]), Xs]).tocsr()
    u = unified_certificate(cell, E, F, y, K, .05, .5, 3, 0)
    S = u["rows"]; G = (1 - (P[S] ** 2).sum(1)).mean(); V = 0.0
    for c in np.unique(cell[S]):
        I = S[cell[S] == c]; m = len(I); V += m / (m - 1) * ((P[I] - P[I].mean(0)) ** 2).sum()
    V /= len(S)
    assert u["G"][0] <= G <= u["G"][1] and u["V_lower"] <= V and u["table"][0] <= G + V <= u["table"][1]


def test_nli_concentration_routes_are_finite_and_portfolio_is_declared():
    rng = np.random.default_rng(445)
    n, K = 500, 3
    cell = np.repeat(np.arange(25), n // 25)
    Q = rng.dirichlet(np.ones(K), size=n)
    y = np.array([rng.choice(K, p=q) for q in Q])
    for route in ("hoeffding", "bentkus", "empirical_bernstein",
                  "heterogeneous_chernoff", "stratified_bentkus", "safe_portfolio"):
        b, info = nli_cohort_bound(cell, Q, y, .05, concentration=route)
        assert np.isfinite(b) and b >= 0
        assert info["concentration"] == route
        assert info["selected_concentration"] in {
            "hoeffding", "bentkus", "empirical_bernstein",
            "heterogeneous_chernoff", "stratified_bentkus"
        }
        assert info["C_lower"] <= info["C"] + 1e-12
    _, pinfo = nli_cohort_bound(cell, Q, y, .05, concentration="safe_portfolio")
    assert set(pinfo["candidates"]) == {
        "hoeffding", "bentkus", "empirical_bernstein",
        "heterogeneous_chernoff", "stratified_bentkus"
    }


def test_nli_unknown_concentration_refuses():
    cell = np.repeat(np.arange(2), 4)
    Q = np.tile([[.8, .2], [.2, .8]], (4, 1))
    y = np.arange(8) % 2
    with np.testing.assert_raises(ValueError):
        nli_cohort_bound(cell, Q, y, .05, concentration="peek_then_choose")
