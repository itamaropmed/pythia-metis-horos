import numpy as np
from pythia.core.agreement_field import afi_certificate


def _data(rep, text_signal):
    r = np.random.default_rng(rep); n = 3000; cell = r.integers(0, 6, n); E = r.normal(size=(n, 4))
    lin = .4 * (cell - 2.5) + (2.5 * (E[:, 0] > 0) - 1.25) * text_signal
    p = 1 / (1 + np.exp(-lin)); y = (r.random(n) < p).astype(int); return cell, E, y, p


def test_table_information_is_covered_and_no_false_text_information():
    for rep in range(6):
        cell, E, y, p = _data(rep, 0.0)
        c = afi_certificate(cell, E, y, 2, .05, levels=3, seed=rep)
        k = c["keep"]; pk = p[k]; ck = cell[k]
        pbar = np.array([pk[ck == j].mean() for j in ck]); IX = np.mean(2 * (pbar - pk.mean()) ** 2)
        assert c["I_X"][0] <= IX <= c["I_X"][1]
        assert c["dI_lower"] == 0.0


def test_detects_text_information_beyond_table():
    cell, E, y, p = _data(0, 1.0)
    assert afi_certificate(cell, E, y, 2, .05, levels=3)["dI_lower"] > 0
