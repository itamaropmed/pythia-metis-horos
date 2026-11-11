"""Metis reader gain: gate accepts an informative reader, rejects noise; refinement certifies a real risk split."""
import numpy as np
from metis.reader_gain import gate_readers, refine_and_certify


def test_gate_and_refine():
    r = np.random.default_rng(0); n = 6000
    cells = r.integers(0, 4, n).astype(str); z = r.random(n) < .3
    p = np.where(z, .45, .12); miss = (r.random(n) < p).astype(int)
    good = np.column_stack([1 - p, p]); noise = np.full((n, 2), .5) + r.normal(0, .01, (n, 2))
    res = {g.reader: g for g in gate_readers(cells, miss, {"good": good, "noise": noise})}
    assert res["good"].accepted and not res["noise"].accepted
    out = refine_and_certify({"cell": cells}, z.astype(int), np.arange(n).astype(str), miss.astype(float),
                             min_entities=300)
    assert out["relation_counts"].get("certified_higher_than_global", 0) > 0
