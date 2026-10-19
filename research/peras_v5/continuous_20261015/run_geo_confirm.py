"""Frozen confirmation of the geometry rule (GEOMETRY_CONFIRMATION_REGISTRATION.md)."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import runpy
H = os.path.dirname(os.path.abspath(__file__))
S = runpy.run_path(os.path.join(H, "run_continuous_suite.py"), run_name="lib")
G = runpy.run_path(os.path.join(H, "run_geo_dev.py"), run_name="lib")
DES, noise, CAPS, KS = S["DES"], S["noise"], S["CAPS"], S["KS"]
accepted_bound, greedy_nn_pairs, METRIC = S["accepted_bound"], S["greedy_nn_pairs"], S["METRIC"]
from peras.core.smooth_dominated_saa import smooth_dominated_lower_bound, matching_blocks, twonn_dimension

def run(name, n, seed):
    d, f, Lf, sfun, Ls, nk = DES[name]; nz = noise(nk)
    r = np.random.default_rng(seed); X = r.uniform(0, 1, (n, d)); s = sfun(X); y = f(X) + s * nz["g"](r, n)
    A = np.arange(n) < n // 2; Bm = ~A
    models = G["fit_models"](X[A], y[A], d)
    out = []
    for loss, q, z, cc in (("mae", None, nz["med"], 2.), ("pinball", .9, nz["q9"], 1.)):
        key = "mae" if loss == "mae" else "q90"
        sB = s[Bm]; gold = float(np.mean(sB) * nz[key]); kap = cc * nz["fmax"] / sB; cq = 1. if loss == "mae" else .9
        Lraw = G["grad_fn_sup"](name, z, lambda P: np.zeros((len(P), d)), np.random.default_rng(seed + 1))
        cands = {"none": (Lraw, lambda X: 0 * X[:, 0])}
        for mname, (pred, grad) in models.items():
            cands[mname] = (G["grad_fn_sup"](name, z, grad, np.random.default_rng(seed + 1)), pred)
        row = dict(design=name, n=n, seed=seed, loss=key, gold=gold)
        # --- frozen regime-A PERAS-S (previous confirmation) and the accepted routes, on its alignment
        pick0 = "linear" if cands["linear"][0].sum() < .5 * Lraw.sum() else "none"
        Li, pred = cands[pick0]; XB = X[Bm] * Li; yB = y[Bm] - pred(X[Bm]); METRIC["m"] = "l1"
        b0, _ = smooth_dominated_lower_bound(XB, yB, loss=loss, L=1.0, kappa_rows=kap, block_sizes=KS, caps=CAPS, alpha=.05,
                                             quantile=q, seed=seed, geometries=(("kd",) if d <= 2 else ("matching",)), metric="l1")
        P = greedy_nn_pairs(XB)
        row["frozen_peras"] = b0
        row["pair_first"] = accepted_bound(XB, yB, P, loss, q, 1.0, kap, cq, "first", .05)
        row["pair_second"] = accepted_bound(XB, yB, P, loss, q, 1.0, kap, cq, "second", .05)
        row["knn_group_first"] = max(accepted_bound(XB, yB, matching_blocks(XB, k), loss, q, 1.0, kap, cq, "first", .05 / 5) for k in KS[1:])
        # --- new frozen rule
        pick = min(cands, key=lambda m: cands[m][0].sum())
        Li, pred = cands[pick]; XB = X[Bm] * Li; yB = y[Bm] - pred(X[Bm])
        dim = twonn_dimension(XB)
        geos = ("kd",) if dim < 3 else ("matching", "balanced")
        b1, info = smooth_dominated_lower_bound(XB, yB, loss=loss, L=1.0, kappa_rows=kap, block_sizes=KS, caps=CAPS, alpha=.05,
                                                quantile=q, seed=seed, geometries=geos, metric="l1")
        row.update(new_peras=b1, new_align=pick, twonn=dim, new_geo=info["geometry"] if info else None)
        # accepted routes on the new alignment too (fairness: they may also benefit)
        P = greedy_nn_pairs(XB)
        row["pair_second_newalign"] = accepted_bound(XB, yB, P, loss, q, 1.0, kap, cq, "second", .05)
        row["knn_group_newalign"] = max(accepted_bound(XB, yB, matching_blocks(XB, k), loss, q, 1.0, kap, cq, "first", .05 / 5) for k in KS[1:])
        out.append(row)
    return out

if __name__ == "__main__":
    reps, n, base, outp, names = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5].split(",")
    rows = []; t0 = time.time()
    for nm in names:
        for rp in range(reps):
            rows += run(nm, n, base + rp)
        print(nm, round(time.time() - t0, 1), flush=True)
    open(outp, "w").write("".join(json.dumps(x) + "\n" for x in rows))
