"""Development: B1 local modulus field.  Same frozen geometry rule as PERAS-S-G, but the penalty uses
per-block declared moduli L_{b,i} = sup over the block bounding box of |d(a* - g)/dx_i| (simulator truth,
x1.05, corners + 96 random points) instead of one global modulus."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import runpy
H = os.path.dirname(os.path.abspath(__file__))
S = runpy.run_path(os.path.join(H, "run_continuous_suite.py"), run_name="lib")
G = runpy.run_path(os.path.join(H, "run_geo_dev.py"), run_name="lib")
DES, noise, CAPS, KS = S["DES"], S["noise"], S["CAPS"], S["KS"]
from peras.core.smooth_dominated_saa import smooth_dominated_lower_bound, twonn_dimension

def resid_grad(name, z, grad_g):
    d, f, Lf, sfun, Ls, nk = DES[name]; h = 1e-5
    def g(P):
        out = np.zeros((len(P), d))
        for i in range(d):
            E = np.zeros(d); E[i] = h
            out[:, i] = ((f(P + E) + sfun(P + E) * z) - (f(P - E) + sfun(P - E) * z)) / (2 * h)
        return out - grad_g(P)
    return g

def run(name, n, seed):
    d, f, Lf, sfun, Ls, nk = DES[name]; nz = noise(nk)
    r = np.random.default_rng(seed); X = r.uniform(0, 1, (n, d)); s = sfun(X); y = f(X) + s * nz["g"](r, n)
    A = np.arange(n) < n // 2; Bm = ~A
    models = G["fit_models"](X[A], y[A], d)
    rr = np.random.default_rng(seed + 9)
    out = []
    for loss, q, z, cc in (("mae", None, nz["med"], 2.), ("pinball", .9, nz["q9"], 1.)):
        key = "mae" if loss == "mae" else "q90"
        sB = s[Bm]; gold = float(np.mean(sB) * nz[key]); kap = cc * nz["fmax"] / sB
        Lraw = G["grad_fn_sup"](name, z, lambda P: np.zeros((len(P), d)), np.random.default_rng(seed + 1))
        cands = {"none": (Lraw, lambda X: 0 * X[:, 0], lambda P: np.zeros((len(P), d)))}
        for mname, (pred, grad) in models.items():
            cands[mname] = (G["grad_fn_sup"](name, z, grad, np.random.default_rng(seed + 1)), pred, grad)
        pick = min(cands, key=lambda m: cands[m][0].sum())
        Li, pred, grad = cands[pick]
        Xraw = X[Bm]; XB = Xraw * Li; yB = y[Bm] - pred(Xraw)
        geos = ("kd",) if twonn_dimension(XB) < 3 else ("matching", "balanced")
        rg = resid_grad(name, z, grad)
        corners = np.array(np.meshgrid(*[[0, 1]] * d)).reshape(d, -1).T if d <= 6 else rr.integers(0, 2, (64, d))
        def local_moduli(B):
            Lb = np.zeros((len(B), d))
            for j, b in enumerate(B):
                lo, hi = Xraw[b].min(0), Xraw[b].max(0)
                P = np.vstack([lo + corners * (hi - lo), lo + rr.uniform(size=(96, d)) * (hi - lo)])
                P = np.clip(P, 1e-5, 1 - 1e-5)
                Lb[j] = 1.05 * np.abs(rg(P)).max(0)
            return Lb
        row = dict(design=name, n=n, seed=seed, loss=key, gold=gold, align=pick)
        b_glob, _ = smooth_dominated_lower_bound(XB, yB, loss=loss, L=1.0, kappa_rows=kap, block_sizes=KS, caps=CAPS,
                                                 alpha=.05, quantile=q, seed=seed, geometries=geos, metric="l1")
        # local field: blocks built on the same scaled geometry, penalty from local moduli on raw coordinates
        # (penalty function receives scaled X; we pass Xraw for distances via a wrapper)
        from peras.core import smooth_dominated_saa as SD
        orig = SD.smooth_penalty
        def pen_raw(Xs, B, L, kr, cq, order="best", metric="l2", local_moduli=None):
            return orig(Xraw, B, L, kr, cq, order, metric, local_moduli)
        SD.smooth_penalty = pen_raw
        try:
            b_loc, _ = SD.smooth_dominated_lower_bound(XB, yB, loss=loss, L=1.0, kappa_rows=kap, block_sizes=KS, caps=CAPS,
                                                       alpha=.05, quantile=q, seed=seed, geometries=geos, metric="l1",
                                                       local_moduli=local_moduli)
        finally:
            SD.smooth_penalty = orig
        row.update(global_modulus=b_glob, local_field=b_loc)
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
