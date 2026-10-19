"""Development: B2 local curvature (density bound only near the quantile) on the weak designs."""
import sys, os, json, time, math
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import runpy
H = os.path.dirname(os.path.abspath(__file__))
S = runpy.run_path(os.path.join(H, "run_continuous_suite.py"), run_name="lib")
G = runpy.run_path(os.path.join(H, "run_geo_dev.py"), run_name="lib")
DES, noise, CAPS, KS = S["DES"], S["noise"], S["CAPS"], S["KS"]
from peras.core.smooth_dominated_saa import smooth_dominated_lower_bound, twonn_dimension
from scipy.stats import norm, t as tdist

def pdf(kind):
    if kind == "gauss": return norm.pdf
    if kind == "t3": return lambda x: tdist.pdf(x * math.sqrt(3), 3) * math.sqrt(3)
    if kind == "lognorm":
        m, v = math.exp(.5), (math.e - 1) * math.e; sd = math.sqrt(v)
        def f(x):
            u = m + sd * np.asarray(x, float); out = np.zeros_like(u)
            ok = u > 0; out[ok] = np.exp(-np.log(u[ok]) ** 2 / 2) / (u[ok] * math.sqrt(2 * math.pi)) * sd
            return out
        return f
    if kind == "bimodal": return lambda x: .5 * (norm.pdf((x - 1) / .3) + norm.pdf((x + 1) / .3)) / .3

def run(name, n, seed):
    d, f, Lf, sfun, Ls, nk = DES[name]; nz = noise(nk); dens = pdf(nk)
    r = np.random.default_rng(seed); X = r.uniform(0, 1, (n, d)); s = sfun(X); y = f(X) + s * nz["g"](r, n)
    A = np.arange(n) < n // 2; Bm = ~A
    models = G["fit_models"](X[A], y[A], d)
    out = []
    for loss, q, z, cc in (("mae", None, nz["med"], 2.), ("pinball", .9, nz["q9"], 1.)):
        key = "mae" if loss == "mae" else "q90"
        sB = s[Bm]; gold = float(np.mean(sB) * nz[key]); kap = cc * nz["fmax"] / sB
        Lraw = G["grad_fn_sup"](name, z, lambda P: np.zeros((len(P), d)), np.random.default_rng(seed + 1))
        cands = {"none": (Lraw, lambda X: 0 * X[:, 0])}
        for mname, (pred, grad) in models.items():
            cands[mname] = (G["grad_fn_sup"](name, z, grad, np.random.default_rng(seed + 1)), pred)
        pick = min(cands, key=lambda m: cands[m][0].sum()); Li, pred = cands[pick]
        XB = X[Bm] * Li; yB = y[Bm] - pred(X[Bm])
        geos = ("kd",) if twonn_dimension(XB) < 3 else ("matching", "balanced")
        grid = np.linspace(-1, 1, 41)
        def kappa_fn(rows, delta):
            sr = sB[rows]; w = delta / sr                         # drift in noise units
            pts = z + w[:, None] * grid[None, :]
            return cc * dens(pts).max(1) / sr * 1.02
        common = dict(loss=loss, L=1.0, kappa_rows=kap, block_sizes=KS, caps=CAPS, alpha=.05, quantile=q, seed=seed,
                      geometries=geos, metric="l1")
        b_g, _ = smooth_dominated_lower_bound(XB, yB, **common)
        b_l, _ = smooth_dominated_lower_bound(XB, yB, kappa_fn=kappa_fn, **common)
        out.append(dict(design=name, seed=seed, loss=key, gold=gold, global_density=b_g, local_density=b_l))
    return out

if __name__ == "__main__":
    reps, n, base, outp, names = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5].split(",")
    rows = []
    for nm in names:
        for rp in range(reps): rows += run(nm, n, base + rp)
        print(nm, flush=True)
    open(outp, "w").write("".join(json.dumps(x) + "\n" for x in rows))
