"""Development (not confirmation): geometry upgrades on regime A.
 variants: base   = frozen PERAS-S regime A (linear alignment, kd/matching rule)
           bal    = + balanced OT blocks (kd/matching and balanced in the ledger)
           krr    = + smooth kernel-ridge alignment (RBF) fitted on half A when its declared residual
                    modulus is smaller than the linear one's
           both   = bal + krr
 All moduli are computed from the simulator (the fairness/declaration contract)."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import runpy
NS = runpy.run_path(os.path.join(os.path.dirname(__file__), "run_continuous_suite.py"), run_name="lib")
DES, noise, CAPS, KS = NS["DES"], NS["noise"], NS["CAPS"], NS["KS"]
from peras.core.smooth_dominated_saa import smooth_dominated_lower_bound
from sklearn.kernel_ridge import KernelRidge

def grad_fn_sup(name, z, grad_g, rng, n=60_000):
    d, f, Lf, sfun, Ls, nk = DES[name]; h = 1e-5
    P = np.vstack([rng.uniform(h, 1 - h, (n, d)), np.array(np.meshgrid(*[[h, 1 - h]] * d)).reshape(d, -1).T])
    g = np.zeros((len(P), d))
    for i in range(d):
        E = np.zeros(d); E[i] = h
        g[:, i] = ((f(P + E) + sfun(P + E) * z) - (f(P - E) + sfun(P - E) * z)) / (2 * h)
    r = g - grad_g(P)
    return 1.01 * np.abs(r).max(0)            # per-coordinate moduli

def fit_models(XA, yA, d):
    Xa = np.c_[np.ones(len(XA)), XA]
    coef = np.linalg.solve(Xa.T @ Xa + 1e-6 * np.eye(d + 1), Xa.T @ yA)
    lin = (lambda X: coef[0] + X @ coef[1:], lambda P: np.tile(coef[1:], (len(P), 1)))
    gam = 2.0 / d
    kr = KernelRidge(kernel="rbf", gamma=gam, alpha=1.0).fit(XA, yA)
    A, Xt = kr.dual_coef_.ravel(), XA
    def kpred(X): return kr.predict(X)
    def kgrad(P):
        out = np.zeros((len(P), d))
        for s in range(0, len(P), 4000):
            Q = P[s:s + 4000]
            K = np.exp(-gam * ((Q[:, None, :] - Xt[None, :, :]) ** 2).sum(-1))
            out[s:s + 4000] = -2 * gam * ((K * A[None, :])[:, :, None] * (Q[:, None, :] - Xt[None, :, :])).sum(1)
        return out
    return {"linear": lin, "krr": (kpred, kgrad)}

def run(name, n, seed):
    d, f, Lf, sfun, Ls, nk = DES[name]; nz = noise(nk)
    r = np.random.default_rng(seed); X = r.uniform(0, 1, (n, d)); s = sfun(X); y = f(X) + s * nz["g"](r, n)
    A = np.arange(n) < n // 2; Bm = ~A
    models = fit_models(X[A], y[A], d)
    out = []
    for loss, q, z, cc in (("mae", None, nz["med"], 2.), ("pinball", .9, nz["q9"], 1.)):
        key = "mae" if loss == "mae" else "q90"
        sB = s[Bm]; gold = float(np.mean(sB) * nz[key]); kap = cc * nz["fmax"] / sB
        Lraw_i = grad_fn_sup(name, z, lambda P: np.zeros((len(P), d)), np.random.default_rng(seed + 1))
        cands = {"none": (Lraw_i, lambda X: 0 * X[:, 0])}
        for mname, (pred, grad) in models.items():
            cands[mname] = (grad_fn_sup(name, z, grad, np.random.default_rng(seed + 1)), pred)
        row = dict(design=name, n=n, seed=seed, loss=key, gold=gold)
        for variant in ("base", "bal", "krr", "both"):
            allowed = ["none", "linear"] + (["krr"] if variant in ("krr", "both") else [])
            # frozen rule: smallest declared total modulus among allowed alignments (base: linear only if < .5 raw)
            if variant in ("base", "bal"):
                pick = "linear" if cands["linear"][0].sum() < .5 * Lraw_i.sum() else "none"
            else:
                pick = min(allowed, key=lambda m: cands[m][0].sum())
            Li, pred = cands[pick]
            XB = X[Bm] * Li; yB = y[Bm] - pred(X[Bm])
            geos = (("kd",) if d <= 2 else ("matching",)) + (("balanced",) if variant in ("bal", "both") else ())
            b, info = smooth_dominated_lower_bound(XB, yB, loss=loss, L=1.0, kappa_rows=kap, block_sizes=KS, caps=CAPS,
                                                   alpha=.05, quantile=q, seed=seed, geometries=geos, metric="l1")
            row[variant] = b; row[variant + "_align"] = pick; row[variant + "_geo"] = info["geometry"] if info else None
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
