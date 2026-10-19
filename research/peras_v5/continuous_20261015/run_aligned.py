"""Aligned variant: a frozen ridge-linear map g fitted on an INDEPENDENT half; certify on the other half
with outcomes y - g(x).  Declared modulus = Lipschitz of a*(x) - g(x) (computed from the simulator,
given identically to every method).  Baselines receive the same split, alignment and modulus."""
import sys, os, json, math, time
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
exec(open(os.path.join(os.path.dirname(__file__), "run_continuous_suite.py")).read().split("if __name__")[0])

def grad_sup(name, beta, z, rng):
    d, f, Lf, sfun, Ls, nk = DES[name]
    h = 1e-5
    P = np.vstack([rng.uniform(h, 1 - h, (200_000, d)), np.array(np.meshgrid(*[[h, 1 - h]] * d)).reshape(d, -1).T])
    g = np.zeros((len(P), d))
    for i in range(d):
        E = np.zeros(d); E[i] = h
        a1 = f(P + E) + sfun(P + E) * z
        a0 = f(P - E) + sfun(P - E) * z
        g[:, i] = (a1 - a0) / (2 * h)
    return 1.01 * float(np.sqrt(((g - beta) ** 2).sum(1)).max()), 1.01 * np.abs(g - beta).max(0)

def run_aligned(name, n, seed):
    d, f, Lf, sfun, Ls, nk = DES[name]; nz = noise(nk)
    r = np.random.default_rng(seed); X = r.uniform(0, 1, (n, d)); s = sfun(X); y = f(X) + s * nz["g"](r, n)
    A = np.arange(n) < n // 2; Bm = ~A
    Xa = np.c_[np.ones(A.sum()), X[A]]
    out = []
    for loss, q, z, cc in (("mae", None, nz["med"], 2.), ("pinball", .9, nz["q9"], 1.)):
        key = "mae" if loss == "mae" else "q90"
        # frozen linear map for the q-th conditional quantile: least squares on dev half shifted by quantile
        coef = np.linalg.solve(Xa.T @ Xa + 1e-6 * np.eye(d + 1), Xa.T @ y[A])
        beta = coef[1:]
        XB, yB, sB = X[Bm], y[Bm] - (coef[0] + X[Bm] @ beta), s[Bm]
        L2, Li = grad_sup(name, beta, z, np.random.default_rng(seed + 1))
        L_raw = Lf + Ls * abs(z)
        use_align = L2 < .5 * L_raw                       # frozen rule: declared moduli only
        if not use_align:
            XB, yB = X[Bm], y[Bm]
            _, Li = grad_sup(name, np.zeros(d), z, np.random.default_rng(seed + 1)); L2 = L_raw
        XB = XB * Li; L = 1.0; METRIC["m"] = "l1"     # anisotropic l1 metric with per-coordinate moduli
        gold = float(np.mean(sB) * nz[key]); kap = cc * nz["fmax"] / sB; cq = 1. if loss == "mae" else .9
        P = greedy_nn_pairs(XB)
        res = dict(design=name, n=n, seed=seed, loss=key, gold=gold, L_res=L2, L_raw=L_raw, aligned=bool(use_align))
        res["pair_first"] = accepted_bound(XB, yB, P, loss, q, L, kap, cq, "first", .05)
        res["pair_second"] = accepted_bound(XB, yB, P, loss, q, L, kap, cq, "second", .05)
        res["knn_group_first"] = max(accepted_bound(XB, yB, matching_blocks(XB, k), loss, q, L, kap, cq, "first", .05 / 5) for k in KS[1:])
        geos = ("kd",) if d <= 2 else ("matching",)
        b, info = smooth_dominated_lower_bound(XB, yB, loss=loss, L=L, kappa_rows=kap, block_sizes=KS, caps=CAPS,
                                               alpha=.05, quantile=q, seed=seed, geometries=geos, metric="l1")
        res["peras"] = b; res["peras_k"] = info["k"] if info else None
        out.append(res)
    return out

if __name__ == "__main__":
    reps, n, base, outp, names = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5].split(",")
    rows = []; t0 = time.time()
    for name in names:
        for rp in range(reps):
            rows += run_aligned(name, n, base + rp)
        print(name, round(time.time() - t0, 1), flush=True)
    open(outp, "w").write("".join(json.dumps(x) + "\n" for x in rows))
