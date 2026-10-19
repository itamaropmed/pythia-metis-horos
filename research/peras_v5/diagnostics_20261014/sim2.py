"""Round 2: penalty geometry.  Same blocks / witnesses / concentration; compare penalties for MAE & Q90:

  bary : barycentric coupling penalty (current Peras): c_L * mean int |F_j^-1 - Fbar^-1|                 FIRST order
  tri  : triangle penalty  c_q * mean_j |a*_j - abar|      (c_MAE = 1, c_q = max(q,1-q))                  FIRST order
  curv : curvature penalty (kappa/2) * mean_j (a*_j - abar)^2, kappa = sup density * (2 for MAE, 1 pinball)  SECOND order
         (per block we use min(tri, curv): both are valid upper bounds on the same Jensen gap)
  jensen: exact R(block mixture) - mean_j R(Q_j)  (oracle; the sharpest possible penalty for this witness)

All follow from  E[witness_b] <= inf_a mean_j E l(Y_j - a) = R(mixture_b)  (needs NO within-block independence),
and  E l(Y_j - a) <= R_j + (kappa_j/2)(a - a*_j)^2  since the expected loss has second derivative = (2x) density.
Also the local-linear witness (raw Z, d<=5) with the curvature penalty on the LS residual of a*_j.
"""
import sys, json, time
import numpy as np
import sim, sim_ll
from sklearn.ensemble import HistGradientBoostingRegressor

FMAX = {"gauss": 1 / np.sqrt(2 * np.pi), "t3": 0.36755 * np.sqrt(3), "lognorm": np.exp(0.5) / np.sqrt(2 * np.pi) * np.sqrt((np.e - 1) * np.e)}
QG = (np.arange(1000) + 0.5) / 1000


def rho(d, q):
    return np.where(d >= 0, q * d, (q - 1) * d)


def jensen_gap(mub, sb, qe, q, mult):
    """exact R(mixture) - mean R_j per block, averaged. qe: noise quantiles on QG."""
    v = mub[..., None] + sb[..., None] * qe[None, None, :]          # (m,k,G)
    m = v.shape[0]
    pooled = v.reshape(m, -1)
    a = np.quantile(pooled, q, axis=1)
    Rmix = mult * rho(pooled - a[:, None], q).mean(1)
    aj = mub + sb * np.quantile(qe, q)
    Rj = mult * rho(v - aj[..., None], q).mean(-1).mean(1)
    return float(np.mean(Rmix - Rj))


def run(name, n, seed):
    rng = np.random.default_rng(seed)
    X, f, s, Y, noise = sim.draw(name, n, rng)
    nc = sim.noise_constants(noise)
    e = sim.noise_sampler(noise)(np.random.default_rng(5), 1_000_000)
    qe = np.quantile(e, QG)
    A = np.arange(n) < n // 2; B = ~A
    Xa, Ya, Xb, Yb, fb, sb = X[A], Y[A], X[B], Y[B], f[B], s[B]
    Zb = (Xb - Xa.mean(0)) / (Xa.std(0) + 1e-12)
    hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(Xa, Ya)
    fhat_b = hgb.predict(Xb)
    out = []
    cfgs = [("raw_kd", Zb, None, (2, 8, 32)), ("learned_key", fhat_b[:, None], None, (2, 8, 32)),
            ("raw_align", Zb, fhat_b, (2, 8, 32)), ("oracle_key", np.c_[fb, sb], None, (8,))]
    for cname, key, shift, ks in cfgs:
        y = Yb - (0 if shift is None else shift)
        mu = fb - (0 if shift is None else shift)
        for k in ks:
            blocks = sim.kd_blocks(key, k, np.random.default_rng(seed + 1))
            yb, mub, sbb = y[blocks], mu[blocks], sb[blocks]
            wits = [("const", None)]
            if Zb.shape[1] <= 5 and k >= 16 and cname in ("raw_kd", "raw_align"):
                zz = Zb[blocks]; wits.append(("ll", zz - zz.mean(1, keepdims=True)))
            for loss, q, mult, cq in (("mae", 0.5, 2.0, 1.0), ("q90", 0.9, 1.0, 0.9)):
                T = float(np.mean(sbb) * nc[loss])
                astar = mub + sbb * np.quantile(e, q)
                kappa = (2.0 if loss == "mae" else 1.0) * FMAX[noise] / sbb.min(1)      # per-block sup curvature
                for wname, zz in wits:
                    S = sim.block_scores(yb, loss) if zz is None else sim_ll.ll_scores(yb, zz, loss)
                    cap = float(np.quantile(S, 0.99)) * 1.5
                    Sc = np.minimum(S, cap)
                    lcb = max(sim.bentkus_lower_mean(Sc, cap, 0.025), sim.dkelly_lower_mean(Sc, cap, 0.025))
                    if zz is None:
                        dev = astar - astar.mean(1, keepdims=True)
                        P, _, _ = sim.penalties(mub, sbb, nc["qgrid"], nc["mean"])
                        pens = dict(bary=(1.0 if loss == "mae" else 0.5) * P,
                                    tri=cq * float(np.mean(np.abs(dev))),
                                    curv=float(np.mean(np.minimum(cq * np.abs(dev).mean(1), kappa / 2 * (dev ** 2).mean(1)))),
                                    jensen=jensen_gap(mub, sbb, qe, q, mult))
                    else:
                        res = np.empty_like(astar)
                        for b in range(len(blocks)):
                            Am = np.c_[np.ones(k), zz[b]]
                            res[b] = astar[b] - Am @ np.linalg.lstsq(Am, astar[b], rcond=None)[0]
                        pens = dict(tri=cq * float(np.mean(np.abs(res))),
                                    curv=float(np.mean(np.minimum(cq * np.abs(res).mean(1), kappa / 2 * (res ** 2).mean(1)))))
                    for pn, pv in pens.items():
                        b = max(0.0, lcb - pv)
                        out.append(dict(mech=name, n=n, seed=seed, cfg=cname, k=k, wit=wname, loss=loss, pen_type=pn,
                                        pen=pv / T, meanS=float(S.mean()) / T, lcb=lcb / T, ratio=b / T,
                                        tip=(lcb / pv if pv > 0 else 1e9), viol=bool(b > T + 1e-12)))
    return out


if __name__ == "__main__":
    mechs = sys.argv[1].split(","); n = int(sys.argv[2]); seeds = int(sys.argv[3]); outp = sys.argv[4]
    t = time.time()
    with open(outp, "w") as fh:
        for nm in mechs:
            for sd in range(seeds):
                for r in run(nm, n, 3000 + sd):
                    fh.write(json.dumps(r) + "\n")
            print(nm, round(time.time() - t, 1), flush=True)
