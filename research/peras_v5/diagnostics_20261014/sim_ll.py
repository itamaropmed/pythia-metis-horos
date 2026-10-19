"""Round 2: local-polynomial block witnesses (candidate innovation).

Constant witness: inf_a mean_j loss(Y_j - a)                      -> drift penalty is FIRST order (gradient x spread)
Local-linear witness: inf_{a,b} mean_j loss(Y_j - a - b.z_j)       -> drift penalty is SECOND order (curvature)
Validity (location-scale units, independent):
  MAE/pinball: E[inf_{a,b} mean loss] <= inf_{a,b} mean_j E loss(Y_j - c_j) <= T + c_q * min_{a,b} mean_j |q_j - a - b.z_j|
               (triangle inequality; c_q = max(q,1-q); q_j = unit quantile). We use the LS-fit residual of q_j (>= the min) => valid.
  MSE: W = RSS/(k-p-1); E W = [sum sigma_j^2 (1-h_jj) + ||(I-H) mu||^2]/(k-p-1)  (exact) ; pen = E W - T (oracle exact).
"""
import sys, json, time
import numpy as np
import sim
from sklearn.ensemble import HistGradientBoostingRegressor

Q = 0.9


def ll_scores(Yb, Zb, loss):
    """Yb (m,k), Zb (m,k,p). Returns per-block witness. L1/pinball fits via IRLS-free LP-lite: use quantile regression by
    iteratively reweighted least squares (approximate inf -> still >= true inf => conservative? NO: approx inf >= true inf
    makes witness LARGER (anti-conservative). So we compute an exact LP solution with scipy.linprog."""
    from scipy.optimize import linprog
    m, k = Yb.shape
    p = Zb.shape[2]
    out = np.empty(m)
    if loss in ("mse",):
        for b in range(m):
            A = np.c_[np.ones(k), Zb[b]]
            coef, res, rk, _ = np.linalg.lstsq(A, Yb[b], rcond=None)
            r = Yb[b] - A @ coef
            out[b] = r @ r / (k - p - 1)
        return out
    q = 0.5 if loss == "mae" else Q
    w = 1.0 if loss == "mae" else 1.0  # mae = 2*rho_.5 ; handled below
    for b in range(m):
        A = np.c_[np.ones(k), Zb[b]]
        # variables: beta (p+1, free), u+ (k), u- (k); min sum q u+ + (1-q) u-  s.t. A beta + u+ - u- = y
        c = np.r_[np.zeros(p + 1), q * np.ones(k), (1 - q) * np.ones(k)]
        Aeq = np.c_[A, np.eye(k), -np.eye(k)]
        bounds = [(None, None)] * (p + 1) + [(0, None)] * (2 * k)
        res = linprog(c, A_eq=Aeq, b_eq=Yb[b], bounds=bounds, method="highs")
        v = res.fun / k
        out[b] = 2 * v if loss == "mae" else v
    return out


def run(name, n, seed, k_list=(8, 16)):
    rng = np.random.default_rng(seed)
    X, f, s, Y, noise = sim.draw(name, n, rng)
    nc = sim.noise_constants(noise)
    A = np.arange(n) < n // 2
    B = ~A
    Xa, Ya, Xb, Yb, fb, sb = X[A], Y[A], X[B], Y[B], f[B], s[B]
    mu_z, sd_z = Xa.mean(0), Xa.std(0) + 1e-12
    Zb = (Xb - mu_z) / sd_z
    hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(Xa, Ya)
    fhat_b = hgb.predict(Xb)
    eps_med = np.quantile(sim.noise_sampler(noise)(np.random.default_rng(3), 2_000_000), [0.5, Q])
    out = []
    configs = [("raw", Zb, Zb, None), ("fhatkey", fhat_b[:, None], fhat_b[:, None], None),
               ("raw_align", Zb, Zb, fhat_b)]
    if Zb.shape[1] > 5:   # high-d: local-linear in Z impossible with small k; use fhat as the regressor
        configs = [c for c in configs if c[0] == "fhatkey"]
    for cname, key, reg, shift in configs:
        y = Yb - (0 if shift is None else shift)
        mu = fb - (0 if shift is None else shift)
        for k in k_list:
            if reg.shape[1] + 2 >= k:
                continue
            blocks = sim.kd_blocks(key, k, np.random.default_rng(seed + 1))
            yb, zb, mub, sbb = y[blocks], reg[blocks], mu[blocks], sb[blocks]
            zb = zb - zb.mean(1, keepdims=True)
            for loss in ("mae", "q90", "mse"):
                for wit in ("const", "ll"):
                    if wit == "const":
                        S = sim.block_scores(yb, "mseu" if loss == "mse" else loss)
                    else:
                        S = ll_scores(yb, zb, loss)
                    # oracle penalties
                    if loss == "mse":
                        T = float(np.mean(sbb ** 2) * nc["mse"])
                        if wit == "const":
                            dm = mub - mub.mean(1, keepdims=True)
                            pen = k / (k - 1) * np.mean(dm ** 2)
                        else:
                            pens = []
                            for b in range(len(blocks)):
                                Am = np.c_[np.ones(k), zb[b]]
                                H = Am @ np.linalg.pinv(Am)
                                R = np.eye(k) - H
                                ew = (np.sum(sbb[b] ** 2 * nc["mse"] * (1 - np.diag(H))) + mub[b] @ R @ R @ mub[b]) / (k - Am.shape[1])
                                pens.append(ew - np.mean(sbb[b] ** 2) * nc["mse"])
                            pen = float(np.mean(pens))
                    else:
                        T = float(np.mean(sbb) * nc[loss])
                        qeps = eps_med[0] if loss == "mae" else eps_med[1]
                        qj = mub + sbb * qeps
                        if wit == "const":
                            Pabs, _, _ = sim.penalties(mub, sbb, nc["qgrid"], nc["mean"])
                            pen = (1.0 if loss == "mae" else 0.5) * Pabs
                        else:
                            res = []
                            for b in range(len(blocks)):
                                Am = np.c_[np.ones(k), zb[b]]
                                coef = np.linalg.lstsq(Am, qj[b], rcond=None)[0]
                                res.append(np.mean(np.abs(qj[b] - Am @ coef)))
                            cq = 1.0 if loss == "mae" else max(Q, 1 - Q)
                            pen = cq * float(np.mean(res))
                    cap = float(np.quantile(S, 0.99)) * 1.5  # NOTE: diagnostic cap from B (mild optimism); same for both witnesses
                    Sc = np.minimum(S, cap)
                    lcb = max(sim.bentkus_lower_mean(Sc, cap, 0.025), sim.dkelly_lower_mean(Sc, cap, 0.025))
                    bound = max(0.0, lcb - pen)
                    out.append(dict(mech=name, n=n, seed=seed, cfg=cname, k=k, loss=loss, wit=wit,
                                    meanS=float(S.mean()) / T, pen=pen / T, width=(float(Sc.mean()) - lcb) / T,
                                    ratio=bound / T, viol=bool(bound > T + 1e-12)))
    return out


if __name__ == "__main__":
    mechs = sys.argv[1].split(","); n = int(sys.argv[2]); seeds = int(sys.argv[3]); outp = sys.argv[4]
    t = time.time()
    with open(outp, "w") as fh:
        for nm in mechs:
            for sd in range(seeds):
                for r in run(nm, n, 2000 + sd):
                    fh.write(json.dumps(r) + "\n")
            print(nm, round(time.time() - t, 1), flush=True)
