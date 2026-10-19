"""Known-signal / known-noise diagnostics for Peras-style training-free floor certificates.

Model: Y = f(X) + s(X) * eps, eps standardized (mean 0, var 1), independent units.
Fixed-design target T = mean_j R(Q_j) over certified units (exactly computable).
Certificate (one route): LCB_Bentkus(mean_b min(S_b, cap)) - c_L * P_true  (barycentric theorem, penalty at TRUE
discrepancy => oracle-discrepancy check; this isolates every OTHER loss).
"""
from __future__ import annotations
import sys, json, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
sys.path.insert(0, "/home/claude/audit")
from peras.core.concentration import bentkus_lower_mean

Q = 0.9
UGRID = (np.arange(4000) + 0.5) / 4000


# ---------------------------------------------------------------- noise laws
def noise_sampler(kind):
    if kind == "gauss":
        return lambda r, n: r.standard_normal(n)
    if kind == "t3":
        return lambda r, n: r.standard_t(3, n) / np.sqrt(3.0)
    if kind == "lognorm":
        m, v = np.exp(0.5), (np.exp(1) - 1) * np.exp(1)
        return lambda r, n: (np.exp(r.standard_normal(n)) - m) / np.sqrt(v)
    raise ValueError(kind)


NOISE_CONST = {}


def noise_constants(kind):
    if kind in NOISE_CONST:
        return NOISE_CONST[kind]
    r = np.random.default_rng(7)
    e = noise_sampler(kind)(r, 4_000_000)
    med, q90 = np.median(e), np.quantile(e, Q)
    d = e - q90
    c = dict(mae=float(np.mean(np.abs(e - med))),
             q90=float(np.mean(np.where(d >= 0, Q * d, (Q - 1) * d))),
             mse=float(np.var(e)),
             qgrid=np.quantile(e, UGRID), mean=float(e.mean()))
    # exact-replicate optimism ratios E[witness]/R for each k
    opt = {}
    for k in (2, 4, 8, 16, 32):
        b = e[: (4_000_000 // k) * k].reshape(-1, k)[:100_000]
        opt[k] = {L: float(np.mean(block_scores(b, L)) / c[L if L != "mseu" else "mse"])
                  for L in ("mae", "q90", "mse", "mseu")}
    c["opt"] = opt
    NOISE_CONST[kind] = c
    return c


# ---------------------------------------------------------------- witnesses
def block_scores(Yb, loss):
    k = Yb.shape[1]
    if loss == "mae":
        return np.mean(np.abs(Yb - np.median(Yb, axis=1, keepdims=True)), axis=1)
    if loss == "q90":
        srt = np.sort(Yb, axis=1)
        a = srt[:, int(np.ceil(Q * k)) - 1][:, None]
        d = Yb - a
        return np.mean(np.where(d >= 0, Q * d, (Q - 1) * d), axis=1)
    if loss == "mse":
        return np.var(Yb, axis=1)
    if loss == "mseu":
        return np.var(Yb, axis=1, ddof=1)
    raise ValueError(loss)


def penalties(mu, s, qgrid, qmean):
    """mu, s: (m,k) residual-law location/scale per unit. Returns P (abs barycentric), P2 (sq), Dmu."""
    dm = mu - mu.mean(1, keepdims=True)
    ds = s - s.mean(1, keepdims=True)
    # int |dm + ds*q(u)| du per unit, averaged
    P = np.mean(np.abs(dm[..., None] + ds[..., None] * qgrid[None, None, :]).mean(-1))
    Eq2 = 1.0 + qmean ** 2
    P2 = np.mean(dm ** 2 + 2 * dm * ds * qmean + ds ** 2 * Eq2)
    Dmu = np.mean(dm ** 2)
    return float(P), float(P2), float(Dmu)


# ---------------------------------------------------------------- blocking (outcome-blind)
def dkelly_lower_mean(values, cap, alpha, G=40):
    """Diversified-Kelly fixed-time LCB on the AVERAGE mean of independent [0,cap] scores.
    E_m = mean_j prod_t (1 + (f_j/m)(x_t - m)), f_j in (0,1): each product is a supermartingale-type e-value
    under H0: avg E x_t <= m (AM-GM), the mixture is an e-value; reject when E_m >= 1/alpha."""
    x = np.clip(np.asarray(values, float) / cap, 0, 1)
    fr = np.geomspace(1e-3, 0.99, G)
    def logE(m):
        lk = np.log1p(np.outer(fr / m, x - m)).sum(1)
        mx = lk.max()
        return mx + np.log(np.mean(np.exp(lk - mx)))
    hi = float(x.mean())
    if hi <= 0:
        return 0.0
    lo, thr = 1e-9, np.log(1 / alpha)
    if logE(lo) < thr:
        return 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if logE(mid) >= thr:
            lo = mid
        else:
            hi = mid
    return float(lo * cap)


def kd_blocks(Z, k, rng):
    n = (len(Z) // k) * k
    idx = rng.permutation(len(Z))[:n]
    Z = np.asarray(Z, float).reshape(len(Z), -1)
    out = []
    stack = [idx]
    while stack:
        I = stack.pop()
        if len(I) == k:
            out.append(I); continue
        j = int(np.argmax(Z[I].var(0)))
        I = I[np.argsort(Z[I, j], kind="stable")]
        cut = k * ((len(I) // k) // 2)
        stack += [I[:cut], I[cut:]]
    return np.array(out)


# ---------------------------------------------------------------- mechanisms
def mech(name):
    lin = lambda X: X[:, :5] @ np.ones(5) * np.sqrt(3 / 5)          # var 1 on U[-1,1]
    one = lambda X: np.ones(len(X))
    M = {
        "linear5":   dict(d=5,  f=lin, s=lambda X: 0.5 * one(X), noise="gauss"),
        "friedman5": dict(d=5,  f=lambda X: (np.sin(np.pi * (X[:, 0] + 1) * (X[:, 1] + 1) / 4) * 2
                                              + 2 * ((X[:, 2] + 1) / 2 - .5) ** 2 * 4 + X[:, 3] + .5 * X[:, 4]),
                          s=lambda X: 0.5 * one(X), noise="gauss"),
        "sparse20":  dict(d=20, f=lambda X: X[:, 0] * 1.5 + np.sin(np.pi * X[:, 1]),
                          s=lambda X: 0.5 * one(X), noise="gauss"),
        "hifreq2":   dict(d=2,  f=lambda X: np.sin(4 * np.pi * X[:, 0]) + np.sin(4 * np.pi * X[:, 1]),
                          s=lambda X: 0.5 * one(X), noise="gauss"),
        "hetero5":   dict(d=5,  f=lin, s=lambda X: 0.15 + 0.85 * np.abs(X[:, 0]), noise="gauss"),
        "t3_5":      dict(d=5,  f=lin, s=lambda X: 0.5 * one(X), noise="t3"),
        "skew5":     dict(d=5,  f=lin, s=lambda X: 0.5 * one(X), noise="lognorm"),
        "discrete3": dict(d=3,  f=None, s=lambda X: 0.5 * one(X), noise="gauss", discrete=5),
        "highSNR5":  dict(d=5,  f=lin, s=lambda X: 0.1 * one(X), noise="gauss"),
        "lowSNR5":   dict(d=5,  f=lin, s=lambda X: 2.0 * one(X), noise="gauss"),
    }
    return M[name]


def draw(name, n, rng):
    m = mech(name)
    if m.get("discrete"):
        L = m["discrete"]
        X = rng.integers(0, L, size=(n, m["d"])).astype(float)
        tab = np.random.default_rng(99).standard_normal((L,) * m["d"])
        f = tab[tuple(X.astype(int).T)]
    else:
        X = rng.uniform(-1, 1, size=(n, m["d"]))
        f = m["f"](X)
    s = m["s"](X)
    e = noise_sampler(m["noise"])(rng, n)
    return X, f, s, f + s * e, m["noise"]


# ---------------------------------------------------------------- one certificate
CL = {"mae": 1.0, "q90": 0.5}


def certify(Y, mu, s, blocks, loss, cap, nc, alpha=0.05):
    Yb, mub, sb = Y[blocks], mu[blocks], s[blocks]
    k = blocks.shape[1]
    S = block_scores(Yb, loss)
    Sc = np.minimum(S, cap)
    lcb_b = bentkus_lower_mean(Sc, cap, alpha / 2)
    lcb_k = dkelly_lower_mean(Sc, cap, alpha / 2)
    lcb = max(lcb_b, lcb_k)
    P, P2, Dmu = penalties(mub, sb, nc["qgrid"], nc["mean"])
    if loss in ("mae", "q90"):
        pen = CL[loss] * P
        T = float(np.mean(sb) * nc[loss])
    elif loss == "mse":
        pen = P2; T = float(np.mean(sb ** 2) * nc["mse"])
    else:  # mseu: exact for independent units
        pen = k / (k - 1) * Dmu; T = float(np.mean(sb ** 2) * nc["mse"])
    bound = max(0.0, lcb - pen)
    return dict(T=T, bound=bound, ratio=bound / T, meanS=float(S.mean()) / T, meanSc=float(Sc.mean()) / T,
                lcb=lcb / T, pen=pen / T, width=(float(Sc.mean()) - lcb) / T,
                tip=(lcb / pen if pen > 0 else np.inf), kelly_win=bool(lcb_k > lcb_b), viol=bool(bound > T + 1e-12),
                opt=nc["opt"][k][loss], k=k, m=len(blocks))


def run(name, n, seed, losses=("mae", "q90", "mse", "mseu")):
    rng = np.random.default_rng(seed)
    X, f, s, Y, noise = draw(name, n, rng)
    nc = noise_constants(noise)
    A = np.arange(n) < n // 2
    B = ~A
    Xa, Ya, Xb, Yb, fb, sb = X[A], Y[A], X[B], Y[B], f[B], s[B]
    Za = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-12)
    Zb = (Xb - Xa.mean(0)) / (Xa.std(0) + 1e-12)
    # learned location map + learned scale map on half A only (independent of B)
    hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
    fhat_a = cross_val_predict(hgb, Xa, Ya, cv=5)
    hgb.fit(Xa, Ya)
    fhat_b = hgb.predict(Xb)
    sh = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0)
    sh.fit(Xa, np.abs(Ya - fhat_a))
    shat_b = sh.predict(Xb)
    out = []
    strategies = [
        # name, blocking key, outcome shift (frozen), ks
        ("raw_kd", Zb, None, (2, 8)),
        ("oracle_key", np.c_[fb, sb], None, (2, 8)),
        ("learned_key", np.c_[fhat_b], None, (2, 8)),
        ("raw_align", Zb, fhat_b, (2, 8)),
        ("learned_align", np.c_[shat_b], fhat_b, (8, 32)),
        ("oracle_align", np.c_[sb], fb, (8, 32)),
    ]
    for sname, key, shift, ks in strategies:
        y = Yb - (0 if shift is None else shift)
        mu = fb - (0 if shift is None else shift)
        for k in ks:
            blocks = kd_blocks(key, k, np.random.default_rng(seed + 1))
            for L in losses:
                # cap: from half A only (in-sample scores on A with same construction, generous)
                ya = Ya - (0 if shift is None else fhat_a)
                ba = kd_blocks(np.c_[ya * 0 + np.arange(len(ya))], k, np.random.default_rng(1))  # random A blocks
                cap = float(np.quantile(block_scores(ya[ba], L), 0.99)) * 1.5
                r = certify(y, mu, sb, blocks, L, cap, nc)
                r.update(mech=name, n=n, seed=seed, strat=sname, loss=L)
                out.append(r)
    # estimate tier: NN residual-variance (Devroye et al.) on raw Z and on learned key, MSE only
    from sklearn.neighbors import NearestNeighbors
    T_mse = float(np.mean(sb ** 2) * nc["mse"])
    for ename, key in (("nn_raw", Zb), ("nn_learned", np.c_[fhat_b])):
        nn = NearestNeighbors(n_neighbors=2).fit(key)
        j = nn.kneighbors(key, return_distance=False)[:, 1]
        out.append(dict(mech=name, n=n, seed=seed, strat=ename, loss="mse_est",
                        ratio=float(np.mean((Yb - Yb[j]) ** 2) / 2 / T_mse)))
    oof = float(np.mean((Yb - fhat_b) ** 2) / T_mse)
    out.append(dict(mech=name, n=n, seed=seed, strat="hgb_oof", loss="mse_est", ratio=oof))
    return out


if __name__ == "__main__":
    mechs = sys.argv[1].split(",")
    ns = [int(v) for v in sys.argv[2].split(",")]
    seeds = range(int(sys.argv[3]))
    outp = sys.argv[4]
    t = time.time()
    with open(outp, "w") as fh:
        for nm in mechs:
            for n in ns:
                for sd in seeds:
                    for r in run(nm, n, 1000 + sd):
                        fh.write(json.dumps(r, default=float) + "\n")
            print(nm, round(time.time() - t, 1), flush=True)
