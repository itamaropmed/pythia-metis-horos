"""Known-law check of the barycentric penalty theorem (exact T, Monte-Carlo E S with 2e5 tuples)."""
import numpy as np
from core import erm_risk
from scipy import stats
rng = np.random.default_rng(1)
bad = 0; rows = []
for trial in range(120):
    k = int(rng.choice([2, 3, 4, 8])); mu = rng.normal(0, rng.uniform(0.1, 3), k); sd = rng.uniform(.5, 2, k)
    fam = rng.choice(["norm", "laplace", "gamma"])
    def ppf(u, j):
        if fam == "norm": return mu[j] + sd[j] * stats.norm.ppf(u)
        if fam == "laplace": return mu[j] + sd[j] * stats.laplace.ppf(u)
        return mu[j] + sd[j] * stats.gamma(2).ppf(u)
    U = rng.random((100000, k)); Y = np.column_stack([ppf(U[:, j], j) for j in range(k)])        # independent evaluation
    u = rng.random((100000, 1)); V = np.column_stack([ppf(u[:, 0], j) for j in range(k)])        # comonotone coupling
    dev = V - V.mean(1, keepdims=True); pair = np.mean([np.abs(V[:, i] - V[:, j]).mean() for i in range(k) for j in range(k)]) 
    for L, lip in [("MAE", 1.0), ("Q90", .5), ("MSE", None)]:
        ug = (np.arange(2_000_000) + .5) / 2_000_000; z = ppf(ug, 0) - mu[0]; z = z / sd[0]      # standardized quantile grid
        if L == "MSE": base = z.var()
        elif L == "MAE": base = np.abs(z - np.quantile(z, .5)).mean()
        else: a = np.quantile(z, .9); base = np.mean(np.maximum(.9 * (z - a), -.1 * (z - a)))
        T = np.mean(base * (sd ** 2 if L == "MSE" else sd))                                          # exact Bayes risk (quadrature)
        ES = erm_risk(L, Y).mean()
        pen = (dev ** 2).mean() if L == "MSE" else lip * np.abs(dev).mean()
        pen_pair = None if L == "MSE" else (1.0 if L == "MAE" else .9) * pair
        if ES - pen > T + 1e-3 * T: bad += 1
        rows.append((L, k, (ES - pen) / T, None if pen_pair is None else (ES - pen_pair) / T))
print("violations", bad, "of", len(rows))
for L in ["MAE", "Q90", "MSE"]:
    r = np.array([x[2] for x in rows if x[0] == L]); p = np.array([x[3] for x in rows if x[0] == L and x[3] is not None])
    print(L, "bary (E S - pen)/T median %.3f max %.3f" % (np.median(r), r.max()), "| pairwise median %.3f" % np.median(p) if len(p) else "")
