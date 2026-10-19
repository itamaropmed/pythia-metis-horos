"""Fair continuous-feature suite (Codex ten-design style): Peras smooth-domination vs accepted pair/kNN
lower bounds.  ALL methods receive the same declared (true) Lipschitz constant L of the Bayes action
and the same density/curvature bound; same caps; same alpha .05; rows iid."""
import sys, json, math, time
import numpy as np
sys.path.insert(0, __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), "../../..")))
from peras.core.smooth_dominated_saa import smooth_dominated_lower_bound, kd_blocks, matching_blocks
from peras.core.dominated_saa import truncated_cell_sum, bentkus_lower_from_statistic
from peras.core.truncated_saa import dkelly_lower_mean
from scipy.stats import norm, t as tdist

Z = {}
def noise(kind):
    if kind == "gauss": g = lambda r, n: r.standard_normal(n); fmax = 1/math.sqrt(2*math.pi)
    elif kind == "t3": g = lambda r, n: r.standard_t(3, n)/math.sqrt(3); fmax = tdist.pdf(0, 3)*math.sqrt(3)
    elif kind == "lognorm":
        m, v = math.exp(.5), (math.e-1)*math.e
        g = lambda r, n: (np.exp(r.standard_normal(n)) - m)/math.sqrt(v); fmax = math.exp(.5)/math.sqrt(2*math.pi)*math.sqrt(v)
    elif kind == "bimodal":
        g = lambda r, n: r.choice((-1., 1.), n) + .3*r.standard_normal(n); fmax = .5/(.3*math.sqrt(2*math.pi)) + 1e-3
    if kind not in Z:
        e = np.sort(g(np.random.default_rng(1), 2_000_000))
        med, q9 = np.median(e), np.quantile(e, .9)
        Z[kind] = dict(g=g, fmax=fmax, med=med, q9=q9, mae=float(np.mean(np.abs(e-med))),
                       q90=float(np.mean(np.where(e >= q9, .9*(e-q9), .1*(q9-e)))))
    return Z[kind]

S2 = math.sqrt(2)
DES = {  # name: (d, f, grad-norm sup Lf, s(x), Ls (sup |grad s|), noise)
 "linear1":   (1, lambda X: 2*X[:,0], 2.0, lambda X: .5+0*X[:,0], 0.0, "gauss"),
 "sine1":     (1, lambda X: np.sin(2*np.pi*X[:,0]), 2*np.pi, lambda X: .5+0*X[:,0], 0.0, "gauss"),
 "hetero1":   (1, lambda X: X[:,0], 1.0, lambda X: .2+.6*X[:,0], .6, "gauss"),
 "smooth2":   (2, lambda X: X[:,0]+np.sin(np.pi*X[:,1]), math.sqrt(1+np.pi**2), lambda X: .5+0*X[:,0], 0.0, "gauss"),
 "linear5":   (5, lambda X: X.sum(1)*2/math.sqrt(5), 2.0, lambda X: .5+0*X[:,0], 0.0, "gauss"),
 "quad2":     (2, lambda X: 2*(X[:,0]**2+X[:,1]**2), 4*S2, lambda X: .5+0*X[:,0], 0.0, "gauss"),
 "interact2": (2, lambda X: 3*X[:,0]*X[:,1], 3*S2, lambda X: .5+0*X[:,0], 0.0, "gauss"),
 "bimodal1":  (1, lambda X: 2*X[:,0], 2.0, lambda X: .5+0*X[:,0], 0.0, "bimodal"),
 "t3_2":      (2, lambda X: X[:,0]+X[:,1], S2, lambda X: .5+0*X[:,0], 0.0, "t3"),
 "skew1":     (1, lambda X: 2*X[:,0], 2.0, lambda X: .5+0*X[:,0], 0.0, "lognorm"),
 "sparse10":  (10, lambda X: 2*X[:,0]+np.sin(np.pi*X[:,1]), math.sqrt(4+np.pi**2), lambda X: .5+0*X[:,0], 0.0, "gauss"),
}
CAPS = (.25, .5, 1., 2., 4., 8.)
KS = (2, 4, 8, 16, 32, 64)

def greedy_nn_pairs(X):
    n = len(X); D = ((X[:, None, :]-X[None, :, :])**2).sum(-1); iu = np.triu_indices(n, 1)
    order = np.argsort(D[iu], kind="stable"); used = np.zeros(n, bool); P = []
    for e in order:
        i, j = iu[0][e], iu[1][e]
        if not used[i] and not used[j]:
            used[i] = used[j] = True; P.append((i, j))
            if len(P) == n//2: break
    return np.array(P)

def block_scores(y, B, loss, tau, q):
    return np.array([truncated_cell_sum(y[b], loss=loss, kappa=tau, quantile=q) for b in B]) / B.shape[1]

METRIC = {"m": "l2"}
def pen_rows(X, B, L, kap, cq, order):
    Xb = X[B]; c = Xb.mean(1, keepdims=True)
    d = np.sqrt(((Xb-c)**2).sum(-1)) if METRIC["m"] == "l2" else np.abs(Xb-c).sum(-1)
    f = cq*L*d; s = kap[B]/2*(L*d)**2
    return {"first": f, "second": np.minimum(f, s)}[order].sum()

def accepted_bound(X, y, B, loss, q, L, kap, cq, order, alpha):
    """independent block scores + best of Bentkus/dKelly (alpha split), cap grid Bonferroni, penalty."""
    n = len(y); m, k = B.shape
    pen = pen_rows(X, B, L, kap, cq, order) / n
    a = alpha / (len(CAPS)*2); best = 0.
    for tau in CAPS:
        sc = block_scores(y, B, loss, tau, q)                     # in [0, tau]
        frac = m*k/n
        for lcb in (bentkus_lower_from_statistic(sc.mean(), m, tau, a), dkelly_lower_mean(sc, cap=tau, alpha=a)):
            best = max(best, frac*lcb - pen)
    return max(0., best)

def run(name, n, seed):
    d, f, Lf, sfun, Ls, nk = DES[name]; nz = noise(nk)
    r = np.random.default_rng(seed); X = r.uniform(0, 1, (n, d)); s = sfun(X)
    y = f(X) + s*nz["g"](r, n)
    out = []
    for loss, q, z, cc in (("mae", None, nz["med"], 2.), ("pinball", .9, nz["q9"], 1.)):
        key = "mae" if loss == "mae" else "q90"
        gold = float(np.mean(s)*nz[key]); L = Lf + Ls*abs(z); kap = cc*nz["fmax"]/s
        cq = 1. if loss == "mae" else .9
        P = greedy_nn_pairs(X)
        res = dict(design=name, n=n, seed=seed, loss=key, gold=gold)
        res["pair_first"] = accepted_bound(X, y, P, loss, q, L, kap, cq, "first", .05)          # accepted standard
        res["pair_second"] = accepted_bound(X, y, P, loss, q, L, kap, cq, "second", .05)        # upgraded accepted
        g1 = max(accepted_bound(X, y, matching_blocks(X, k), loss, q, L, kap, cq, "first", .05/len(KS)) for k in KS[1:])
        res["knn_group_first"] = g1
        geos = ("kd",) if d <= 2 else ("matching",)      # frozen outcome-blind rule (dev 2026-10-15)
        b, info = smooth_dominated_lower_bound(X, y, loss=loss, L=L, kappa_rows=kap, block_sizes=KS, caps=CAPS,
                                               alpha=.05, quantile=q, seed=seed, geometries=geos)
        res["peras"] = b; res["peras_k"] = info["k"] if info else None
        out.append(res)
    return out

if __name__ == "__main__":
    reps, ns, base, outp = int(sys.argv[1]), [int(v) for v in sys.argv[2].split(",")], int(sys.argv[3]), sys.argv[4]
    rows = []; t0 = time.time()
    names = sys.argv[5].split(",") if len(sys.argv) > 5 else list(DES)
    for di, name in enumerate(DES):
        if name not in names: continue
        for n in ns:
            for rp in range(reps):
                rows += run(name, n, base + 1000*di + rp)
        print(name, round(time.time()-t0, 1), flush=True)
    open(outp, "w").write("".join(json.dumps(x)+"\n" for x in rows))
