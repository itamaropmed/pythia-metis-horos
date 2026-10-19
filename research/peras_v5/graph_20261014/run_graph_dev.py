"""Known-floor stress: surgeon-day shocks, patient repeats, time drift (nonstationary)."""
import sys, json, math, time
import numpy as np
sys.path.insert(0, ".")
from peras.core.graph_dominated_saa import graph_dominated_saa_lower_bound
from peras.core.dominated_saa import dominated_saa_lower_bound
from peras.core.concentration import bentkus_lower_mean

CAPS = (1., 2., 4., 8., 16.)

def design(rng, shock_sd, drift, lo=2, hi=7, days=120, S=12):
    rows = []
    pid = 0; prev = []
    for d in range(days):
        for s in range(S):
            if rng.random() < .6:
                for j in range(int(rng.integers(lo, hi))):
                    if prev and rng.random() < .15:
                        p = int(rng.choice(prev))
                    else:
                        p = pid; pid += 1; prev.append(p)
                    c = int(rng.integers(0, 6))
                    rows.append((d, s, p, c))
    d, s, p, c = map(np.array, zip(*rows))
    n = len(d)
    sd_key = d * 100 + s
    _, sd_inv = np.unique(sd_key, return_inverse=True)
    shock = rng.standard_normal(sd_inv.max() + 1) * shock_sd
    _, p_inv = np.unique(p, return_inverse=True)
    peff = rng.standard_normal(p_inv.max() + 1) * .3
    mu = 2.0 * c + drift * (d / days) * (1 + .3 * c)          # nonstationary cell means
    noise = rng.standard_t(4, n) / math.sqrt(2) * (.6 + .1 * c)
    y = mu + shock[sd_inv] + peff[p_inv] + noise
    return dict(day=d, surgeon=s, patient=p, cell=c, y=y, mu=mu, scale=.6 + .1 * c, shock_sd=shock_sd)

def zsample(shock_sd, scale, m=400_000, seed=5):
    r = np.random.default_rng(seed)
    return np.sort(r.standard_normal(m) * shock_sd + r.standard_normal(m) * .3 + r.standard_t(4, m) / math.sqrt(2) * scale)

def pooled_floor(D):
    """realized-period floor for MAE: (1/n) sum_c min_a sum_{i in c} E|mu_i + Z_c - a|"""
    tot = 0.0
    for c in np.unique(D["cell"]):
        m = D["cell"] == c
        z = zsample(D["shock_sd"], .6 + .1 * c)
        P = np.concatenate([[0], np.cumsum(z)]); N = len(z)
        mu = D["mu"][m]
        def f(a):
            t = a - mu                                   # E|Z - t|
            k = np.searchsorted(z, t)
            below = t * k - P[k]; above = (P[N] - P[k]) - t * (N - k)
            return np.sum((below + above) / N)
        lo, hi = mu.min() - 10, mu.max() + 10
        for _ in range(100):                               # golden section (convex)
            a1 = lo + .382 * (hi - lo); a2 = lo + .618 * (hi - lo)
            if f(a1) < f(a2): hi = a2
            else: lo = a1
        tot += f(.5 * (lo + hi))
    return tot / len(D["y"])

reps = int(sys.argv[1]); out = sys.argv[2]
rows = []; t = time.time()
SCEN = {"no_shock": (0.0, 0.0, 2, 7), "drift_only": (0.0, 3.0, 2, 7), "shock_moderate": (.6, 3.0, 2, 7),
        "shock_strong": (1.2, 3.0, 2, 7), "shock_dominant_big_days": (3.0, 3.0, 8, 15)}
for si, (scen, (shock_sd, drift, lo, hi)) in enumerate(SCEN.items()):
    for r in range(reps):
        rng = np.random.default_rng(int(sys.argv[3]) + 10_000 * si + r)
        D = design(rng, shock_sd, drift, lo, hi, days=(120 if hi <= 7 else 40))
        gold = pooled_floor(D)
        lab = D["cell"]; y = D["y"]; n = len(y)
        sday = np.char.add(D["day"].astype(str), "_" + D["surgeon"].astype(str))
        common = dict(loss="mae", caps=CAPS, alpha=.05, representation_provenance="sim", outcome_blind_labels_attested=True)
        g = graph_dominated_saa_lower_bound(lab, y, {"patient": D["patient"], "surgeon_day": sday},
                                            dependency_graph_attested=True, retention_grid=(None,), **common)
        gp = graph_dominated_saa_lower_bound(lab, y, {"patient": D["patient"]}, dependency_graph_attested=True, **common)
        pc = dominated_saa_lower_bound(lab, y, D["patient"], inequalities=("bentkus",), independent_clusters_attested=True, **common)
        gr = graph_dominated_saa_lower_bound(lab, y, {"row": np.arange(n)}, dependency_graph_attested=True, **common)
        rows.append(dict(scenario=scen, rep=r, n=n, gold=gold, graph=g.bound, K=g.diagnostics["selected_colours"],
                         omega=g.diagnostics["selected_max_clique"], graph_patient_only=gp.bound,
                         patient_bentkus=pc.bound, row_iid=gr.bound))
    print(scen, round(time.time() - t, 1), flush=True)
open(out, "w").write("".join(json.dumps(r) + "\n" for r in rows))
