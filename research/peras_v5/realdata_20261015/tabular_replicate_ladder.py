"""Known-signal / known-noise plasmode on real clinical tabular covariates (PMLB 1196_BNG_pharynx, 1M rows used as
the population).  A planted label law p(x) (binary or K=3) depends on ALL covariates (incl. continuous Age), so for a
coarse representation S the within-cell rows differ but are iid draws from the pool given the cell -> exact
replicates with cell law pbar_c (computed exactly on the 1M pool).  Gold floor = mean over certification rows of
1 - max_k pbar_{c(i)k}.  Ladder of representations from coarse (big cells) to fine (fragmented cells).
Also: certificates for REAL labels (Status, Grade) on the same ladder (no gold; compare routes)."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from peras.core.replicate_pair_certificates import (cells_from_keys, cell_counts, pair_certificate,
    mincount_certificate, fit_g_minorant, choose_rule, linear_certificate, chunk_cells, within_cell_independence_audit)

CHUNKS = (None, 4, 8, 16, 32)


def tuned_routes(cdv, yd, cc, yc, K, alpha, seed):
    """Choose the chunk size (and the per-size rule) for the pair and linear routes on development data only."""
    best = {}
    for M in CHUNKS:
        dch = chunk_cells(cdv, M, seed); cnt_d = cell_counts(dch, yd, K)
        coefs = {d: fit_g_minorant(cnt_d, K, d) for d in (1, 2)}; rule = choose_rule(cnt_d, K, coefs, alpha / 2)
        vp = pair_certificate(dch, yd, K, alpha / 2)["bound"]; vl = linear_certificate(dch, yd, K, rule, coefs, alpha / 2)["bound"]
        if vp > best.get("pair", (-1,))[0]: best["pair"] = (vp, M)
        if vl > best.get("linear", (-1,))[0]: best["linear"] = (vl, M, coefs, rule)
    Mp = best["pair"][1]; _, Ml, coefs, rule = best["linear"]
    pr = pair_certificate(chunk_cells(cc, Mp, seed + 1), yc, K, alpha / 2)["bound"]
    li = linear_certificate(chunk_cells(cc, Ml, seed + 1), yc, K, rule, coefs, alpha / 2)["bound"]
    aud = within_cell_independence_audit(cc, yc, K)
    return pr, li, Mp, Ml, rule, aud

D = pd.read_csv(__import__("os").environ.get("PERAS_DATA", "/home/claude/data") + "/1196_BNG_pharynx.tsv.gz", sep="\t")
D["Age5"] = pd.qcut(D.Age, 5, labels=False); D["Age20"] = pd.qcut(D.Age, 20, labels=False)
LADDER = {"S1": ["T", "N"], "S2": ["T", "N", "Site", "Grade"], "S3": ["T", "N", "Site", "Grade", "Condition", "Inst"],
          "S4": ["T", "N", "Site", "Grade", "Condition", "Inst", "sex", "Treatment", "Age5"],
          "S5": ["T", "N", "Site", "Grade", "Condition", "Inst", "sex", "Treatment", "Age20"]}
rng0 = np.random.default_rng(77)
cat = ["Inst", "sex", "Treatment", "Grade", "Condition", "Site", "T", "N"]
Z = np.column_stack([pd.get_dummies(D[c], prefix=c).to_numpy(float) for c in cat] + [((D.Age - D.Age.mean()) / D.Age.std()).to_numpy()[:, None]])
inter = (D["T"].to_numpy() * D["N"].to_numpy() % 3 == 0).astype(float)


def law(K):
    W = rng0.normal(0, .9, (Z.shape[1], K if K > 2 else 1))
    L = Z @ W + 1.2 * inter[:, None] * rng0.normal(0, 1, (1, W.shape[1]))
    if K == 2:
        p = 1 / (1 + np.exp(-(L[:, 0] - L[:, 0].mean()))); return np.column_stack([1 - p, p])
    e = np.exp(L - L.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


LAWS = {2: law(2), 3: law(3)}


def run(K, rep, n=20000, alpha=.05):
    r = np.random.default_rng(5000 + 10 * rep + K)
    P = LAWS[K]; idx_d = r.integers(len(D), size=n); idx_c = r.integers(len(D), size=n)
    yd = np.array([r.choice(K, p=p) for p in P[idx_d]]); yc = np.array([r.choice(K, p=p) for p in P[idx_c]])
    out = []
    for nm, cols in LADDER.items():
        key_all = D.groupby(cols, sort=False).ngroup().to_numpy() if f"_k{nm}" not in CACHE else CACHE[f"_k{nm}"]
        CACHE[f"_k{nm}"] = key_all
        pbar = np.zeros((key_all.max() + 1, K)); np.add.at(pbar, key_all, P); pbar /= np.bincount(key_all)[:, None]
        gold = float((1 - pbar[key_all[idx_c]].max(1)).mean())
        cc = key_all[idx_c]; cdv = key_all[idx_d]
        mc = mincount_certificate(cc, yc, K, alpha)["bound"]
        pr, li, Mp, Ml, rule, aud = tuned_routes(cdv, yd, cc, yc, K, alpha, rep)
        m = np.bincount(cells_from_keys(cc))
        out.append(dict(K=K, rep=rep, rep_name=nm, gold=gold, mincount=mc, pair=pr, linear=li,
                        peras=max(pr, li, mc) if not aud["reject"] else mc, new_routes=max(pr, li), audit_p=aud["p"], chunk_pair=Mp, chunk_linear=Ml,
                        median_cell_rows=float(np.median(m[cells_from_keys(cc)])), singleton_share=float((m == 1).sum() / n),
                        rule=rule))
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[-1].items()}), flush=True)
    return out


def real_labels(alpha=.05, n=20000, reps=5):
    res = []
    for target, K in (("Status", 2), ("Grade", 3)):
        y_all = cells_from_keys(D[target])
        for rep in range(reps):
            r = np.random.default_rng(900 + rep); idx_d = r.integers(len(D), size=n); idx_c = r.integers(len(D), size=n)
            for nm, cols in LADDER.items():
                cols = [c for c in cols if c != target]
                key_all = D.groupby(cols, sort=False).ngroup().to_numpy()
                cc, yc = key_all[idx_c], y_all[idx_c]
                pr, li, Mp, Ml, rule, aud = tuned_routes(key_all[idx_d], y_all[idx_d], cc, yc, K, alpha, rep)
                o = dict(target=target, rep=rep, rep_name=nm, mincount=mincount_certificate(cc, yc, K, alpha)["bound"],
                         pair=pr, linear=li, audit_p=aud["p"], audit_reject=bool(aud["reject"]))
                # large-sample reference: plug-in floor of the representation on 400k pool rows (near-exact)
                big = r.integers(len(D), size=400000)
                ct = cell_counts(key_all[big], y_all[big], K); mm = ct.sum(1)
                o["reference_floor_400k"] = float(((mm - ct.max(1))).sum() / mm.sum())
                o["peras"] = max(pr, li, o["mincount"]) if not aud["reject"] else o["mincount"]; res.append(o)
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()}), flush=True)
    return res


CACHE = {}
if __name__ == "__main__":
    mode = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    if mode == "plasmode":
        reps = int(sys.argv[2]); res = [o for K in (2, 3) for rep in range(reps) for o in run(K, rep)]
        json.dump(res, open(os.path.join(here, "tabular_ladder_plasmode.json"), "w"), indent=1)
    else:
        json.dump(real_labels(), open(os.path.join(here, "tabular_ladder_real.json"), "w"), indent=1)
