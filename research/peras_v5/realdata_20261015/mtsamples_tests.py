"""MTSamples (free clinical transcriptions) tests: exact-replicate floor, leakage, embedding-free plasmode."""
import sys, os, re, json, hashlib, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from sklearn.feature_extraction.text import CountVectorizer
from peras.core.metric_dominated_saa import metric_dominated_lower_bound
from peras.core.dominated_saa import bentkus_lower_from_statistic
from peras.core.truncated_saa import dkelly_lower_mean

df = pd.read_csv(__import__("os").environ.get("PERAS_DATA", "/home/claude/data") + "/mtsamples.csv").dropna(subset=["transcription"]).reset_index(drop=True)
canon = df.transcription.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
key = canon.map(lambda s: hashlib.md5(s.encode()).hexdigest())
out = {"n_rows": len(df)}

def exact_floor(cells, labels):
    t = pd.DataFrame({"c": cells, "y": labels})
    return float(sum(len(v) - v.y.value_counts().iloc[0] for _, v in t.groupby("c")) / len(t))

# ---- A1 exact-replicate floor and A2 leakage ladder (0-1 loss, specialty)
lab = df.medical_specialty.str.strip()
kw = df.keywords.fillna("").str.lower()
desc = df.description.fillna("").str.lower()
out["floor_text"] = exact_floor(key, lab)
out["floor_text_plus_description"] = exact_floor(key + "|" + desc, lab)
out["floor_text_plus_keywords"] = exact_floor(key + "|" + kw, lab)
out["floor_text_plus_samplename"] = exact_floor(key + "|" + df.sample_name.fillna(""), lab)
out["majority_error"] = float(1 - lab.value_counts().iloc[0] / len(lab))
# leakage scanner: does the field literally contain the label?
for fld, s in (("keywords", kw), ("description", desc), ("transcription", canon)):
    out[f"label_string_in_{fld}"] = float(np.mean([l.lower().split("/")[0].split("-")[0].strip()[:6] in t for l, t in zip(lab, s)]))
print(json.dumps(out, indent=1), flush=True)

# ---- A3 embedding-free plasmode on unique texts: y = g(text) + noise, known MAE floor
U = df.loc[~key.duplicated()].reset_index(drop=True); T = U.transcription.str.lower()
cv = CountVectorizer(analyzer="char", ngram_range=(4, 4), binary=True, min_df=2)
X = cv.fit_transform(T).astype(np.float32)
inter = (X @ X.T).toarray(); sz = np.asarray(X.sum(1)).ravel()
D = 1 - inter / (sz[:, None] + sz[None, :] - inter + 1e-9); np.fill_diagonal(D, 0)
doses = T.str.count(r"\d+\s?mg").clip(upper=5).to_numpy() / 5
g = 3 * doses + 2 * T.str.contains("diabet").to_numpy() + 1.5 * T.str.contains("hypertens").to_numpy() \
    + np.log(T.str.len().to_numpy()) / 3
iu = np.triu_indices(len(g), 1)
Lglob = float(np.max(np.abs(g[iu[0]] - g[iu[1]]) / np.maximum(D[iu], 1e-9)))
print("n_unique", len(g), "declared L (max over all realized pairs)", round(Lglob, 2), flush=True)

def nn_pairs(D):
    n = len(D); E = np.argsort(D[np.triu_indices(n, 1)], kind="stable"); iu = np.triu_indices(n, 1); used = np.zeros(n, bool); P = []
    for e in E:
        i, j = iu[0][e], iu[1][e]
        if not used[i] and not used[j]:
            used[i] = used[j] = True; P.append((i, j))
            if len(P) == n // 2: break
    return np.array(P)
P = nn_pairs(D)
rows = []
for seed in range(int(sys.argv[1]) if len(sys.argv) > 1 else 10):
    r = np.random.default_rng(9000 + seed)
    for noise, sd, R1 in (("gauss", .7, .7 * np.sqrt(2 / np.pi)), ("laplace", .5, .5)):
        e = r.normal(0, sd, len(g)) if noise == "gauss" else r.laplace(0, sd, len(g))
        y = g + e
        fmax = 1 / (sd * np.sqrt(2 * np.pi)) if noise == "gauss" else 1 / (2 * sd)
        kap = np.full(len(y), 2 * fmax)
        caps = (.5, 1., 2., 4.)
        b, info = metric_dominated_lower_bound(D, y, loss="mae", L=Lglob, block_sizes=(2, 4, 8, 16, 32), caps=caps,
                                               kappa_rows=kap, seed=seed)
        bo, _ = metric_dominated_lower_bound(D, y, loss="mae", L=Lglob, block_sizes=(2, 4, 8, 16, 32), caps=caps,
                                             kappa_rows=kap, oracle_action=g, seed=seed)
        # accepted: NN pairs, first-order Lipschitz penalty, best of Bentkus/dKelly, cap grid
        pen = (Lglob * D[P[:, 0], P[:, 1]]).sum() / len(y)        # each pair: drift L*D split... first order: E|Y_i - Y_j|/2 <= (R_i+R_j)/2 + L D/2 per row
        a = .05 / (len(caps) * 2); acc = 0
        for tau in caps:
            sc = np.minimum(np.abs(y[P[:, 0]] - y[P[:, 1]]), tau) / 2
            for lcb in (bentkus_lower_from_statistic(sc.mean(), len(sc), tau / 2, a), dkelly_lower_mean(sc, cap=tau / 2, alpha=a)):
                acc = max(acc, lcb * 2 * len(P) / len(y) - pen)
        rows.append(dict(seed=seed, noise=noise, gold=R1, peras=b, peras_oracle_pen=bo, nn_pair=max(acc, 0),
                         k=info["k"] if info else None))
        print(rows[-1], flush=True)
json.dump({"summary": out, "plasmode": rows, "declared_L": Lglob}, open("mtsamples_results.json", "w"), indent=1)
