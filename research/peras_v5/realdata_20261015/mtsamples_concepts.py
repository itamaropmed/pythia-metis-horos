"""Embedding-free text certificates via a registered clinical concept dictionary + residual Jaccard scale.
Drift metric D(x,x') = sum_i L_i |f_i(x) - f_i(x')| + L_res * Jaccard_4gram(x,x'); declared moduli.
S1: signal uses only dictionary concepts (L_res = 0).  S2: signal also uses an out-of-dictionary channel
('smok' mentions); its modulus must be declared on the Jaccard scale (computed over realized pairs)."""
import sys, os, json, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from sklearn.feature_extraction.text import CountVectorizer
from peras.core.metric_dominated_saa import metric_dominated_lower_bound
from peras.core.dominated_saa import bentkus_lower_from_statistic
from peras.core.truncated_saa import dkelly_lower_mean

df = pd.read_csv(__import__("os").environ.get("PERAS_DATA", "/home/claude/data") + "/mtsamples.csv").dropna(subset=["transcription"])
canon = df.transcription.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
U = canon[~canon.map(lambda s: hashlib.md5(s.encode()).hexdigest()).duplicated()].reset_index(drop=True)
TERMS = ["diabet", "hypertens", "cancer|carcinoma|malignan", "fractur", "infect", "pain", "fever", "nausea",
         "chest", "abdom", "knee", "hip", "spine|lumbar|cervical", "cardiac|heart", "renal|kidney", "lung|pulmonary",
         "stroke", "seizure", "depress", "asthma", "copd", "anemia", "obes", "allerg", "pregnan", "surgery|operative",
         "biopsy", "mri", "ct scan|cat scan", "x-ray|xray|radiograph"]
F = np.column_stack([U.str.contains(t).to_numpy().astype(float) for t in TERMS] +
                    [U.str.count(r"\d+\s?mg").clip(upper=5).to_numpy() / 5, np.log(U.str.len().to_numpy()) / 3])
names = TERMS + ["dose_count", "log_len"]
cv = CountVectorizer(analyzer="char", ngram_range=(4, 4), binary=True, min_df=2)
X = cv.fit_transform(U).astype(np.float32); inter = (X @ X.T).toarray(); sz = np.asarray(X.sum(1)).ravel()
J = 1 - inter / (sz[:, None] + sz[None, :] - inter + 1e-9); np.fill_diagonal(J, 0)
w = np.zeros(F.shape[1]); w[names.index("dose_count")] = 3; w[TERMS.index("diabet")] = 2; w[TERMS.index("hypertens")] = 1.5
w[names.index("log_len")] = 1.0
g1 = F @ w                                                   # S1: dictionary-sufficient signal
smok = U.str.contains("smok").to_numpy().astype(float)
g2 = g1 + 1.0 * smok                                         # S2: plus an out-of-dictionary channel
iu = np.triu_indices(len(U), 1)

def drift_metric(Li, Lres):
    Dc = np.zeros((len(U), len(U)))
    for i in np.flatnonzero(Li > 0):
        Dc += Li[i] * np.abs(F[:, i][:, None] - F[:, i][None, :])
    return Dc + Lres * J

def nn_pairs(D):
    n = len(D); iu = np.triu_indices(n, 1); E = np.argsort(D[iu], kind="stable"); used = np.zeros(n, bool); P = []
    for e in E:
        i, j = iu[0][e], iu[1][e]
        if not used[i] and not used[j]:
            used[i] = used[j] = True; P.append((i, j))
            if len(P) == n // 2: break
    return np.array(P)

res = []
for scen, g in (("S1_dictionary_sufficient", g1), ("S2_out_of_dictionary", g2)):
    Li = np.abs(w)                                         # declared per-concept moduli (exact for g1)
    resid = g - F @ w
    Lres = float(np.max(np.abs(resid[iu[0]] - resid[iu[1]]) / np.maximum(J[iu], 1e-9)))  # declared residual modulus
    D = drift_metric(Li, Lres)
    P = nn_pairs(D)
    for seed in range(int(sys.argv[1])):
        r = np.random.default_rng(9100 + seed)
        y = g + r.normal(0, .7, len(g)); gold = .7 * np.sqrt(2 / np.pi); kap = np.full(len(y), 2 / (.7 * np.sqrt(2 * np.pi)))
        caps = (.5, 1., 2., 4.)
        b, info = metric_dominated_lower_bound(D, y, loss="mae", L=1.0, block_sizes=(2, 4, 8, 16, 32), caps=caps, kappa_rows=kap, seed=seed)
        a = .05 / (len(caps) * 2); acc = 0; pen = D[P[:, 0], P[:, 1]].sum() / len(y)
        for tau in caps:
            sc = np.minimum(np.abs(y[P[:, 0]] - y[P[:, 1]]), tau) / 2
            for lcb in (bentkus_lower_from_statistic(sc.mean(), len(sc), tau / 2, a), dkelly_lower_mean(sc, cap=tau / 2, alpha=a)):
                acc = max(acc, lcb * 2 * len(P) / len(y) - pen)
        res.append(dict(scenario=scen, seed=seed, gold=gold, L_res=Lres, peras=b, nn_pair=max(acc, 0.), k=info["k"] if info else None))
        print(res[-1], flush=True)
json.dump(res, open("mtsamples_concepts_results.json", "w"), indent=1)
