"""S3: raw-text floor from exact textual replicates (no metric, no smoothness), on the plasmode signal that
includes an out-of-dictionary channel (where every metric route returned 0)."""
import sys, os, json, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from peras.core.dominated_saa import dominated_saa_lower_bound
df = pd.read_csv(__import__("os").environ.get("PERAS_DATA", "/home/claude/data") + "/mtsamples.csv").dropna(subset=["transcription"]).reset_index(drop=True)
T = df.transcription.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
cell = T.map(lambda s: hashlib.md5(s.encode()).hexdigest()).to_numpy()
g = 3 * (T.str.count(r"\d+\s?mg").clip(upper=5) / 5) + 2 * T.str.contains("diabet") + 1.5 * T.str.contains("hypertens") \
    + np.log(T.str.len()) / 3 + 1.0 * T.str.contains("smok")
g = g.to_numpy(float)
res = []
for seed in range(int(sys.argv[1])):
    r = np.random.default_rng(9200 + seed)
    for noise, e, gold in (("gauss", r.normal(0, .7, len(g)), .7 * np.sqrt(2 / np.pi)), ("laplace", r.laplace(0, .5, len(g)), .5)):
        y = g + e
        out = dominated_saa_lower_bound(cell, y, np.arange(len(y)), loss="mae", caps=(.5, 1., 2., 4.), alpha=.05,
                                        inequalities=("bentkus",), representation_provenance="exact-canonical-text",
                                        independent_clusters_attested=True, outcome_blind_labels_attested=True)
        res.append(dict(seed=seed, noise=noise, gold=gold, bound=out.bound, ratio=out.bound / gold))
        print(res[-1], flush=True)
json.dump(res, open("mtsamples_replicates_results.json", "w"), indent=1)
