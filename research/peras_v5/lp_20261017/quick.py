import sys, os, numpy as np, pandas as pd
sys.path.insert(0, __import__("os").environ.get("PERAS_REPO", __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), "../../..")))); sys.path.insert(0, __import__("os").environ.get("PERAS_REPO", __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), "../../.."))) + "/research/peras_v5/afi_20261017")
from peras.core.lp_bounds import design_statistics, variogram_lp, relaxation_path, moment_boxes, moment_lp_01
from peras.core.replicate_pair_certificates import f_K
from generator import make
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
d = make("D02_novel_findings", 8000, 5); y, P, K, cell = d["y"], d["P"], d["K"], d["cell"]
Xs = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(d["notes"])
E = TruncatedSVD(64, random_state=0).fit_transform(Xs); E /= np.linalg.norm(E, axis=1, keepdims=True)
st = design_statistics(cell, E, y, K, levels=3, alpha=.05)
k = st["keep"]; Pk = P[k]; G = (1 - (Pk ** 2).sum(1)).mean(); pb = pd.DataFrame(Pk).groupby(cell[k]).transform("mean").to_numpy()
GX = (1 - (pb ** 2).sum(1)).mean()
print("true G", round(G, 4), "GX", round(GX, 4), "V", round(GX - G, 4), "boxes", [tuple(round(v, 3) for v in b) for b in st["box"]], "gx", [round(v, 3) for v in st["gx"]], "s", [round(v, 4) for v in st["s"]])
for g in (1.0, .5, .25):
    print("gamma", g, "L", variogram_lp(st, g, 0.0), "Vmin", variogram_lp(st, g, 0.0, "V", "min"), "Vmax", variogram_lp(st, g, 0.0, "V", "max"))
print("assumption-free: L", variogram_lp(st, .5, np.inf), "U", variogram_lp(st, .5, np.inf, "G", "max"), "Vmin", variogram_lp(st, .5, np.inf, "V", "min"))
path, tip = relaxation_path(st, .5); print("tip", tip, [(round(e, 4), None if v is None else round(v, 4)) for e, v in path[::5]])
groups = [np.flatnonzero(cell == c) for c in np.unique(cell)]
pc = np.array([P[g, 1].mean() for g in groups]); w = np.array([len(g) for g in groups]) / 8000
true01 = float((w * np.minimum(pc, 1 - pc)).sum())
for J in (2, 3, 4, 6):
    bx = moment_boxes(groups, y, J); print("J", J, "01 table floor LP [", moment_lp_01(bx), ",", moment_lp_01(bx, sense="max"), "] true", round(true01, 4))
