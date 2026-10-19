"""LP bounds with relaxation path + moment LP + conditional text geometry maps (generated with exact truth; MTSamples)."""
import sys, os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, __import__("os").environ.get("PERAS_REPO", __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), "../../..")))); sys.path.insert(0, __import__("os").environ.get("PERAS_REPO", __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), "../../.."))) + "/research/peras_v5/afi_20261017")
from peras.core.lp_bounds import (design_statistics, variogram_lp, relaxation_path, moment_boxes, moment_lp_01,
                                  conditional_text_map)
from peras.core.replicate_pair_certificates import f_K
from generator import make
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def geo(notes):
    Xs = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(notes)
    E = TruncatedSVD(64, random_state=0).fit_transform(Xs); return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)


def lp_block(st, K, target):
    o = {}
    for g in (1.0, .5):
        L = variogram_lp(st, g, 0.0); o[f"G_lower_g{g}"] = L
        o[f"V_lower_g{g}"] = variogram_lp(st, g, 0.0, "V", "min"); o[f"V_upper_g{g}"] = variogram_lp(st, g, 0.0, "V", "max")
        path, tip = relaxation_path(st, g, 30); o[f"tipping_eps_g{g}"] = tip
        o[f"path_g{g}"] = [(round(e, 5), None if v is None else round(v, 4)) for e, v in path]
    o["G_upper_free"] = variogram_lp(st, .5, np.inf, "G", "max"); o["V_lower_free"] = variogram_lp(st, .5, np.inf, "V", "min")
    if target == "brier":
        o["R01_lower_g0.5"] = float(f_K(o["G_lower_g0.5"] or 0, K)); o["R01_upper_free"] = min(o["G_upper_free"], (K - 1) / K)
    return o


def clf(name, rep, n=8000):
    d = make(name, n, 4040 + rep + 50 * len(name)); y, P, K, cell = d["y"], d["P"], d["K"], d["cell"]
    E = geo(d["notes"]); st = design_statistics(cell, E, y, K, 3, .05, rep)
    k = st["keep"]; Pk = P[k]; pb = pd.DataFrame(Pk).groupby(cell[k]).transform("mean").to_numpy()
    G = float((1 - (Pk ** 2).sum(1)).mean()); GX = float((1 - (pb ** 2).sum(1)).mean())
    o = dict(dataset=name, rep=rep, true_G=G, true_V=GX - G, true_R01=float((1 - Pk.max(1)).mean()), **lp_block(st, K, "brier"))
    if K == 2:
        groups = [np.flatnonzero(cell == c) for c in np.unique(cell)]; w = np.array([len(g) for g in groups]) / n
        pc = np.array([P[g, 1].mean() for g in groups]); o["true_table01"] = float((w * np.minimum(pc, 1 - pc)).sum())
        for J in (2, 3):
            bx = moment_boxes(groups, y, J, .05); o[f"table01_LP_J{J}"] = (moment_lp_01(bx), moment_lp_01(bx, sense="max"))
    mp = conditional_text_map(cell, E, y, K, .05, seed=rep)
    trueV = {c: float(((P[cell == c] - P[cell == c].mean(0)) ** 2).sum(1).mean()) for c in np.unique(cell)}
    tv = np.array([trueV[m["cell"]] for m in mp]); dp = np.array([m["dI_point"] for m in mp]); dl = np.array([m["dI_lower"] for m in mp])
    o["map_spearman_point_vs_trueV"] = float(spearmanr(dp, tv).correlation) if tv.std() > 0 else None
    o["map_cells_certified"] = int((dl > 0).sum()); o["map_cells"] = len(mp); o["map_false_cert"] = int(((dl > 0) & (dl > tv + 1e-12)).sum())
    return o


def reg(rep, n=8000):
    d = make("D02_novel_findings", n, 999 + rep); post, cell, tab = d["post"], d["cell"], d["table"]; r = np.random.default_rng(rep)
    mu = .5 * tab[:, 0] - .4 * tab[:, 1] + .3 * tab[:, 2] + 1.5 * post[:, 0] - 1.0 * post[:, 1] + .8 * post[:, 2] * post[:, 3]
    sd = .4 + .6 * post[:, 3] + .2 * tab[:, 2]; y = mu + sd * r.normal(size=n)
    R = float(np.quantile(y, .999) - np.quantile(y, .001)) * 1.5; y = np.clip(y, np.median(y) - R / 2, np.median(y) + R / 2)
    E = geo(d["notes"]); st = design_statistics(cell, E, y, None, 3, .05, rep, target="mse", y_range=R)
    k = st["keep"]; s2 = sd[k] ** 2; muk = mu[k]
    V = float(pd.Series(muk).groupby(cell[k]).transform(lambda v: ((v - v.mean()) ** 2)).mean())
    return dict(dataset="R_estimation_mse", rep=rep, true_G=float(s2.mean()), true_V=V / 1.0, **lp_block(st, None, "mse"))


if __name__ == "__main__":
    mode, reps = sys.argv[1], int(sys.argv[2])
    if mode == "clf":
        for nm in sys.argv[3].split(","):
            for rep in range(reps):
                print(json.dumps(clf(nm, rep), default=float), flush=True)
    elif mode == "reg":
        for rep in range(reps):
            print(json.dumps(reg(rep), default=float), flush=True)


def real():
    import runpy
    R = runpy.run_path(__import__("os").environ.get("PERAS_REPO", __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), "../../.."))) + "/research/peras_v5/afi_20261017/run_afi_real.py", run_name="lib")
    E, LAD, LAB = R["E"], R["LADDER"], R["LABELS"]
    names = {0: "operative", 1: "discharge", 2: "H&P/consult", 3: "imaging/report", 4: "other"}
    for lname, y in LAB.items():
        K = int(y.max() + 1)
        for tname in ("T1_note_type", "T2_type_age_sex"):
            cell = LAD[tname]; st = design_statistics(cell, E, y, K, 2, .05, 0)
            o = dict(dataset="MTSamples_" + lname, table=tname, **lp_block(st, K, "brier"))
            if tname == "T1_note_type":
                o["map"] = [dict(note_type=names[m["cell"]], m=m["m"], Q_cell=round(m["Q_cell"], 3), dI_point=round(m["dI_point"], 3),
                                 dI_lower=round(m["dI_lower"], 3), resolution=round(m["resolution"], 3))
                            for m in conditional_text_map(cell, E, y, K, .05)]
            print(json.dumps(o, default=float), flush=True)


if __name__ == "__main__" and sys.argv[1] == "real":
    real()
