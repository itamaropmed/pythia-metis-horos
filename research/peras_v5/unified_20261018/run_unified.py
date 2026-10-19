"""PERAS-U vs the earlier PERAS components on generated data with exact laws (cohort truth computed on the
certified rows) and on real MTSamples."""
import sys, os, json, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, __import__("os").environ.get("PERAS_REPO", __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), "../../..")))); sys.path.insert(0, __import__("os").environ.get("PERAS_REPO", __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), "../../.."))) + "/research/peras_v5/afi_20261017")
sys.path.insert(0, __import__("os").environ.get("PERAS_REPO", __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), "../../.."))) + "/research/peras_v5/nli_20261017")
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from horos.models.pythia_adapters import unified_certificate, stratified_certificate
from peras.core.lp_bounds import design_statistics, variogram_lp
from peras.core.anchored_information import anchored_information


def feats(notes, cell):
    Xs = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(notes)
    E = TruncatedSVD(64, random_state=0).fit_transform(Xs); E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    oh = csr_matrix(np.eye(cell.max() + 1)[cell]); return hstack([oh, Xs]).tocsr(), E


def cohort_truth(P, cells, rows):
    """G and V on the certified rows, V with the in-sample cell means and the m/(m-1) factor."""
    G, V, N = 0.0, 0.0, len(rows)
    for rows in [rows]:
        G += (1 - (P[rows] ** 2).sum(1)).sum()
        for c in np.unique(cells[rows]):
            I = rows[cells[rows] == c]; m = len(I)
            if m >= 2:
                V += m / (m - 1) * ((P[I] - P[I].mean(0)) ** 2).sum()
    return G / N, V / N


def run_generated(src, name, rep, n=8000):
    if src == "afi":
        from generator import make; d = make(name, n, 4242 + 100 * rep)
    else:
        from generator_ood import make_ood; from run_nli import BETAS; d = make_ood(name, n, BETAS[name][2], 5151 + 100 * rep)
    y, cell, P, K = d["y"], d["cell"], d["P"], d["K"]; F, E = feats(d["notes"], cell); t0 = time.time()
    models = None
    if PORTFOLIO:
        from sklearn.ensemble import HistGradientBoostingClassifier
        tab = np.column_stack([cell // 6, (cell // 3) % 2, cell % 3])          # age band, sex, lab band
        Z = TruncatedSVD(100, random_state=0).fit_transform(TfidfVectorizer(token_pattern=r"\S+", min_df=5, sublinear_tf=True).fit_transform(d["notes"]))
        models = [(F, None), (np.hstack([tab, Z]), lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, early_stopping=True))]
    u = unified_certificate(cell, E, F, y, K, .05, .5, 3, rep, models=models)
    Gt, Vt = cohort_truth(P, cell, u["rows"])
    strata = cell // 6                                                  # age band: the coarse table variable
    s = stratified_certificate(strata, cell, E, F, y, K, .05, .5, 3, rep, models=models)
    Gs = []; 
    for sid, r in s["strata"].items():
        g, v = cohort_truth(P, cell, s["rows"][sid]); Gs.append((r["n_S"], g, v))
    NS = sum(a for a, _, _ in Gs); Gst = sum(a * g for a, g, _ in Gs) / NS; Vst = sum(a * v for a, _, v in Gs) / NS
    # earlier PERAS components on the same data (all rows): LP-1 (no NLI / model) and AFI anchored information
    st = design_statistics(cell, E, y, K, 3, .05, rep)
    lp_U = variogram_lp(st, .5, None, "G", "max"); lp_L = variogram_lp(st, .5, 0.0, "G", "min"); lp_V = variogram_lp(st, .5, None, "V", "min")
    keep = st["keep"]; Gk = float((1 - (P[keep] ** 2).sum(1)).mean())
    ai = anchored_information(cell, E, y, K, .05, seed=rep)
    G_pop = float((1 - (P ** 2).sum(1)).mean())
    return dict(src=src, dataset=name, rep=rep, K=K, n=n, sec=time.time() - t0,
                true_G=Gt, true_V=Vt, true_table=Gt + Vt, U_G=u["G"], U_V_lower=u["V_lower"], U_nli=u["nli_lower"],
                U_model=u["model_upper"], U_pair=u["pair_upper"], U_table=u["table"], U_marg=u["marginal"],
                U_table_info=u["table_information"], U_refuted=u["refuted"], U_eps_min=u["eps_min_feasible"],
                U_eps_tip=u["eps_tipping"], U_binding=u["binding_upper"], U_Gpop=u["G_population"], true_G_all=G_pop,
                S_G=s["G"], S_V=s["V_lower"], true_G_strat=Gst, true_V_strat=Vst,
                LP_G=(lp_L or 0.0, lp_U), LP_V=lp_V, LP_true_G=Gk, AFI_dI=ai["dI_lower"], AFI_GXT_U=ai["GXT_upper"])


PORTFOLIO = os.environ.get("PORTFOLIO") == "1"

if __name__ == "__main__":
    jobs = [(a.split(":")[0], a.split(":")[1]) for a in sys.argv[1].split(",")]; reps = int(sys.argv[2])
    for src, name in jobs:
        for rep in range(reps):
            print(json.dumps(run_generated(src, name, rep), default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)), flush=True)
