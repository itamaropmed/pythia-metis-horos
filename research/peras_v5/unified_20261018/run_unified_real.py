"""PERAS-U on real MTSamples (2,357 notes): outcomes specialty (K=11) and surgery-vs-not; tables note type and
note type x age x sex.  Compared with the earlier LP-1 (no NLI, no model) and AFI on the same data."""
import sys, json, runpy, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, __import__("os").environ.get("PERAS_REPO", __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), "../../.."))))
from scipy.sparse import hstack, csr_matrix
from horos.models.pythia_adapters import unified_certificate, stratified_certificate
from peras.core.lp_bounds import design_statistics, variogram_lp
from peras.core.anchored_information import anchored_information
R = runpy.run_path(__import__("os").environ.get("PERAS_REPO", __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), "../../.."))) + "/research/peras_v5/afi_20261017/run_afi_real.py", run_name="lib")
E, Xs, LAD, LAB = R["E"], R["Xs"], R["LADDER"], R["LABELS"]
names = {0: "operative", 1: "discharge", 2: "H&P/consult", 3: "imaging/report", 4: "other"}
for lname, y in LAB.items():
    K = int(y.max() + 1)
    for tname in ("T1_note_type", "T2_type_age_sex"):
        cell = np.unique(LAD[tname], return_inverse=True)[1]
        oh = csr_matrix(pd.get_dummies(pd.Series(cell).astype(str)).to_numpy(float)); F = hstack([oh, Xs]).tocsr()
        for seed in range(3):
            u = unified_certificate(cell, E, F, y, K, .05, .5, 3, seed)
            st = design_statistics(cell, E, y, K, 3, .05, seed)
            o = dict(label=lname, K=K, table=tname, seed=seed, n_S=u["n_S"], G=u["G"], V_lower=u["V_lower"], nli=u["nli_lower"],
                     model_upper=u["model_upper"], model_point=u["model_point"], pair_upper=u["pair_upper"], table_box=u["table"],
                     table_info=u["table_information"], refuted=u["refuted"], eps_tip=u["eps_tipping"], binding=u["binding_upper"],
                     G_pop=u["G_population"], losses=u["losses"],
                     LP_G=(variogram_lp(st, .5, 0.0, "G", "min") or 0.0, variogram_lp(st, .5, None, "G", "max")),
                     LP_V=variogram_lp(st, .5, None, "V", "min"), AFI_dI=anchored_information(cell, E, y, K, .05, seed=seed)["dI_lower"])
            if tname == "T1_note_type" and seed == 0:
                s = stratified_certificate(LAD["T1_note_type"], cell, E, F, y, K, .05, .5, 3, seed, min_rows=150)
                o["map"] = {names[int(k)]: dict(n_S=r["n_S"], G=[round(v, 3) for v in r["G"]], V_lower=round(r["V_lower"], 3),
                                                table=[round(v, 3) for v in r["table"]], binding=r["binding_upper"])
                            for k, r in s["strata"].items()}
            print(json.dumps(o, default=float), flush=True)
