"""REAL: MTSamples extraction-completeness ladder with dynamic geometries, including a neural sentence embedding
(e5-small-v2, run locally from the user's cached weights; unsupervised => outcome-blind).
Geometries (portfolio, alpha/5 each): TF-IDF->SVD static, e5 static, e5 diffusion-time portfolio, sequential
posterior geometry (learned on earlier batches from e5 + TF-IDF), sequential NCA on e5."""
import sys, os, json, runpy
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from pythia.core.dynamic_geometry import diffusion_coords
from horos.models.pythia_adapters import dynamic_certificate
from peras.core.agreement_field import afi_certificate
from scipy.sparse import hstack, csr_matrix

R = runpy.run_path(os.path.join(os.path.dirname(__file__), "../afi_20261017/run_afi_real.py"), run_name="lib")
LADDER, LABELS, E_tfidf, Xs = R["LADDER"], R["LABELS"], R["E"], R["Xs"]
E5 = np.load(os.path.join(os.path.dirname(__file__), "mtsamples_e5.npy")); assert len(E5) == len(E_tfidf)
D5 = diffusion_coords(E5, k=15, n_comp=20, ts=(1, 4, 16))
F = hstack([csr_matrix(E5), Xs]).tocsr()

if __name__ == "__main__":
    for lname, y in LABELS.items():
        K = int(y.max() + 1)
        for tname, cell in list(LADDER.items())[1:]:
            for seed in range(3):
                a = .05 / 5
                o = dict(label=lname, table=tname, seed=seed)
                o["tfidf_static"] = dynamic_certificate(cell, E_tfidf, y, K, a, B=1, learner="static", seed=seed)["dI_lower"]
                o["e5_static"] = dynamic_certificate(cell, E5, y, K, a, B=1, learner="static", seed=seed)["dI_lower"]
                o["e5_diffusion"] = max(dynamic_certificate(cell, D5[t], y, K, a / 3, B=1, learner="static", seed=seed)["dI_lower"] for t in D5)
                po = dynamic_certificate(cell, E5, y, K, a, B=3, learner="posterior", F=F, seed=seed, fracs=(.2, .4, .4))
                o["posterior_seq"] = po["dI_lower"]; o["posterior_batches"] = [round(v, 4) for v in po["batch_contrasts"]]
                o["nca_seq"] = dynamic_certificate(cell, E5, y, K, a, B=3, learner="nca", seed=seed, fracs=(.2, .4, .4))["dI_lower"]
                o["portfolio"] = max(o[k] for k in ("tfidf_static", "e5_static", "e5_diffusion", "posterior_seq", "nca_seq"))
                print(json.dumps(o), flush=True)
        c5 = afi_certificate(LADDER["T0_no_table"], E5, y, K, .05, levels=2, seed=0)
        ct = afi_certificate(LADDER["T0_no_table"], E_tfidf, y, K, .05, levels=2, seed=0)
        print(json.dumps(dict(label=lname, table="text_only", IT_L_e5=c5["I_T_lower"], IT_L_tfidf=ct["I_T_lower"],
                              GU_e5=c5["G_XT_upper"], GU_tfidf=ct["G_XT_upper"])), flush=True)
