"""Dynamic geometries on the ten generated table+text datasets (exact truth): static SVD, diffusion-time portfolio,
sequential predictable posterior geometry, sequential NCA geometry, and the 4-way portfolio (alpha/4 each)."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../afi_20261017")))
from pythia.core.dynamic_geometry import diffusion_coords
from horos.models.pythia_adapters import dynamic_certificate
from generator import make, MECH
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def run(name, rep, n=8000):
    d = make(name, n, 10_000 * (MECH.index(name) + 1) + rep); y, P, K, cell = d["y"], d["P"], d["K"], d["cell"]
    Xs = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(d["notes"])
    E = TruncatedSVD(64, random_state=0).fit_transform(Xs); E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    pc = pd.DataFrame(P).groupby(cell).transform("mean").to_numpy()
    true = float((1 - (pc ** 2).sum(1)).mean() - (1 - (P ** 2).sum(1)).mean())
    a = .05 / 4
    st = dynamic_certificate(cell, E, y, K, a, B=1, learner="static", seed=rep)["dI_lower"]
    D = diffusion_coords(E, seed=rep)
    di = max(dynamic_certificate(cell, D[t], y, K, a / 3, B=1, learner="static", seed=rep)["dI_lower"] for t in D)
    po = dynamic_certificate(cell, E, y, K, a, B=3, learner="posterior", F=Xs, seed=rep, fracs=(.2, .4, .4))
    nc = dynamic_certificate(cell, E, y, K, a, B=3, learner="nca", seed=rep, fracs=(.2, .4, .4))
    return dict(dataset=name, rep=rep, true_dI=true, static=st, diffusion=di, posterior=po["dI_lower"],
                posterior_batches=po["batch_contrasts"], nca=nc["dI_lower"], portfolio=max(st, di, po["dI_lower"], nc["dI_lower"]))


if __name__ == "__main__":
    reps = int(sys.argv[1]); names = sys.argv[2].split(",") if len(sys.argv) > 2 else MECH
    for nm in names:
        for rep in range(reps):
            print(json.dumps(run(nm, rep)), flush=True)
