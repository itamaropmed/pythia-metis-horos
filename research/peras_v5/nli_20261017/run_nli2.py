"""Add the NLI projection certificate/test and the weighted-GCM baseline on the SAME datasets as run_nli.py."""
import sys, json, time, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from generator_ood import make_ood, FAMILIES
from run_nli import BETAS
from methods import nli_projection

def run(fam, level, rep, n=4000):
    beta = BETAS[fam][level]; d = make_ood(fam, n, beta, 900_000 + 1000 * FAMILIES.index(fam) + 100 * level + rep)
    y, cell = d["y"], d["cell"]; t0 = time.time()
    Xs = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(d["notes"])
    o = dict(family=fam, level=level, rep=rep, true_nli=d["true_nli"], **nli_projection(cell, Xs, y, .05, rep, p_true=d["P"][:, 1]))
    o["sec2"] = time.time() - t0; return o

if __name__ == "__main__":
    fams = sys.argv[1].split(","); reps = int(sys.argv[2]); levels = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0,1,2").split(",")]
    for f in fams:
        for level in levels:
            for rep in range(reps):
                print(json.dumps(run(f, level, rep), default=float), flush=True)
