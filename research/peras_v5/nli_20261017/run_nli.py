import sys, json, time, numpy as np
sys.path.insert(0, __import__("os").environ.get("PERAS_REPO", __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), "../../..")))); sys.path.insert(0, __import__("os").environ.get("PERAS_REPO", __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), "../../.."))) + "/research/peras_v5/nli_20261017")
from generator_ood import make_ood, FAMILIES
from methods import nli_certificate, nli_permutation_test, nli_perm_learned, model_delta_bootstrap, model_local_permutation, codec_permutation_test
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
BETAS = {"O1_grammar_severity": (0, .5, .95), "O2_long_range_negation": (0, .4, .75), "O3_numbers_in_words": (0, .45, .85),
         "O4_copy_forward_template": (0, .5, .95), "O5_two_languages": (0, .45, .85), "O6_local_rare_token": (0, 2.0, 3.0)}


def run(fam, level, rep, n=4000):
    beta = BETAS[fam][level]; d = make_ood(fam, n, beta, 900_000 + 1000 * FAMILIES.index(fam) + 100 * level + rep)
    y, cell = d["y"], d["cell"]; t0 = time.time()
    Xs = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(d["notes"])
    E = TruncatedSVD(64, random_state=0).fit_transform(Xs); E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    L, parts, D = nli_certificate(cell, E, Xs, y, .05, rep)
    o = dict(family=fam, level=level, beta=beta, rep=rep, true_nli=d["true_nli"], nli_lower=L, **{"nli_" + k: v for k, v in parts.items()})
    o["nli_perm_p"] = nli_permutation_test(cell, [E, D], y, 199, rep)
    o["nli_perm_learned_p"] = nli_perm_learned(cell, Xs, y, 199, rep)
    est, lo, rej = model_delta_bootstrap(cell, Xs, y, 1000, rep); o.update(model_delta=est, model_delta_lo=lo, model_boot_reject=bool(rej))
    o["model_perm_p"], o["model_perm_stat"] = model_local_permutation(cell, Xs, y, 199, rep)
    o["codec_p"], o["codec_stat"] = codec_permutation_test(cell, E, y, 199, rep)
    o["sec"] = time.time() - t0
    return o


if __name__ == "__main__":
    fams = sys.argv[1].split(","); reps = int(sys.argv[2])
    for f in fams:
        for level in (0, 1, 2):
            for rep in range(reps):
                print(json.dumps(run(f, level, rep), default=float), flush=True)
