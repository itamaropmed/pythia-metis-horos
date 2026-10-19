"""Real notes (MTSamples).  (a) REAL outcome surgery-vs-not; (b) exact NULL: real labels permuted within table cells;
(c) SEMI-SYNTHETIC: real notes, outcome law p = sigmoid(cell effect + beta * 1[note mentions a real clinical word]),
so the true NLI is known exactly while the text is real clinical language."""
import sys, json, time, runpy, numpy as np
R = runpy.run_path(__import__("os").environ.get("PERAS_REPO", __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), "../../.."))) + "/research/peras_v5/afi_20261017/run_afi_real.py", run_name="lib")
from methods import (nli_certificate, nli_permutation_test, nli_perm_learned, model_delta_bootstrap,
                     model_local_permutation, codec_permutation_test, nli_projection)
E, Xs, LAD, T = R["E"], R["Xs"], R["LADDER"], R["T"]
low = T.str.lower().to_numpy()

def all_methods(cell, y, rep, p=None):
    t0 = time.time(); L, parts, D = nli_certificate(cell, E, Xs, y, .05, rep)
    o = dict(nli_lower=L, **{"nli_" + k: v for k, v in parts.items()})
    o["nli_perm_p"] = nli_permutation_test(cell, [E, D], y, 199, rep); o["nli_perm_learned_p"] = nli_perm_learned(cell, Xs, y, 199, rep)
    est, lo, rej = model_delta_bootstrap(cell, Xs, y, 1000, rep); o.update(model_delta=est, model_delta_lo=lo, model_boot_reject=bool(rej))
    o["model_perm_p"], _ = model_local_permutation(cell, Xs, y, 199, rep); o["codec_p"], _ = codec_permutation_test(cell, E, y, 199, rep)
    o.update(nli_projection(cell, Xs, y, .05, rep, p_true=p)); o["sec"] = time.time() - t0; return o

if __name__ == "__main__" and sys.argv[1] == "proj":
    pass

if __name__ == "__main__":
    mode, reps = sys.argv[1], int(sys.argv[2]); ysurg = R["LABELS"]["surgery_vs_not"]
    for tname in ("T1_note_type", "T2_type_age_sex"):
        cell = np.unique(LAD[tname], return_inverse=True)[1]
        for rep in range(reps):
            rng = np.random.default_rng(7000 + rep)
            if mode == "real":
                y, tn, lv, p = ysurg, None, "real", None
            elif mode == "null":
                y = ysurg.copy(); lv = "null"; tn = 0.0; p = np.zeros(len(y))
                for c in np.unique(cell):
                    I = np.flatnonzero(cell == c); y[I] = ysurg[I][rng.permutation(len(I))]
            else:
                beta = float(mode.split("=")[1]); lv = mode
                kw = np.array(["hypertension" in s for s in low], float)
                a = rng.normal(0, .8, cell.max() + 1); p = 1 / (1 + np.exp(-(a[cell] - .5 + beta * (kw - kw.mean()))))
                pb = np.zeros_like(p)
                for c in np.unique(cell):
                    pb[cell == c] = p[cell == c].mean()
                tn = float((2 * (p - pb) ** 2).mean()); y = (rng.random(len(p)) < p).astype(int)
            print(json.dumps(dict(family="MTSamples_" + tname, level=lv, rep=rep, true_nli=tn, **all_methods(cell, y, rep, p)), default=float), flush=True)
            if mode == "real":
                break
