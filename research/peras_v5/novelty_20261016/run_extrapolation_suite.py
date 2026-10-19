"""Known-signal suite for modulus-free multiscale extrapolation certificates (PERAS-X).

Synthetic designs (X ~ U[0,1]^d, n = 4000) and real-covariate plasmodes (PMLB pharynx covariates; MTSamples notes in
TF-IDF/SVD geometry).  For every run: gold floor, PERAS-X for gamma in {1, 1/2} and the gamma dial, the accepted
near-replicate pair bound with an ORACLE modulus (and misdeclared x0.5 / x2), the 1-NN Cover-Hart estimate and the
Gamma-test-style extrapolation estimate (both estimates, not bounds), and whether A(gamma) held on the realized
matchings (computed from the true laws)."""
import sys, os, json, re
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from peras.core.multiscale_pair_extrapolation import (nested_matchings, extrapolation_certificate, per_gamma_bounds,
                                                       lipschitz_pair_bound)
from peras.core.replicate_pair_certificates import f_K

LEVELS = 4


def softmax(a):
    e = np.exp(a - a.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def binary(logit):
    p = 1 / (1 + np.exp(-logit)); return np.column_stack([1 - p, p])


DESIGNS = {
    "linear2": (2, lambda X: binary(3 * (X[:, 0] + X[:, 1] - 1))),
    "smooth5": (5, lambda X: binary(2 * np.sin(2 * X[:, 0]) + 3 * X[:, 1] * X[:, 2] - 2 * X[:, 3] ** 2 - .3)),
    "sharp2": (2, lambda X: binary(15 * (X[:, 0] - .5))),
    "wiggle2": (2, lambda X: binary(2 * np.sin(12 * X[:, 0]))),
    "sparse10": (10, lambda X: binary(3 * (X[:, 0] - X[:, 1]))),
    "K4_3d": (3, lambda X: softmax(np.column_stack([2 * X[:, 0], 2 * X[:, 1], 2 * X[:, 2], 1.5 * X.sum(1) - 1.5]))),
    "mse2": (2, None),
    # adversarial stress (Jensen-tight: |p - 1/2| constant, so f_K(Gbar) = floor exactly)
    "checker_coarse": (2, lambda X: binary(np.where((np.floor(X[:, 0] / .1) + np.floor(X[:, 1] / .1)) % 2 == 0, -.847, .847))),
    "checker_fine": (2, lambda X: binary(np.where((np.floor(X[:, 0] / .008) + np.floor(X[:, 1] / .008)) % 2 == 0, -.847, .847))),
    "rowwise_random": (2, lambda X: binary(np.where(np.random.default_rng(int(X[0, 0] * 1e9)).random(len(X)) < .5, -.847, .847))),
}


def assumption_held(mats, P, info, loss="01", mu=None):
    if info is None:
        return None
    D = []
    for Pm, s in mats:
        if loss == "01":
            D.append(float((.5 * ((P[Pm[:, 0]] - P[Pm[:, 1]]) ** 2).sum(1)).mean()))
        else:
            D.append(float((.5 * (mu[Pm[:, 0]] - mu[Pm[:, 1]]) ** 2).mean()))
    a, b, t = info["a"], info["b"], info["t"]
    return bool(D[a] <= t * D[b] + 1e-12)


def evaluate(X, P, y, K, name, rep, loss="01", mu=None, sigma2=None, y_range=None, mats=None):
    mats = mats or nested_matchings(X, LEVELS)
    if loss == "01":
        gold = float((1 - P.max(1)).mean())
    else:
        gold = sigma2
    o = dict(design=name, rep=rep, K=K, gold=gold, n=len(y))
    both = extrapolation_certificate(mats, y, K, .05, (1.0, .5), loss, y_range)
    o["peras_x"] = both["bound"]; o["peras_x_choice"] = both["best"]
    o["assumption_held_at_choice"] = assumption_held(mats, P, both["best"], loss, mu)
    for g, v in per_gamma_bounds(mats, y, K, .05, (1.0, .75, .5, .25), loss, y_range).items():
        o[f"gamma_{g}"] = v
    o["Q_levels"] = [round(q, 4) for q in both["Q"]]; o["s_levels"] = [round(s, 5) for s in both["s"]]
    if loss == "01":
        P1, s1 = mats[0]
        d = np.sqrt(((X[P1[:, 0]] - X[P1[:, 1]]) ** 2).sum(1)); tv = .5 * np.abs(P[P1[:, 0]] - P[P1[:, 1]]).sum(1)
        L = float(np.max(tv / np.maximum(d, 1e-12)))
        o["L_oracle"] = L
        for sc in (.5, 1., 2.):
            o[f"lip_oracle_x{sc}"] = lipschitz_pair_bound(X, mats, y, K, sc * L)
        o["nn_estimate"] = float(f_K(min(both["Q"][0], (K - 1) / K), K))
        A = np.column_stack([np.ones(LEVELS), both["s"]]); icpt = np.linalg.lstsq(A, np.array(both["Q"]), rcond=None)[0][0]
        o["gamma_test_estimate"] = float(f_K(min(max(icpt, 0), (K - 1) / K), K))
    else:
        A = np.column_stack([np.ones(LEVELS), both["s"]])
        Qr = np.array(both["Q"]) * .5 * y_range ** 2
        o["gamma_test_estimate"] = float(np.linalg.lstsq(A, Qr, rcond=None)[0][0])
        o["nn_estimate"] = float(Qr[0])
    return o


def synthetic(name, rep, n=4000):
    r = np.random.default_rng(10_000 * (list(DESIGNS).index(name) + 1) + rep)
    d, law = DESIGNS[name]; X = r.random((n, d))
    if name == "mse2":
        mu = np.sin(3 * X[:, 0]) + X[:, 1]; y = mu + r.uniform(-.5, .5, n)
        return evaluate(X, None, y, None, name, rep, loss="mse", mu=mu, sigma2=1 / 12, y_range=3.0)
    P = law(X); K = P.shape[1]; y = np.array([r.choice(K, p=p) for p in P])
    return evaluate(X, P, y, K, name, rep)


def real_pharynx(reps, n=4000):
    import runpy
    L = runpy.run_path(os.path.join(os.path.dirname(__file__), "../realdata_20261015/tabular_replicate_ladder.py"), run_name="lib")
    D, Z, P_all = L["D"], L["Z"], L["LAWS"][2]
    out = []
    for rep in range(reps):
        r = np.random.default_rng(31_000 + rep); idx = r.choice(len(D), n, replace=False)
        X = Z[idx].copy(); X[:, -1] *= 1.0
        P = P_all[idx]; y = np.array([r.choice(2, p=p) for p in P])
        out.append(evaluate(X, P, y, 2, "pharynx_real_covariates", rep)); print(json.dumps(out[-1]), flush=True)
    return out


def real_notes(reps):
    import hashlib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    df = pd.read_csv(__import__("os").environ.get("PERAS_DATA", "/home/claude/data") + "/mtsamples.csv").dropna(subset=["transcription"])
    key = df.transcription.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    T = df.transcription[~key.map(lambda s: hashlib.md5(s.encode()).hexdigest()).duplicated()].reset_index(drop=True)
    T = T.iloc[: (len(T) // 16) * 16]; low = T.str.lower()
    Xs = TfidfVectorizer(min_df=3, sublinear_tf=True).fit_transform(T)
    X = TruncatedSVD(100, random_state=0).fit_transform(Xs); X /= np.linalg.norm(X, axis=1, keepdims=True)
    z = np.column_stack([low.str.contains("diabet"), low.str.contains("hypertens"), low.str.contains("smok"),
                         low.str.count(r"\d+\s?mg").clip(upper=5) / 5, (np.log(low.str.len()) - 7.7),
                         low.str.contains("pain"), low.str.contains("surgery|operative")]).astype(float)
    lin = z @ np.array([1.6, 1.2, 1.4, 1.5, 1.1, -.9, -1.0]) - .4
    mats = nested_matchings(X, LEVELS)
    out = []
    for variant, rough in (("notes_signal_from_text", 0.0), ("notes_signal_plus_note_level_noise", 1.0)):
        noise = np.random.default_rng(99).normal(0, 1, len(T)) * rough
        P = binary(lin + noise)
        for rep in range(reps):
            r = np.random.default_rng(41_000 + rep); y = np.array([r.choice(2, p=p) for p in P])
            out.append(evaluate(X, P, y, 2, variant, rep, mats=mats)); print(json.dumps(out[-1]), flush=True)
    return out


if __name__ == "__main__":
    mode, reps = sys.argv[1], int(sys.argv[2]); here = os.path.dirname(os.path.abspath(__file__))
    if mode == "synthetic":
        names = sys.argv[3].split(",") if len(sys.argv) > 3 else list(DESIGNS)
        res = []
        for nm in names:
            for rep in range(reps):
                res.append(synthetic(nm, rep)); print(json.dumps(res[-1]), flush=True)
        json.dump(res, open(os.path.join(here, f"synthetic_{'_'.join(names)}.json"), "w"), indent=1)
    elif mode == "pharynx":
        json.dump(real_pharynx(reps), open(os.path.join(here, "pharynx.json"), "w"), indent=1)
    else:
        json.dump(real_notes(reps), open(os.path.join(here, "notes.json"), "w"), indent=1)
