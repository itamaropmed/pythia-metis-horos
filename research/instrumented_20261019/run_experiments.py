"""Instrumented-information experiments E1-E6 (simulated readers, known truth).  Writes results/*.json.

  python3 run_experiments.py [--reps 100] [--only E1,E2]
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from pythia.core.instrumented import (_groups, between_scores, loo_centre_groups, projection_bound,  # noqa
                                      true_spread)
from sim_lib import BETA, crossfit_probs, read, simulate  # noqa

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
ALPHA = .05


def two_fold(d, rng):
    return rng.permutation(len(d["y"])) % 2


def fold_bound(d, folds, Q, groups_fn, alpha, method="chernoff", scores="loo"):
    """Sum over folds of (w_f * bound_f) with alpha/2 per fold; truth computed the same way."""
    L = T = 0.0
    for f in (0, 1):
        I = np.flatnonzero(folds == f); w = len(I) / len(folds)
        g = groups_fn(I)
        H = loo_centre_groups(Q[I], g) if scores == "loo" else scores(Q[I], I)
        L += w * projection_bound(H, d["y"][I], alpha / 2, method, groups=g).bound
        T += w * true_spread(d["P"][I], g)
    return L, T


# ---------------------------------------------------------------------------------------------- E1
def e1(reps, rng):
    """Validity and tightness for readers of different quality, two concentration methods, three n."""
    readers = {"strong": dict(concepts=[0, 1, 2, 3, 4, 5], err=.05),
               "cheap": dict(concepts=[0, 1, 2], err=.15),
               "paraphrase_weak": dict(concepts=[3, 4], err=.25),
               "hallucinating": dict(concepts=[4, 5], err=.4, n_hallucinated=6),
               "double_dip_strong": dict(concepts=[0, 1, 2, 3, 4, 5], err=.05)}
    out = {}
    for n in (1000, 4000, 16000):
        for name, cfg in readers.items():
            for method in ("hoeffding", "chernoff"):
                rows = []
                for _ in range(reps if n < 16000 else max(reps // 4, 10)):
                    d = simulate(n, rng); fo = two_fold(d, rng)
                    r = read(d, rng, **cfg)
                    Q = crossfit_probs(d, r, fo, double_dip=name.startswith("double"))
                    rows.append(fold_bound(d, fo, Q, lambda I: d["cell"][I], ALPHA, method))
                a = np.array(rows)
                out[f"n{n}|{name}|{method}"] = dict(
                    reps=len(a), miss_rate=float((a[:, 0] > a[:, 1] + 1e-12).mean()),
                    mean_bound=float(a[:, 0].mean()), mean_truth=float(a[:, 1].mean()),
                    tightness=float(a[:, 0].mean() / a[:, 1].mean()))
    return out


# ---------------------------------------------------------------------------------------------- E2
def e2(reps, rng):
    """Reader-hacking: text carries NO information (beta = 0); K prompt/model configurations; report a
    'certified' positive bound if any config certifies.  Naive (alpha each) vs portfolio (alpha / K)."""
    out = {}
    for K in (1, 10, 50, 200):
        naive = port = 0
        for _ in range(reps):
            d = simulate(2000, rng, beta=np.zeros(6), gamma=0.0); fo = two_fold(d, rng)
            g = d["cell"]; best_naive = best_port = 0.0
            for k in range(K):
                cfg = dict(concepts=list(rng.choice(6, rng.integers(1, 4), replace=False)),
                           err=float(rng.uniform(0, .3)), n_hallucinated=int(rng.integers(0, 4)))
                Q = crossfit_probs(d, read(d, rng, **cfg), fo)
                # single fold-0 evaluation to keep cost low
                I = np.flatnonzero(fo == 0); H = loo_centre_groups(Q[I], g[I])
                pb = lambda a: projection_bound(H, d["y"][I], a, "chernoff", groups=g[I]).bound
                best_naive = max(best_naive, pb(ALPHA)); best_port = max(best_port, pb(ALPHA / K))
            naive += best_naive > 0; port += best_port > 0
        out[f"K{K}"] = dict(reps=reps, false_claim_naive=naive / reps, false_claim_portfolio=port / reps)
    # power cost of the portfolio when the text DOES carry information
    for K in (1, 10, 100, 1000, 10000):
        rows = []
        for _ in range(max(reps // 2, 10)):
            d = simulate(4000, rng); fo = two_fold(d, rng)
            Q = crossfit_probs(d, read(d, rng, [0, 1, 2, 3], .05), fo)
            rows.append(fold_bound(d, fo, Q, lambda I: d["cell"][I], ALPHA / K))
        a = np.array(rows)
        out[f"power_K{K}"] = dict(mean_bound=float(a[:, 0].mean()), mean_truth=float(a[:, 1].mean()),
                                  ratio=float(a[:, 0].mean() / a[:, 1].mean()))
    return out


# ---------------------------------------------------------------------------------------------- E3
def e3(reps, rng):
    """Certified additive ledger NLI >= B(a) + V(X, a): cheap reader A bins + strong reader B."""
    rows = []
    for _ in range(reps):
        d = simulate(8000, rng); fo = two_fold(d, rng)
        ra = read(d, rng, [0, 1, 2], .15); rb = read(d, rng, [0, 1, 2, 3, 4, 5], .05)
        Qa = crossfit_probs(d, ra, fo); Qb = crossfit_probs(d, rb, fo)
        bins = _groups(*ra.T)
        res = {}
        for f in (0, 1):
            I = np.flatnonzero(fo == f); w = .5
            gx = _groups(d["cell"][I]); gxa = _groups(d["cell"][I], bins[I])
            y = d["y"][I]; P = d["P"][I]
            parts = dict(
                between=(projection_bound(between_scores(Qa[I], gx, gxa), y, ALPHA / 4, groups=gx).bound,
                         true_spread(P, gx) - true_spread(P, gxa)),
                beyond=(projection_bound(loo_centre_groups(Qb[I], gxa), y, ALPHA / 4, groups=gxa).bound, true_spread(P, gxa)),
                single_b=(projection_bound(loo_centre_groups(Qb[I], gx), y, ALPHA / 4, groups=gx).bound, true_spread(P, gx)),
                single_a=(projection_bound(loo_centre_groups(Qa[I], gx), y, ALPHA / 4, groups=gx).bound, true_spread(P, gx)))
            for k, (l, t) in parts.items():
                res.setdefault(k, [0, 0]); res[k][0] += w * l; res[k][1] += w * t
        rows.append(res)
    agg = {}
    for k in rows[0]:
        a = np.array([r[k] for r in rows])
        agg[k] = dict(mean_bound=float(a[:, 0].mean()), mean_truth=float(a[:, 1].mean()),
                      miss_rate=float((a[:, 0] > a[:, 1] + 1e-12).mean()))
    tot = np.array([[r["between"][0] + r["beyond"][0], r["single_b"][1]] for r in rows])
    agg["ledger_total"] = dict(mean_bound=float(tot[:, 0].mean()), mean_truth=float(tot[:, 1].mean()),
                               miss_rate=float((tot[:, 0] > tot[:, 1] + 1e-12).mean()))
    return agg


# ---------------------------------------------------------------------------------------------- E4
def e4(reps, rng):
    """Upgrade regression test: new reader drops concept 0 (lost) vs new reader superset (null)."""
    out = {}
    for n in (2000, 8000, 32000):
        for scenario, new_concepts in (("lost_concept0", [1, 2, 3]), ("superset_null", [0, 1, 2, 3, 4, 5])):
            hits = []
            for _ in range(reps if n < 32000 else max(reps // 4, 10)):
                d = simulate(n, rng); fo = two_fold(d, rng)
                old = read(d, rng, [0, 1, 2, 3], .05); new = read(d, rng, new_concepts, .0)
                Qo = crossfit_probs(d, old, fo); bins_new = _groups(*new.T)
                I = np.flatnonzero(fo == 0)
                g = _groups(d["cell"][I], bins_new[I])
                b = projection_bound(loo_centre_groups(Qo[I], g), d["y"][I], ALPHA, groups=g).bound
                hits.append((b > 0, b, true_spread(d["P"][I], g)))
            a = np.array(hits, float)
            out[f"n{n}|{scenario}"] = dict(detect_rate=float(a[:, 0].mean()), mean_bound=float(a[:, 1].mean()),
                                           mean_truth_inside_new_bins=float(a[:, 2].mean()),
                                           miss_rate=float((a[:, 1] > a[:, 2] + 1e-12).mean()))
    return out


# ---------------------------------------------------------------------------------------------- E5
def e5(reps, rng):
    """Selective reading: cheap reader on all; strong reader only on a label-free selected share q
    (top-q by the cheap model's predictive entropy, threshold fixed on the other fold)."""
    out = {}
    for q in (0.0, .1, .25, .5, 1.0):
        rows = []
        for _ in range(reps):
            d = simulate(8000, rng); fo = two_fold(d, rng)
            ra = read(d, rng, [0, 1], .15); rb = read(d, rng, [0, 1, 2, 3, 4, 5], .05)
            Qa = crossfit_probs(d, ra, fo)
            ent = -(Qa * np.log(np.clip(Qa, 1e-12, 1))).sum(1)
            sel = np.zeros(len(ent), bool)
            for f in (0, 1):
                thr = np.quantile(ent[fo != f], 1 - q) if 0 < q < 1 else (np.inf if q == 0 else -np.inf)
                sel[fo == f] = ent[fo == f] >= thr
            if q == 1.0:
                sel[:] = True
            rpol = np.column_stack([ra, np.where(sel[:, None], rb, 0), sel.astype(int)])
            Qp = crossfit_probs(d, rpol, fo)
            rows.append(fold_bound(d, fo, Qp, lambda I: d["cell"][I], ALPHA) + (sel.mean(),))
        a = np.array(rows)
        out[f"q{q}"] = dict(read_share=float(a[:, 2].mean()), mean_bound=float(a[:, 0].mean()),
                            nli_truth=float(a[:, 1].mean()), certified_share_of_truth=float(a[:, 0].mean() / a[:, 1].mean()))
    return out


# ---------------------------------------------------------------------------------------------- E6
def e6(reps, rng):
    """Leakage: reader sees a post-outcome note (feature overwritten by Y w.p. leak).  Estimand = information
    of PRE-outcome text.  Plus the masking diagnostic: same reader on pre-cutoff text only."""
    out = {}
    for leak in (0.0, .05, .2):
        rows = []
        for _ in range(reps):
            d = simulate(4000, rng); fo = two_fold(d, rng)
            Q = crossfit_probs(d, read(d, rng, [0, 1, 2, 3], .05, leak=leak), fo)
            Qm = crossfit_probs(d, read(d, rng, [0, 1, 2, 3], .05, leak=0.0), fo)   # masked late notes
            l, t = fold_bound(d, fo, Q, lambda I: d["cell"][I], ALPHA / 2)
            lm, _ = fold_bound(d, fo, Qm, lambda I: d["cell"][I], ALPHA / 2)
            rows.append((l, t, lm))
        a = np.array(rows)
        out[f"leak{leak}"] = dict(miss_rate=float((a[:, 0] > a[:, 1] + 1e-12).mean()),
                                  mean_bound=float(a[:, 0].mean()), mean_truth=float(a[:, 1].mean()),
                                  mean_bound_masked=float(a[:, 2].mean()),
                                  unmasked_minus_masked=float((a[:, 0] - a[:, 2]).mean()))
    return out


# ---------------------------------------------------------------------------------------------- E7
def e7(reps, rng):
    """Double dipping on a NULL text (beta = 0) with a flexible over-fitted reader model: does the certificate
    break?  Cross-fitted vs fitted-on-evaluation-fold, 40 hallucinated features, weak regularisation."""
    from sklearn.linear_model import LogisticRegression
    from sim_lib import design_matrix
    out = {}
    for mode in ("crossfit", "double_dip"):
        hits = []
        for _ in range(reps):
            d = simulate(1000, rng, beta=np.zeros(6), gamma=0.0); fo = two_fold(d, rng)
            r = read(d, rng, [0, 1, 2, 3, 4, 5], .0, n_hallucinated=40)
            X = design_matrix(d["cell"], r); Q = np.zeros((len(r), 2))
            for f in (0, 1):
                te = fo == f; tr = te if mode == "double_dip" else ~te
                Q[te] = LogisticRegression(C=100., max_iter=2000).fit(X[tr], d["y"][tr]).predict_proba(X[te])
            l, t = fold_bound(d, fo, Q, lambda I: d["cell"][I], ALPHA)
            hits.append(l > 0)
        out[mode] = dict(reps=reps, false_certificate_rate=float(np.mean(hits)))
    return out


# ---------------------------------------------------------------------------------------------- E8
def e8(reps, rng):
    """Tightening levers: (a) prompt ensembles -- average the probabilities of M independent noisy reads;
    (b) more cross-fitting folds (model trained on more data; alpha split over folds)."""
    out = {}
    for M in (1, 3, 9):
        rows = []
        for _ in range(reps):
            d = simulate(4000, rng); fo = two_fold(d, rng)
            Q = np.mean([crossfit_probs(d, read(d, rng, [0, 1, 2, 3, 4, 5], .2), fo) for _ in range(M)], 0)
            rows.append(fold_bound(d, fo, Q, lambda I: d["cell"][I], ALPHA))
        a = np.array(rows); out[f"ensemble_M{M}"] = dict(tightness=float(a[:, 0].mean() / a[:, 1].mean()))
    for F in (2, 5, 10):
        rows = []
        for _ in range(reps):
            d = simulate(4000, rng); fo = rng.permutation(4000) % F
            Q = crossfit_probs(d, read(d, rng, [0, 1, 2, 3, 4, 5], .05), fo)
            L = T = 0.0
            for f in range(F):
                I = np.flatnonzero(fo == f); w = len(I) / 4000
                L += w * projection_bound(loo_centre_groups(Q[I], d["cell"][I]), d["y"][I],
                                          ALPHA / F, groups=d["cell"][I]).bound
                T += w * true_spread(d["P"][I], d["cell"][I])
            rows.append((L, T))
        a = np.array(rows); out[f"folds_{F}"] = dict(tightness=float(a[:, 0].mean() / a[:, 1].mean()))
    # (c) pooled evaluation: one bound over ALL rows (scores from the fold-out models), alpha not split
    rows = []
    for _ in range(reps):
        d = simulate(4000, rng); fo = two_fold(d, rng)
        Q = crossfit_probs(d, read(d, rng, [0, 1, 2, 3, 4, 5], .05), fo)
        rows.append((projection_bound(loo_centre_groups(Q, d["cell"]), d["y"], ALPHA,
                                      groups=d["cell"]).bound,
                     true_spread(d["P"], d["cell"])))
    a = np.array(rows); out["pooled_2fold_INVALID?"] = dict(tightness=float(a[:, 0].mean() / a[:, 1].mean()),
                                                           miss_rate=float((a[:, 0] > a[:, 1]).mean()))
    return out


# ---------------------------------------------------------------------------------------------- E9
def e9(reps, rng):
    """Selective reading with a LEARNED value-of-reading rule: on the fit fold, regress |Q_b - Q_a| on the
    cheap reader's features + cell; read the expensive note only where predicted value is in the top q."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sim_lib import design_matrix
    out = {}
    for q in (.1, .25, .5):
        rows = []
        for _ in range(reps):
            d = simulate(8000, rng); fo = two_fold(d, rng)
            ra = read(d, rng, [0, 1], .15); rb = read(d, rng, [0, 1, 2, 3, 4, 5], .05)
            Qa = crossfit_probs(d, ra, fo); Qb = crossfit_probs(d, rb, fo)
            Xa = design_matrix(d["cell"], ra); sel = np.zeros(8000, bool)
            for f in (0, 1):
                tr, te = fo != f, fo == f
                vor = GradientBoostingRegressor(n_estimators=60, max_depth=3).fit(Xa[tr], np.abs(Qb[tr, 1] - Qa[tr, 1]))
                thr = np.quantile(vor.predict(Xa[tr]), 1 - q); sel[te] = vor.predict(Xa[te]) >= thr
            rpol = np.column_stack([ra, np.where(sel[:, None], rb, 0), sel.astype(int)])
            rows.append(fold_bound(d, fo, crossfit_probs(d, rpol, fo), lambda I: d["cell"][I], ALPHA) + (sel.mean(),))
        a = np.array(rows)
        out[f"q{q}"] = dict(read_share=float(a[:, 2].mean()), certified_share_of_truth=float(a[:, 0].mean() / a[:, 1].mean()))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--only", default="E1,E2,E3,E4,E5,E6,E7,E8,E9"); a = ap.parse_args()
    fns = dict(E1=e1, E2=e2, E3=e3, E4=e4, E5=e5, E6=e6, E7=e7, E8=e8, E9=e9)
    for k in a.only.split(","):
        t = time.time(); rng = np.random.default_rng(int(k[1:]) * 7919)
        res = fns[k](a.reps if k != "E2" else max(a.reps // 2, 20), rng)
        (OUT / f"{k}.json").write_text(json.dumps(res, indent=2))
        print(k, f"{time.time() - t:.0f}s", json.dumps(res, indent=1)[:3000], flush=True)


if __name__ == "__main__":
    main()
