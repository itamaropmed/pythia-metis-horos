"""Three-gap certificates + per-sample risk labels on generated (exact truth) and real data."""
import sys, os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.sparse import hstack, csr_matrix
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../afi_20261017")))
from peras.core.global_bound import (three_gap_certificate, local_irreducibility, certified_easy_threshold, loss_value)
from peras.core.multiscale_pair_extrapolation import nested_matchings, certificate_v2
from peras.core.replicate_pair_certificates import f_K
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from generator import make

PHI90 = 0.17550


def geometry(notes):
    Xs = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(notes)
    E = TruncatedSVD(64, random_state=0).fit_transform(Xs); E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    return Xs, E


def gamma_lower_brier(E, y, K, alpha):
    n = (len(y) // 16) * 16
    G = certificate_v2(nested_matchings(E[:n], 4, perm_seed=1), y[:n], K, alpha, (.5,), "brier", conc="kl")["bound"]
    return G * n / len(y)        # rows beyond n contribute >= 0 to the floor


def classification(name, rep, n=8000):
    d = make(name, n, 777 + 100 * rep + len(name)); y, P, K, cell = d["y"], d["P"], d["K"], d["cell"]
    Xs, E = geometry(d["notes"]); r = np.random.default_rng(rep); perm = r.permutation(n)
    tr, ce = perm[: int(.4 * n)], perm[int(.4 * n):]
    oh = np.eye(cell.max() + 1)[cell]; F = hstack([csr_matrix(oh), Xs]).tocsr()
    m = LogisticRegression(C=2, max_iter=3000).fit(F[tr], y[tr]); Q = np.zeros((len(ce), K)); Q[:, m.classes_] = m.predict_proba(F[ce])
    yc, Pc, cc, Ec = y[ce], P[ce], cell[ce], E[ce]
    GL = gamma_lower_brier(Ec, yc, K, .05 / 4)            # Brier (1-|p|^2) floor lower under A(1/2)
    out = dict(dataset=name, rep=rep, K=K)
    pbar = pd.DataFrame(Pc).groupby(cc).transform("mean").to_numpy()
    truth = {"01": ((1 - Pc.max(1)).mean(), (1 - pbar.max(1)).mean(), (1 - Pc[np.arange(len(ce)), Q.argmax(1)]).mean()),
             "brier": ((1 - (Pc ** 2).sum(1)).mean() / 2, (1 - (pbar ** 2).sum(1)).mean() / 2,
                       ((Pc[:, :, None] * ((Q[:, None, :] - np.eye(K)[None]) ** 2).sum(-1)[:, :, :].transpose(0, 2, 1).diagonal(axis1=1, axis2=2) if False else
                        np.array([[((Q[i] - np.eye(K)[k]) ** 2).sum() / 2 for k in range(K)] for i in range(len(ce))]) * Pc).sum(1)).mean()),
             "log": ((-(Pc * np.log(np.clip(Pc, 1e-12, 1))).sum(1)).mean(), (-(pbar * np.log(np.clip(pbar, 1e-12, 1))).sum(1)).mean(),
                     (-(Pc * np.log(np.clip(Q, 1e-3, 1))).sum(1)).mean())}
    for L in ("01", "brier", "log"):
        pred = Q.argmax(1) if L == "01" else Q
        low = {"01": float(f_K(GL, K)), "brier": GL / 2, "log": GL}[L]
        c = three_gap_certificate(L, cc, Ec, yc, pred, K, .05, gamma_lower=low, seed=rep)
        tj, tx, tm = truth[L]
        out[L] = dict(c, true_joint=tj, true_table=tx, true_model=tm, true_data_gap=tx - tj, true_est_gap=tm - tj)
    # ---- per-sample labels: reference = training split labels; certify split split into calibration / test
    g_hat = local_irreducibility(cell[tr], E[tr], y[tr], cc, Ec, K, k=15)
    g_true = 1 - (Pc ** 2).sum(1); u = 1 - (Q ** 2).sum(1); r_true = ((Pc - Q) ** 2).sum(1)
    ok = ~np.isnan(g_hat)
    head = u - g_hat
    err = (yc != Q.argmax(1)).astype(float)
    half = np.arange(len(ce)) % 2 == 0; cal, te = half & ok, (~half) & ok
    score = np.maximum(g_hat, u)
    base = float(err[cal].mean()); eps = round(.5 * base, 3)                       # target: half the overall error
    s_star = certified_easy_threshold(score[cal], err[cal], eps=eps, delta=.05)
    easy_te = te & (score <= s_star) if s_star is not None else np.zeros(len(ce), bool)
    s_u = certified_easy_threshold(u[cal], err[cal], eps=eps, delta=.05)
    easy_u = te & (u <= s_u) if s_u is not None else np.zeros(len(ce), bool)
    q75 = np.quantile(g_true, .75)
    out["per_sample"] = dict(
        spearman_ghat_gtrue=float(spearmanr(g_hat[ok], g_true[ok]).correlation),
        spearman_model_unc_gtrue=float(spearmanr(u[ok], g_true[ok]).correlation),
        auroc_ghat_top25=float(roc_auc_score(g_true[ok] >= q75, g_hat[ok])), auroc_unc_top25=float(roc_auc_score(g_true[ok] >= q75, u[ok])),
        spearman_headroom_rtrue=float(spearmanr(head[ok], r_true[ok]).correlation),
        spearman_unc_rtrue=float(spearmanr(u[ok], r_true[ok]).correlation),
        eps=eps, easy_threshold=s_star, easy_share_test_model_unc_only=float(easy_u.sum() / te.sum()),
        easy_test_error_model_unc_only=float(err[easy_u].mean()) if easy_u.any() else None, easy_share_test=float(easy_te.sum() / te.sum()),
        easy_test_error=float(err[easy_te].mean()) if easy_te.any() else None,
        easy_test_true_error=float((1 - Pc[easy_te, Q.argmax(1)[easy_te]]).mean()) if easy_te.any() else None,
        overall_test_error=float(err[te].mean()))
    return out


def regression(rep, n=8000):
    d = make("D02_novel_findings", n, 999 + rep); post, cell = d["post"], d["cell"]; tab = d["table"]
    r = np.random.default_rng(rep)
    mu = .5 * tab[:, 0] - .4 * tab[:, 1] + .3 * tab[:, 2] + 1.5 * post[:, 0] - 1.0 * post[:, 1] + .8 * post[:, 2] * post[:, 3]
    sd = .4 + .6 * post[:, 3] + .2 * tab[:, 2]
    y = mu + sd * r.normal(size=n); R = float(np.quantile(y, .999) - np.quantile(y, .001)) * 1.5
    y = np.clip(y, np.quantile(y, .001) - .25 * R, np.quantile(y, .999) + .25 * R)       # declared range holds
    Xs, E = geometry(d["notes"]); perm = r.permutation(n); tr, ce = perm[: int(.4 * n)], perm[int(.4 * n):]
    oh = np.eye(cell.max() + 1)[cell]; F = np.hstack([oh, E])
    mdl = HistGradientBoostingRegressor(max_iter=300).fit(F[tr], y[tr]); mq = HistGradientBoostingRegressor(loss="quantile", quantile=.9, max_iter=300).fit(F[tr], y[tr])
    out = dict(dataset="R01_regression_text_table", rep=rep)
    sdc = sd[ce]; muc = mu[ce]
    # table-only truth: Y | cell is a mixture over rows -> use cell moments for MSE; MAE/pinball truth via simulation
    cc = cell[ce]
    for L in ("mse", "mae", "pinball"):
        pred = (mq if L == "pinball" else mdl).predict(F[ce])
        c = three_gap_certificate(L, cc, E[ce], y[ce], pred, None, .05, y_range=R, seed=rep)
        zs = r.normal(size=(200, len(ce))); Ys = muc + sdc * zs
        tj = {"mse": (sdc ** 2).mean(), "mae": (sdc * np.sqrt(2 / np.pi)).mean(), "pinball": (sdc * PHI90).mean()}[L]
        tm = loss_value(L, Ys.ravel(), np.tile(pred, 200)).mean()
        # table floor truth by simulation within cells
        tx = 0.0
        for c_ in np.unique(cc):
            I = np.flatnonzero(cc == c_); Yc = Ys[:, I].ravel()
            a = {"mse": Yc.mean(), "mae": np.median(Yc), "pinball": np.quantile(Yc, .9)}[L]
            tx += len(I) / len(ce) * loss_value(L, Yc, np.full(len(Yc), a)).mean()
        out[L] = dict(c, true_joint=tj, true_table=tx, true_model=tm, true_data_gap=tx - tj, true_est_gap=tm - tj)
    return out


if __name__ == "__main__":
    mode, reps = sys.argv[1], int(sys.argv[2])
    if mode == "real":
        pass
    elif mode == "clf":
        for nm in sys.argv[3].split(","):
            for rep in range(reps):
                print(json.dumps(classification(nm, rep), default=float), flush=True)
    else:
        for rep in range(reps):
            print(json.dumps(regression(rep), default=float), flush=True)


def real_mtsamples(rep):
    import runpy
    R = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../afi_20261017/run_afi_real.py"), run_name="lib")
    E, Xs, cellT = R["E"], R["Xs"], R["LADDER"]["T2_type_age_sex"]; n = len(cellT)
    out = []
    for lname, y in R["LABELS"].items():
        K = int(y.max() + 1); r = np.random.default_rng(rep); perm = r.permutation(n)
        tr, ce = perm[: int(.4 * n)], perm[int(.4 * n):]
        oh = np.eye(cellT.max() + 1)[cellT]; F = hstack([csr_matrix(oh), Xs]).tocsr()
        m = LogisticRegression(C=2, max_iter=3000).fit(F[tr], y[tr]); Q = np.zeros((len(ce), K)); Q[:, m.classes_] = m.predict_proba(F[ce])
        yc, cc, Ec = y[ce], cellT[ce], E[ce]
        GL = gamma_lower_brier(Ec, yc, K, .05 / 4)
        o = dict(dataset="MTSamples_" + lname, rep=rep, K=K)
        for L in ("01", "brier", "log"):
            pred = Q.argmax(1) if L == "01" else Q
            low = {"01": float(f_K(GL, K)), "brier": GL / 2, "log": GL}[L]
            o[L] = three_gap_certificate(L, cc, Ec, yc, pred, K, .05, gamma_lower=low, seed=rep)
        g_hat = local_irreducibility(cellT[tr], E[tr], y[tr], cc, Ec, K, k=15); ok = ~np.isnan(g_hat)
        u = 1 - (Q ** 2).sum(1); err = (yc != Q.argmax(1)).astype(float)
        half = np.arange(len(ce)) % 2 == 0; cal, te = half & ok, (~half) & ok
        eps = round(.5 * float(err[cal].mean()), 3); res = {}
        for nm_, sc in (("combined", np.maximum(g_hat, u)), ("model_unc", u), ("irreducibility", g_hat)):
            s = certified_easy_threshold(sc[cal], err[cal], eps=eps, delta=.05); sel = te & (sc <= s) if s is not None else np.zeros(len(ce), bool)
            res[nm_] = dict(share=float(sel.sum() / te.sum()), test_error=float(err[sel].mean()) if sel.any() else None)
        # categories on the test half: irreducible-hard (top-quartile g_hat), reducible (low g_hat, model wrong-ish)
        qg = np.nanquantile(g_hat[cal], .75)
        hard = te & (g_hat >= qg); red = te & (g_hat < qg) & (u >= np.quantile(u[cal], .75))
        o["per_sample"] = dict(eps=eps, overall_test_error=float(err[te].mean()), easy=res,
                               irreducible_hard=dict(share=float(hard.sum() / te.sum()), test_error=float(err[hard].mean())),
                               reducible=dict(share=float(red.sum() / te.sum()), test_error=float(err[red].mean())),
                               spearman_ghat_err=float(spearmanr(g_hat[te], err[te]).correlation),
                               spearman_unc_err=float(spearmanr(u[te], err[te]).correlation))
        out.append(o)
    return out


if __name__ == "__main__" and sys.argv[1] == "real":
    for rep in range(int(sys.argv[2])):
        for o in real_mtsamples(rep):
            print(json.dumps(o, default=float), flush=True)
