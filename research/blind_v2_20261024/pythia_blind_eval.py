"""PREREGISTERED blind evaluation runner (frozen before any v2.1 data was seen; see PREREGISTRATION.md).

Input: ONE observed-only CSV(.gz) per (design, replicate).  Oracle files are never opened here.
Output (per input file): <out>/<stem>/
    certificates.json   certified lower bounds on V (text information beyond the table) per method + null false-claim rates
    predictions.csv.gz  row_id, split, cell, f (table model), per-method duration predictions and W30 risks, labeler region
    run_meta.json       column audit, chosen hyper-parameters, hashes, timings
Scoring against the oracle is done separately by the data owner with score_with_oracle.py.

Usage: python3 pythia_blind_eval.py OBSERVED.csv.gz OUT_DIR [--k-null 200] [--seed 0]
"""
from __future__ import annotations

import argparse, hashlib, json, re, sys, time
from pathlib import Path

import numpy as np, pandas as pd
from scipy import sparse
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from pythia.core.tight_bounds import scalar_cert, tune_cert, tame, centre_within          # noqa: E402

PREREG = {
    "alpha": 0.05, "readers": ["bow_unigram", "tfidf_word12", "tfidf_char35", "tfidf_word12_x_cell", "tfidf_svd_hgb", "STACK"],
    "primary_certificate": "betting", "secondary_certificates": ["mp", "clt_reference", "vimp_r2_if", "plugin_bootstrap"],
    "cap_rule": "cap = 3 x train out-of-fold residual SD, rounded up to 5 min (train only)", "miss_threshold_min": 30.0,
    "labeler_m": 250, "k_null": 200, "risk_forecasters": ["metis_bins", "direct_logit_table_text", "direct_hgb_table",
                                                          "resid_gauss_none", "resid_gauss_STACK", "resid_emp_STACK"],
    "table_model_candidates": ["ridge_cell_spline", "hgb"], "cv_folds": 5, "seed": 20261024,
}
FORBIDDEN = re.compile(r"(oracle|true_|latent|conditional_mean|noise_sd|reference_prediction|risk|seed|null)", re.I)
OUTCOME_HINT = re.compile(r"(outcome|duration)", re.I); MISS_HINT = re.compile(r"(miss|event_w30|exceed|over_30)", re.I)


# ------------------------------------------------------------------------------------------------------ columns
def resolve_columns(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    def pick(pats, req=True):
        for p in pats:
            c = [x for x in cols if re.fullmatch(p, x)]
            if c:
                return c[0]
        if req:
            raise SystemExit(f"missing required column matching {pats}")
        return None
    c = {"row": pick([r"row_id"]), "entity": pick([r"entity_id"]), "split": pick([r"split"]), "cell": pick([r"cell"]),
         "text": pick([r"note_text_signal", r"note_text", r"note"]), "null_text": pick([r"note_text_null.*"], req=False),
         "y": pick([r"outcome_duration_minutes", r"duration_minutes", r"outcome_duration", r"duration"]),
         "miss": pick([r".*miss.*", r".*w30.*"], req=False),
         "pred_time": pick([r"prediction_time.*"], req=False), "note_time": pick([r"note_time.*"], req=False)}
    used = set(v for v in c.values() if v)
    num = [x for x in cols if x not in used and pd.api.types.is_numeric_dtype(df[x]) and not FORBIDDEN.search(x)
           and not OUTCOME_HINT.search(x) and not MISS_HINT.search(x)]
    c["table"] = num
    c["excluded"] = [x for x in cols if x not in used and x not in num]
    return c


# ------------------------------------------------------------------------------------------------------ helpers
def splits(df, c):
    s = df[c["split"]].astype(str).str.lower()
    tr = s.isin(["train", "fit"]).to_numpy(); ca = s.str.startswith("cal").to_numpy(); te = (s == "test").to_numpy()
    return tr, ca, te


def folds(n_rows, ent, k, seed):
    u, inv = np.unique(ent, return_inverse=True); return (np.random.default_rng(seed).permutation(len(u)) % k)[inv]


def cell_onehot(g):
    u = np.unique(g); return (g[:, None] == u[None]).astype(float)


def table_design(df, c, g):
    X = df[c["table"]].to_numpy(float); X = np.nan_to_num(X)
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd
    knots = np.array([-1., 0., 1.])
    spl = np.column_stack([np.maximum(Z[:, j:j + 1] - knots, 0) for j in range(Z.shape[1])]) if Z.shape[1] else np.zeros((len(Z), 0))
    oh = cell_onehot(g)
    inter = np.column_stack([oh[:, k:k + 1] * Z for k in range(oh.shape[1])]) if Z.shape[1] else np.zeros((len(Z), 0))
    return X, np.column_stack([oh, Z, spl, inter])


def ridge_fit(A, y, lam):
    A1 = np.column_stack([np.ones(len(A)), A]); P = lam * np.eye(A1.shape[1]); P[0, 0] = 0
    return np.linalg.solve(A1.T @ A1 + P, A1.T @ y)


def ridge_pred(A, b):
    return np.column_stack([np.ones(len(A)), A]) @ b


# ------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("observed"); ap.add_argument("out")
    ap.add_argument("--k-null", type=int, default=PREREG["k_null"]); ap.add_argument("--seed", type=int, default=PREREG["seed"])
    a = ap.parse_args(); t0 = time.time(); seed = a.seed
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import Ridge, LogisticRegression

    path = Path(a.observed); df = pd.read_csv(path); c = resolve_columns(df)
    out = Path(a.out) / re.sub(r"\.csv(\.gz)?$", "", path.name); out.mkdir(parents=True, exist_ok=True)
    meta = {"prereg": PREREG, "input": path.name, "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "columns": c}
    tr, ca, te = splits(df, c); n = len(df)
    g = df[c["cell"]].astype(str).to_numpy(); ent = df[c["entity"]].astype(str).to_numpy()
    y = df[c["y"]].to_numpy(float); T = df[c["text"]].fillna("").astype(str).to_numpy(object)
    if c["pred_time"] and c["note_time"]:
        meta["notes_before_prediction"] = float((pd.to_datetime(df[c["note_time"]]) < pd.to_datetime(df[c["pred_time"]])).mean())
    meta["entity_overlap_train_cal_test"] = [len(set(ent[tr]) & set(ent[ca])), len(set(ent[tr]) & set(ent[te])), len(set(ent[ca]) & set(ent[te]))]
    fo = folds(n, ent, PREREG["cv_folds"], seed)

    # ---------- 1. table model (train only; candidate chosen by train entity-grouped CV MSE)
    Xraw, A = table_design(df, c, g); itr = np.flatnonzero(tr); cv = {}
    oof = {"ridge_cell_spline": np.zeros(n), "hgb": np.zeros(n)}
    for k in range(PREREG["cv_folds"]):
        a_, b_ = itr[fo[itr] != k], itr[fo[itr] == k]
        oof["ridge_cell_spline"][b_] = ridge_pred(A[b_], ridge_fit(A[a_], y[a_], 1.0))
        Xh = np.column_stack([Xraw, cell_onehot(g)])
        oof["hgb"][b_] = HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, min_samples_leaf=30, random_state=seed).fit(Xh[a_], y[a_]).predict(Xh[b_])
    for kname, o in oof.items():
        cv[kname] = float(np.mean((y[itr] - o[itr]) ** 2))
    best = min(cv, key=cv.get); meta["table_model_cv_mse"] = cv; meta["table_model"] = best
    if best == "ridge_cell_spline":
        f = ridge_pred(A, ridge_fit(A[itr], y[itr], 1.0))
    else:
        Xh = np.column_stack([Xraw, cell_onehot(g)])
        f = HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, min_samples_leaf=30, random_state=seed).fit(Xh[itr], y[itr]).predict(Xh)
    f[itr] = oof[best][itr]                                   # train residuals are out-of-fold
    r = y - f
    cap = float(np.ceil(3 * r[itr].std() / 5) * 5); meta["cap_min"] = cap

    # ---------- 2. text readers (train-fitted; label-free for CAL/TEST)
    rc = centre_within(r, g)                                   # within-cell residual (train rows used for fitting only)
    def mats():
        M = {}
        M["bow_unigram"] = CountVectorizer(binary=True, min_df=5).fit(T[itr]).transform(T).astype(float)
        M["tfidf_word12"] = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=60000).fit(T[itr]).transform(T)
        M["tfidf_char35"] = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, sublinear_tf=True, max_features=80000).fit(T[itr]).transform(T)
        oh = sparse.csr_matrix(cell_onehot(g)); W = M["tfidf_word12"]
        M["tfidf_word12_x_cell"] = sparse.hstack([W] + [W.multiply(oh[:, k]) for k in range(oh.shape[1])]).tocsr()
        svd = TruncatedSVD(100, random_state=seed).fit(W[itr]); M["tfidf_svd_hgb"] = svd.transform(W)
        return M
    M = mats(); H = {}; HO = {}; lam_used = {}
    def fit_predict(name, rows_tr, rows_ev, lam):
        X = M[name]
        if name == "tfidf_svd_hgb":
            Xc = np.column_stack([X, cell_onehot(g)])
            return HistGradientBoostingRegressor(max_iter=200, learning_rate=.05, min_samples_leaf=40, random_state=seed).fit(Xc[rows_tr], rc[rows_tr]).predict(Xc[rows_ev])
        return Ridge(alpha=lam).fit(X[rows_tr], rc[rows_tr]).predict(X[rows_ev])
    for name in [x for x in PREREG["readers"] if x != "STACK"]:
        grid = (None,) if name == "tfidf_svd_hgb" else (0.3, 1., 3., 10., 30.)
        bestl, bests, besto = None, np.inf, None
        for lam in grid:
            o = np.zeros(n)
            for k in range(PREREG["cv_folds"]):
                a_, b_ = itr[fo[itr] != k], itr[fo[itr] == k]; o[b_] = fit_predict(name, a_, b_, lam)
            s = np.mean((rc[itr] - o[itr]) ** 2)
            if s < bests:
                bestl, bests, besto = lam, s, o
        lam_used[name] = bestl; HO[name] = besto
        H[name] = fit_predict(name, itr, np.arange(n), bestl); H[name][itr] = besto[itr]
    names = [x for x in PREREG["readers"] if x != "STACK"]
    Ow = np.column_stack([HO[k][itr] for k in names]); w = np.linalg.lstsq(Ow, rc[itr], rcond=None)[0]
    H["STACK"] = np.column_stack([H[k] for k in names]) @ w; HO["STACK"] = np.column_stack([HO[k] for k in names]) @ w
    meta["reader_lambda"] = lam_used; meta["stack_weights"] = dict(zip(names, w.tolist()))
    meta["train_oof_R2_within_cell"] = {k: float(1 - np.mean((rc[itr] - HO[k][itr]) ** 2) / np.var(rc[itr])) for k in H}

    # ---------- 3. certificates on CAL (entity = unit), portfolio alpha / J
    J = len(PREREG["readers"]); al = PREREG["alpha"] / J; ica = np.flatnonzero(ca); nca = len(set(ent[ica]))
    certs = {"J": J, "alpha_each": al, "cap_min": cap, "n_cal_entities": nca, "readers": {}}
    cfgs = {}
    def certify(h_cal_raw, cfg, meth):
        hc = tame(h_cal_raw, g[ica], cfg["q"])
        return scalar_cert(hc, r[ica], cap, al, method="betting" if meth == "betting" else ("mp" if meth == "mp" else "clt"),
                           lam=cfg["lam"] if meth == "betting" else None, groups=g[ica], entity_ids=ent[ica])
    def vimp_if(h_cal):        # Williamson-style R^2 gain with influence-function CI, cap applied to r
        rr = np.clip(centre_within(r[ica], g[ica]), -cap, cap); hh = centre_within(h_cal, g[ica])
        v = rr.var(); mse = np.mean((rr - hh) ** 2); R2 = 1 - mse / v
        IF = -((rr - hh) ** 2 - mse) / v + mse * ((rr - rr.mean()) ** 2 - v) / v ** 2
        lo = R2 - norm.ppf(1 - al) * IF.std(ddof=1) / np.sqrt(len(rr))
        return {"R2": float(R2), "R2_lower": float(lo), "V_lower_equiv": float(max(lo, 0) * v), "claim": bool(lo > 0)}
    def plugin_boot(h_cal, B=300):
        rng = np.random.default_rng(seed); hc = tame(h_cal, g[ica], .02); rr = np.clip(r[ica], -cap, cap); vals = []
        for _ in range(B):
            s = rng.integers(0, len(ica), len(ica)); hb = centre_within(hc[s], g[ica][s])
            vals.append(max(np.mean(hb * rr[s]), 0) ** 2 / max(np.mean(hb ** 2), 1e-12))
        return float(np.quantile(vals, al))
    for k in PREREG["readers"]:
        cfg = tune_cert(centre_within(HO[k][itr], g[itr]), r[itr], g[itr], al, nca, caps=(cap,)); cfgs[k] = cfg
        row = {m: certify(H[k][ica], cfg, m).bound for m in ("betting", "mp", "clt")}
        row["plug_in"] = certify(H[k][ica], cfg, "betting").plug_in
        row["vimp_if"] = vimp_if(H[k][ica]); row["plugin_bootstrap_lower"] = plugin_boot(H[k][ica]); row["tuning"] = cfg
        certs["readers"][k] = row

    # ---------- 4. nulls: K within-cell permutations of the CAL scores (equivalent to permuting CAL notes for these
    #               per-note, train-fitted readers); false-claim rate per method
    rng = np.random.default_rng(seed + 7); K = a.k_null; claims = {k: {"betting": 0, "mp": 0, "clt": 0, "vimp_if": 0} for k in PREREG["readers"]}
    cells_cal = g[ica]; idx_by_cell = {u: np.flatnonzero(cells_cal == u) for u in np.unique(cells_cal)}
    for _ in range(K):
        perm = np.arange(len(ica))
        for u, I in idx_by_cell.items():
            perm[I] = I[rng.permutation(len(I))]
        for k in PREREG["readers"]:
            hp = H[k][ica][perm]
            for m in ("betting", "mp", "clt"):
                claims[k][m] += certify(hp, cfgs[k], m).bound > 0
            claims[k]["vimp_if"] += vimp_if(hp)["claim"]
    certs["null_within_cell_permutation"] = {"K": K, "false_claim_rate": {k: {m: v / K for m, v in d.items()} for k, d in claims.items()}}
    # train-null: whole pipeline on the provided null text column (if present) for the primary STACK reader only
    if c["null_text"]:
        Tn = df[c["null_text"]].fillna("").astype(str).to_numpy(object)
        Wn = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=60000).fit(Tn[itr]).transform(Tn)
        on = np.zeros(n)
        for k in range(PREREG["cv_folds"]):
            a_, b_ = itr[fo[itr] != k], itr[fo[itr] == k]; on[b_] = Ridge(alpha=lam_used["tfidf_word12"]).fit(Wn[a_], rc[a_]).predict(Wn[b_])
        hn = Ridge(alpha=lam_used["tfidf_word12"]).fit(Wn[itr], rc[itr]).predict(Wn)
        cfg = tune_cert(centre_within(on[itr], g[itr]), r[itr], g[itr], al, nca, caps=(cap,))
        certs["provided_null_text_full_pipeline_tfidf_word12"] = {m: certify(hn[ica], cfg, m).bound for m in ("betting", "mp", "clt")}

    # ---------- 5. predictions: duration and W30 risk on all rows (scored by the data owner)
    P = pd.DataFrame({"row_id": df[c["row"]], "split": df[c["split"]], "cell": g, "f_table": f})
    cm = pd.Series(r[itr]).groupby(g[itr]).mean(); base = pd.Series(g).map(cm).fillna(0).to_numpy()
    for k in PREREG["readers"]:
        P[f"dur_{k}"] = f + base + H[k]
    thr = PREREG["miss_threshold_min"]
    miss_def = "|y - f_table| > 30 (runner-defined)"
    if c["miss"]:
        mtr = df[c["miss"]].to_numpy(float)[itr]
        cand = {cc: float(np.mean(((np.abs(y[itr] - df[cc].to_numpy(float)[itr]) > thr).astype(float)) == mtr))
                for cc in c["table"]}
        cand["f_table"] = float(np.mean(((np.abs(r[itr]) > thr).astype(float)) == mtr)); bestc = max(cand, key=cand.get)
        meta["miss_reference_match_train"] = cand
        if cand[bestc] > .999:
            ref = f if bestc == "f_table" else df[bestc].to_numpy(float); miss_def = f"|y - {bestc}| > 30 (matched on train)"
        else:
            ref = None; miss_def = "provided miss column (reference not identified; residual forecasts use f_table)"
        miss = df[c["miss"]].to_numpy(float)
    else:
        ref = f; miss = (np.abs(r) > thr).astype(float)
    meta["miss_definition"] = miss_def
    ref = f if ref is None else ref; off = f - ref                 # residual forecasts are about y - ref = r + off
    P["risk_metis_bins"] = pd.Series(g).map(pd.Series(miss[itr]).groupby(g[itr]).mean()).to_numpy()
    Zt = np.column_stack([Xraw, cell_onehot(g)])
    Zs = sparse.hstack([sparse.csr_matrix((Zt - Zt[itr].mean(0)) / (Zt[itr].std(0) + 1e-9)), M["tfidf_word12"]]).tocsr()
    P["risk_direct_logit_table_text"] = LogisticRegression(C=1., max_iter=3000).fit(Zs[itr], miss[itr]).predict_proba(Zs)[:, 1]
    P["risk_direct_hgb_table"] = HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, min_samples_leaf=30, random_state=seed).fit(Zt[itr], miss[itr]).predict_proba(Zt)[:, 1]
    for k, hk, ho in (("none", np.zeros(n), np.zeros(n)), ("STACK", H["STACK"], HO["STACK"])):
        rh = base + hk + off; err = (r - base - ho)[itr]; s = np.sqrt(np.mean(err ** 2))
        P[f"risk_resid_gauss_{k}"] = norm.cdf((-thr - rh) / s) + 1 - norm.cdf((thr - rh) / s)
        if k == "STACK":
            ph = np.zeros(n)
            for u in np.unique(g):
                e = np.sort(err[g[itr] == u]); I = g == u
                ph[I] = 1 - (np.searchsorted(e, thr - rh[I], side="right") - np.searchsorted(e, -thr - rh[I], side="left")) / len(e)
            P["risk_resid_emp_STACK"] = ph

    # ---------- 6. certified labeler on CAL (forecast tertiles from train, m declared), directions exported
    from metis.reader_gain import refine_and_certify
    ph = P["risk_resid_gauss_STACK"].to_numpy(); edges = np.unique(np.quantile(ph[itr], [1 / 3, 2 / 3]))
    b_all = np.digitize(ph, edges)
    res = refine_and_certify({"cell": g[ica]}, b_all[ica], ent[ica], miss[ica], min_entities=PREREG["labeler_m"])
    regmap = {}
    for key in np.unique(res["region_of_row"]):
        node = res["registry"].nodes.get(f"model|region|{key}"); d = "none"
        if node is not None and node.eligible and node.contrast_lower is not None:
            d = "higher" if node.contrast_lower > 0 else ("lower" if node.contrast_upper < 0 else "none")
        regmap[key] = d
    direction = np.array(["none"] * n, dtype=object); P["labeler_bin"] = b_all
    for key, d in regmap.items():
        parts = key.split("|")
        if d == "none" or parts[0] != "reader":
            continue
        if parts[1] == "cell":
            sel = (b_all == int(parts[2][1:])) & (g == "|".join(parts[3:]))
        else:
            sel = b_all == int(parts[1][1:])
        direction[sel] = d
    P["labeler_direction"] = direction
    certs["labeler"] = {"certified_share_cal": res["certified_share"], "regions": {k: v for k, v in regmap.items()},
                        "cal_global_miss_rate": float(miss[ica].mean())}
    P.to_csv(out / "predictions.csv.gz", index=False)
    meta["seconds"] = time.time() - t0
    (out / "certificates.json").write_text(json.dumps(certs, indent=1, default=float))
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({"input": path.name, "table_model": best, "cap": cap,
                      "STACK": {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in certs["readers"]["STACK"].items() if kk in ("betting", "mp", "plug_in")},
                      "null_FCR_STACK": certs["null_within_cell_permutation"]["false_claim_rate"]["STACK"], "sec": round(meta["seconds"])}))


if __name__ == "__main__":
    main()
