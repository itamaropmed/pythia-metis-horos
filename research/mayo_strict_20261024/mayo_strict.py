"""Mayo rerun with the STRICT note cutoff, nested FIT / CAL / TEST, local readers only (run on the Mac; aggregates only).

Provenance rules (from the red-team audit):
  * note dates are date-only (midnight), so a note dated on the cutoff calendar day cannot be shown to precede the
    cutoff: a bundle is kept only if EVERY note date is > 2 calendar days before the scheduled day; otherwise the whole
    bundle is blanked (bundles cannot be split without per-note text);
  * TEST = the Metis forward-time held-out set (last 20% of time, patients disjoint from development);
    development patients are split by first case time into FIT (earlier) and CAL (later), patient-disjoint;
  * the table model is trained on FIT only; FIT residuals are 5-fold patient-grouped out-of-fold (no in-sample
    residuals), CAL/TEST residuals come from the FIT model; text readers are trained on FIT only;
  * certificates use patients as the independent unit (entity_ids).
Readers (all local): TF-IDF ridge, clinical concept rules (regex; incl. agent-proposed concepts), note-availability
meta, and the historical 11 LLM features (lineage unverified; reported separately, never in the main claim).
Stages (each < 3 min): prep | table | text | agent_diag ROUND | agent_score ROUND FILE | eval
Patient-level state lives in --work (outside the repository); --out receives aggregates only.
"""
import argparse, hashlib, json, os, pickle, re, sys, time
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("arg", nargs="*")
ap.add_argument("--derived", default=os.path.expanduser("~/PycharmProjects/peras/data/derived/mayo"))
ap.add_argument("--work", default=os.path.expanduser("~/mayo_work_20261024"))
ap.add_argument("--lenient", action="store_true", help="DIAGNOSTIC ONLY: keep every bundle (old cutoff) to measure what late notes add")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
a = ap.parse_args(); os.makedirs(a.work, exist_ok=True); os.makedirs(a.out, exist_ok=True)
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")); sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
W = lambda n: os.path.join(a.work, n); t0 = time.time()
CAP = 120.


def halves(ids, seed=0):
    """Patient-level 2-way partition (distinct partitions for distinct seeds)."""
    return (pd.util.hash_array(np.asarray([f"{seed}:{i}" for i in ids], dtype=object)) % 2).astype(int)


def save(name, obj):
    json.dump(obj, open(os.path.join(a.out, f"{name}.json"), "w"), indent=1, default=float)


def centre(v, g):
    v = np.asarray(v, float).copy()
    for c in np.unique(g):
        I = g == c; v[I] -= v[I].mean()
    return v


def centre_from_training(v, g, tr):
    """Group-center using training rows only; unseen groups use train-global mean."""
    values = np.asarray(v, float)
    groups = np.asarray(g)
    train = np.asarray(tr, bool)
    if len(values) != len(groups) or len(train) != len(values) or not train.any():
        raise ValueError("values, groups, and a nonempty training mask must align")
    global_mean = values[train].mean(axis=0)
    centered = values.copy()
    for group in np.unique(groups):
        member = groups == group
        in_train = member & train
        mean = values[in_train].mean(axis=0) if in_train.any() else global_mean
        centered[member] -= mean
    return centered


def strict_tabular_matrix(df, train_mask, rare_min=25):
    """Fit the rare-category rule and one-hot schema on FIT rows only."""
    from scipy.sparse import csr_matrix, hstack
    from sklearn.preprocessing import OneHotEncoder

    train = np.asarray(train_mask, bool)
    if len(train) != len(df) or not train.any():
        raise ValueError("a nonempty, row-aligned FIT mask is required")
    posthoc = {"case_actual_duration_time", "case_actual_prep_time", "case_actual_procedure_time",
               "case_actual_wrapup_time", "case_actual_patient_in_room", "case_actual_patient_out_of_room"}
    nonfeatures = {"case_durable_id", "patient_durable_id", "patient_birth_date", "scheduled_in_room",
                   "scheduled_setup_start", "scheduled_out_of_room", "scheduled_cleanup_complete",
                   "all_procs_not_performed"}
    shipped = {"lag_sp_mean", "lag_sp_std", "lag_sp_count", "lag_proc_mean", "lag_proc_std",
               "lag_proc_count", "prior_same_proc_duration", "redo_time_gap_days"}
    drop = posthoc | nonfeatures | shipped | {c for c in df.columns if c.endswith("_date")}
    X = df[[c for c in df.columns if c not in drop]].copy()
    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(np.int8)
    cats = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    nums = [c for c in X.columns if c not in cats]
    for c in cats:
        s = X[c].astype("string")
        missing = s.isna()
        counts = s[train & ~missing.to_numpy()].value_counts()
        keep = set(counts[counts >= rare_min].index)
        mapped = s.copy()
        mapped.loc[~missing & ~s.isin(keep)] = "__RARE__"
        mapped.loc[missing] = "__MISSING__"
        X[c] = mapped.astype(str)
    numeric = X[nums].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    blocks = [csr_matrix(numeric)]
    names = list(nums)
    if cats:
        encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32, sparse_output=True)
        encoder.fit(X.loc[train, cats])
        blocks.append(encoder.transform(X[cats]))
        names.extend(encoder.get_feature_names_out(cats).tolist())
    matrix = hstack(blocks, format="csr", dtype=np.float32)
    if len(names) != matrix.shape[1] or len(set(names)) != len(names):
        raise ValueError("tabular feature names are not a unique match for the encoded matrix")
    return matrix, tuple(names)


def validate_feature_alignment(df, xa, xb, min_shared=20):
    """Fail closed on row-count or shared raw-feature disagreement.

    Current XA/XB have no case key, so compare all shared numeric values to
    detect positional row-order errors before the matrices are used.
    """
    if len(df) != len(xa) or len(df) != len(xb):
        raise ValueError("df/XA/XB row counts differ; positional feature alignment is unsafe")
    direct = [c for c in df.columns if c in xa.columns and c in xb.columns
              and pd.api.types.is_numeric_dtype(df[c])
              and pd.api.types.is_numeric_dtype(xa[c])
              and pd.api.types.is_numeric_dtype(xb[c])]
    shared = [c for c in xa.columns if c in xb.columns
              and pd.api.types.is_numeric_dtype(xa[c])
              and pd.api.types.is_numeric_dtype(xb[c])]
    if len(direct) < min_shared or len(shared) < min_shared:
        raise ValueError("not enough shared numeric columns to audit positional feature alignment")
    def equal(a, b):
        av = pd.to_numeric(a, errors="coerce").to_numpy(float)
        bv = pd.to_numeric(b, errors="coerce").to_numpy(float)
        return np.isclose(av, bv, atol=1e-7, rtol=1e-7, equal_nan=True).all()
    if not all(equal(df[c], xa[c]) and equal(df[c], xb[c]) for c in direct):
        raise ValueError("shared raw numeric features disagree with df; refusing positional alignment")
    if not all(equal(xa[c], xb[c]) for c in shared):
        raise ValueError("shared XA/XB features disagree; refusing positional alignment")
    return {"direct_df_feature_columns_checked": len(direct),
            "shared_XA_XB_columns_checked": len(shared),
            "feature_matrices_have_case_key": "case_durable_id" in xa.columns and "case_durable_id" in xb.columns}


def paired_cluster_bootstrap(gain, entity_ids, reps=50000, seed=20261024, family_comparisons=5):
    """Case-weighted gain with intact-patient bootstrap and Bonferroni interval."""
    values = np.asarray(gain, float).reshape(-1)
    entities = np.asarray(entity_ids).reshape(-1).astype(str)
    if len(values) == 0 or len(values) != len(entities) or not np.isfinite(values).all():
        raise ValueError("finite gains and aligned entity IDs are required")
    if np.any(entities == "") or np.any(pd.isna(entities)):
        raise ValueError("entity IDs must be nonempty")
    unique, inverse = np.unique(entities, return_inverse=True)
    cluster_sum = np.bincount(inverse, weights=values)
    cluster_n = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    boot = np.empty(reps, float)
    for b in range(reps):
        draw = rng.integers(0, len(unique), size=len(unique))
        boot[b] = cluster_sum[draw].sum() / cluster_n[draw].sum()
    adjusted_tail = .05 / (2 * family_comparisons)
    return {"point_gain_minutes": float(values.mean()),
            "ci95_case_weighted_patient_cluster": [float(x) for x in np.quantile(boot, [.025, .975])],
            "familywise95_bonferroni_ci_across_5_arms": [
                float(x) for x in np.quantile(boot, [adjusted_tail, 1 - adjusted_tail])
            ],
            "bootstrap_replicates": int(reps),
            "family_comparisons": int(family_comparisons),
            "test_patients": int(len(unique)),
            "test_cases": int(len(values))}


# clinical concept rules: generic surgical / cardiac facts + the concepts the agent accepted on the public testbed
BASE_CONCEPTS = {
    "anticoag": r"\b(warfarin|coumadin|apixaban|eliquis|rivaroxaban|xarelto|dabigatran|pradaxa|edoxaban|heparin|enoxaparin|lovenox)\b",
    "antiplatelet": r"\b(clopidogrel|plavix|ticagrelor|brilinta|prasugrel|aspirin)\b",
    "diabetes": r"\b(diabet(es|ic)|dm2|t2dm|insulin)\b", "hypertension": r"\bhypertens(ion|ive)\b",
    "obesity": r"\b(obes(e|ity)|morbid(ly)? obes)", "malignancy": r"\b(cancer|carcinoma|malignan(t|cy)|lymphoma|metasta)",
    "ckd": r"\b(ckd|chronic kidney|renal insufficiency|esrd|dialysis)\b", "heart_failure": r"\b(heart failure|hfref|hfpef|chf|cardiomyopathy)\b",
    "low_ef": r"\b(ef|ejection fraction)\s*(of|is|was|:)?\s*(1\d|2\d|3[0-5])\s*%", "osa": r"\b(sleep apnea|osa|cpap)\b",
    "prior_ablation": r"\b(prior|previous|redo|repeat)\s+(ablation|pvi|isolation)", "persistent_af": r"\b(persistent|long-standing|longstanding|permanent)\s+(af|atrial fibrillation)",
    "lead_dwell": r"\b(lead|leads)\b.{0,40}\b(19|20)\d\d\b", "congenital": r"\b(congenital|tetralogy|fontan|transposition)\b",
    "general_anesthesia": r"\b(general anesthesia|general anesthetic|endotracheal|intubat)", "infection": r"\b(infect(ion|ed)|endocarditis|bacteremia|abscess)\b",
}


def compile_concepts(C, texts):
    low = [t.lower() for t in texts]; cols, kept = [], {}
    for k, pat in C.items():
        try:
            rx = re.compile(pat if isinstance(pat, str) else pat["pattern"], re.I)
        except re.error:
            continue
        v = np.array([1. if rx.search(t) else 0. for t in low]); cols.append(v); kept[k] = pat if isinstance(pat, str) else pat["pattern"]
    return np.column_stack(cols), kept


def ridge_dense(X, target, g, tr, lam=10.):
    Xc = np.asarray(X, float).copy()
    for c in np.unique(g):
        I = g == c; Xc[I] -= Xc[I & tr].mean(0) if (I & tr).any() else Xc[tr].mean(0)
    sd = Xc[tr].std(0) + 1e-9; Xc /= sd; tc = centre_from_training(target, g, tr)
    b = np.linalg.solve(Xc[tr].T @ Xc[tr] + lam * np.eye(Xc.shape[1]), Xc[tr].T @ tc[tr]); return Xc @ b


if a.stage == "prep":
    from run_mayo_metis_case_risk import procedure_family, split_indices
    df = pd.read_parquet(f"{a.derived}/df.parquet"); XA_ref = pd.read_parquet(f"{a.derived}/XA.parquet"); XB = pd.read_parquet(f"{a.derived}/XB.parquet")
    artifact_alignment = validate_feature_alignment(df, XA_ref, XB)
    nb = pd.read_parquet(f"{a.derived}/note_bundles_recut.parquet")
    m = df[["case_durable_id", "scheduled_in_room"]].merge(nb, on="case_durable_id", how="left", validate="one_to_one")
    sday = pd.to_datetime(m.scheduled_in_room, utc=True).dt.tz_localize(None).dt.normalize()

    def min_delta(s, d):
        if not isinstance(s, str) or not s.strip():
            return np.nan
        t = pd.to_datetime(pd.Series([x.strip() for x in s.split(";") if x.strip()]), errors="coerce", utc=True)
        return -999 if t.isna().any() else (d - t.dt.tz_localize(None).dt.normalize()).dt.days.min()
    dm = np.array([min_delta(s, d) for s, d in zip(m.note_dates_used, sday)])
    strict = (~np.isnan(dm)) & (dm > 2)
    if a.lenient:
        strict = ~np.isnan(dm)
    texts = np.where(strict, m.note_text.fillna("").astype(str).to_numpy(object), "")
    train, held, cutoff = split_indices(df)
    split = np.array(["drop"] * len(df), dtype=object); split[held] = "test"
    pid = df.patient_durable_id.astype(str).to_numpy(); tt = pd.to_datetime(df.scheduled_in_room, utc=True)
    dev = pd.DataFrame({"pid": pid[train], "t": tt.iloc[train].values}); first = dev.groupby("pid").t.min()
    c2 = first.quantile(.6); fit_p = set(first[first < c2].index)
    split[train] = np.where(pd.Series(pid[train]).isin(fit_p), "fit", "cal")
    Xtable, table_feature_names = strict_tabular_matrix(df, split == "fit", rare_min=25)
    extra_cols = [c for c in XB.columns if c.startswith("llm_")]
    if len(extra_cols) != 11:
        raise ValueError(f"expected 11 historical LLM features by name, found {len(extra_cols)}")
    pickle.dump(dict(texts=list(texts), y=df.case_actual_duration_time.to_numpy(float), XA=Xtable,
                     table_feature_names=table_feature_names,
                     LLM11=XB[extra_cols].to_numpy(np.float32), fam=df.primary_procedure_name.map(procedure_family).to_numpy(),
                     pid=pid, split=split, strict=strict), open(W("prep.pkl"), "wb"))
    s = pd.Series(split).value_counts().to_dict()
    save("prep", {"cases": len(df), "split_rows": s, "strict_notes_share_all": float(strict.mean()),
                  "strict_notes_share_by_split": {k: float(strict[split == k].mean()) for k in ("fit", "cal", "test")},
                  "bundles_blanked_by_strict_rule": int(((~np.isnan(dm)) & ~strict).sum()), "llm11_columns": len(extra_cols),
                  "tabular_feature_count_fit_only": len(table_feature_names),
                  "tabular_schema_sha256": hashlib.sha256("\n".join(table_feature_names).encode()).hexdigest(),
                  "source_feature_alignment": artifact_alignment,
                  "patient_overlap_fit_cal": len(set(pid[split == "fit"]) & set(pid[split == "cal"])),
                  "patient_overlap_dev_test": len(set(pid[(split == "fit") | (split == "cal")]) & set(pid[split == "test"])), "sec": time.time() - t0})
    print(open(os.path.join(a.out, "prep.json")).read())

elif a.stage == "table":
    from xgboost import XGBRegressor
    from sklearn.model_selection import GroupKFold
    from run_mayo_metis_case_risk import MODEL_KWARGS
    P = pickle.load(open(W("prep.pkl"), "rb")); X, y, sp, pid = P["XA"], P["y"], P["split"], P["pid"]
    kw = dict(MODEL_KWARGS); kw["n_jobs"] = 4; kw["n_estimators"] = int(a.arg[0]) if a.arg else 400
    fit = sp == "fit"; f = np.full(len(y), np.nan)
    f[~fit] = XGBRegressor(random_state=0, **kw).fit(X[fit], y[fit]).predict(X[~fit])
    idx = np.flatnonzero(fit)
    for k, (tr, va) in enumerate(GroupKFold(5).split(idx, groups=pid[idx])):
        f[idx[va]] = XGBRegressor(random_state=k, **kw).fit(X[idx[tr]], y[idx[tr]]).predict(X[idx[va]])
    np.save(W("f.npy"), f)
    save("table", {"MAE_FIT_oof": float(np.abs(y - f)[fit].mean()), "MAE_CAL": float(np.abs(y - f)[sp == "cal"].mean()),
                   "MAE_TEST": float(np.abs(y - f)[sp == "test"].mean()), "sec": time.time() - t0})
    print(open(os.path.join(a.out, "table.json")).read())

elif a.stage == "text":
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    P = pickle.load(open(W("prep.pkl"), "rb")); f = np.load(W("f.npy")); y, sp, fam, pid = P["y"], P["split"], P["fam"], P["pid"]
    T = P["texts"]; r = np.clip(y - f, -400, 400); fit = sp == "fit"; keep = sp != "drop"
    idx = np.flatnonzero(fit)
    V = TfidfVectorizer(min_df=30, max_df=.9, ngram_range=(1, 2), max_features=40000, sublinear_tf=True, dtype=np.float32)
    Xt = V.fit([T[i] for i in idx]).transform(T)                    # FIT-only vocabulary and IDF
    fo = halves(pid[idx], 0)
    best = None
    for lam in (10., 30., 100., 300., 1000., 3000.):
        o = np.zeros(len(idx))
        for q in (0, 1):
            train = np.zeros(len(y), bool); train[idx[fo != q]] = True
            valid_idx = idx[fo == q]
            Vq = TfidfVectorizer(min_df=30, max_df=.9, ngram_range=(1, 2), max_features=40000,
                                 sublinear_tf=True, dtype=np.float32)
            Xtr = Vq.fit_transform([T[i] for i in np.flatnonzero(train)])
            Xva = Vq.transform([T[i] for i in valid_idx])
            rcq = centre_from_training(r, fam, train)
            m = Ridge(alpha=lam).fit(Xtr, rcq[train]); o[fo == q] = m.predict(Xva)
        rc_fit = centre_from_training(r, fam, fit)[idx]
        sc = 1 - np.mean((rc_fit - o) ** 2) / np.var(rc_fit)
        if best is None or sc > best[0]:
            best = (sc, lam, o)
    rc_fit = centre_from_training(r, fam, fit)[idx]
    h_tfidf = Ridge(alpha=best[1]).fit(Xt[idx], rc_fit).predict(Xt); oof_tfidf = np.zeros(len(y)); oof_tfidf[idx] = best[2]
    Cx, kept = compile_concepts(BASE_CONCEPTS, T)
    meta = np.column_stack([(np.array([len(t) for t in T]) > 0).astype(float), np.log1p([len(t) for t in T])])
    np.save(W("tfidf_h.npy"), h_tfidf); np.save(W("tfidf_oof.npy"), oof_tfidf); np.save(W("concepts.npy"), Cx); np.save(W("meta.npy"), meta)
    pickle.dump({"vocab": V.get_feature_names_out().tolist()}, open(W("tfidf_vocab.pkl"), "wb"))
    save("text", {"tfidf_vocab": len(V.vocabulary_), "tfidf_lambda": best[1], "tfidf_FIT_cv_R2": float(best[0]),
                  "concept_rates_FIT": {k: float(Cx[fit, j].mean()) for j, k in enumerate(kept)}, "sec": time.time() - t0})
    print(open(os.path.join(a.out, "text.json")).read())

elif a.stage in ("agent_diag", "agent_score", "eval"):
    P = pickle.load(open(W("prep.pkl"), "rb")); f = np.load(W("f.npy")); y, sp, fam, pid = P["y"], P["split"], P["fam"], P["pid"]
    r = y - f; rcap = np.clip(r, -400, 400); fit, cal, te = sp == "fit", sp == "cal", sp == "test"; n = len(y)
    T = P["texts"]; ST = W("agent_state.json")
    st = json.load(open(ST)) if os.path.exists(ST) else {"accepted": {}, "rejected": {}, "proposed": 0}
    R = {"tfidf": None, "concepts": np.load(W("concepts.npy")), "meta": np.load(W("meta.npy")), "llm11(lineage unverified)": np.nan_to_num(P["LLM11"].astype(float))}
    for s_, spec in st["accepted"].items():
        R[f"agent:{s_}"] = compile_concepts(spec, T)[0]
    h_tfidf, oof_tfidf = np.load(W("tfidf_h.npy")), np.load(W("tfidf_oof.npy"))

    def pred(key, tr):
        return h_tfidf if key == "tfidf" else ridge_dense(R[key], rcap, fam, tr)

    def oofp(key, tr, seed=0):
        if key == "tfidf":
            return oof_tfidf
        idx = np.flatnonzero(tr); fo = halves(pid[idx], seed); o = np.zeros(n)
        for q in (0, 1):
            m = np.zeros(n, bool); m[idx[fo != q]] = True; o[idx[fo == q]] = ridge_dense(R[key], rcap, fam, m)[idx[fo == q]]
        return o

    def stack(keys, tr=fit):
        if not keys:
            return np.zeros(n), np.zeros(n)
        H = np.column_stack([pred(k, tr) for k in keys]); O = np.column_stack([oofp(k, tr) for k in keys])
        centered_target = centre_from_training(rcap, fam, tr)
        w = np.linalg.lstsq(O[tr], centered_target[tr], rcond=None)[0] if len(keys) > 1 else np.array([1.])
        return H @ w, O @ w

    def fit_cv(keys, seeds=(0, 1, 2)):
        out = []

        def fit_predict(key, train_mask, valid_idx):
            if key == "tfidf":
                train_idx = np.flatnonzero(train_mask)
                Vq = TfidfVectorizer(min_df=30, max_df=.9, ngram_range=(1, 2), max_features=40000,
                                     sublinear_tf=True, dtype=np.float32)
                Xtr = Vq.fit_transform([T[i] for i in train_idx])
                Xva = Vq.transform([T[i] for i in valid_idx])
                centered = centre_from_training(rcap, fam, train_mask)
                model = Ridge(alpha=best[1]).fit(Xtr, centered[train_mask])
                return model.predict(Xva)
            return ridge_dense(R[key], rcap, fam, train_mask)[valid_idx]

        for s in seeds:
            idx = np.flatnonzero(fit); fo = halves(pid[idx], s); o = np.zeros(n)
            score_target = np.zeros(n)
            for q in (0, 1):
                m = np.zeros(n, bool); m[idx[fo != q]] = True
                valid = idx[fo == q]
                # Stack weights use inner-OOF predictions made without the
                # outer validation fold's rows, labels, or note vocabulary.
                inner_idx = np.flatnonzero(m); inner_fold = halves(pid[inner_idx], s + 100 + q)
                H_train = np.zeros((n, len(keys))); target_oof = np.zeros(n)
                for iq in (0, 1):
                    inner_train = np.zeros(n, bool); inner_train[inner_idx[inner_fold != iq]] = True
                    inner_valid = inner_idx[inner_fold == iq]
                    target_inner = centre_from_training(rcap, fam, inner_train)
                    target_oof[inner_valid] = target_inner[inner_valid]
                    for j, key in enumerate(keys):
                        H_train[inner_valid, j] = fit_predict(key, inner_train, inner_valid)
                w = np.linalg.lstsq(H_train[m], target_oof[m], rcond=None)[0] if len(keys) > 1 else np.array([1.])
                H_valid = np.column_stack([fit_predict(key, m, valid) for key in keys])
                o[valid] = H_valid @ w
                score_target[valid] = centre_from_training(rcap, fam, m)[valid]
            out.append(1 - np.mean((score_target[idx] - o[idx]) ** 2) / np.var(score_target[idx]))
        return np.array(out)
    MAIN = ["tfidf", "concepts", "meta"] + [k for k in R if k.startswith("agent:")]

    if a.stage == "agent_diag":
        import csv
        csv.field_size_limit(10 ** 8)
        pub = os.path.expanduser("~/mnt/PycharmProjects/_claude_transfer/llm_run/mtsamples.csv")
        pubv = set(w for row in csv.DictReader(open(pub, newline="")) for w in re.findall(r"[a-z]{3,}", (row["transcription"] or "").lower()))
        _, o = stack(MAIN); res = (centre_from_training(rcap, fam, fit) - o)[fit]
        from sklearn.feature_extraction.text import CountVectorizer
        idx = np.flatnonzero(fit); cv = CountVectorizer(min_df=50, ngram_range=(1, 2), binary=True, stop_words="english", max_features=30000)
        X = cv.fit_transform([T[i] for i in idx]).astype(float); rr = res - res.mean()
        mm = np.asarray(X.mean(0)).ravel(); cov = np.asarray(X.T @ rr).ravel() / len(rr); corr = cov / (np.sqrt(mm * (1 - mm)) * rr.std() + 1e-12)
        names = cv.get_feature_names_out()
        # PRIVACY FILTER: only terms whose every word occurs in the PUBLIC MTSamples corpus, >= 50 FIT notes (aggregates)
        ok = [j for j in np.argsort(-np.abs(corr)) if all(w in pubv for w in names[j].split()) and not re.search(r"\d", names[j])][:40]
        famtab = pd.DataFrame({"fam": fam[fit], "r": r[fit]}).groupby("fam").r.agg(["count", "std"]).round(0)
        ctx = {"setting": "cardiac electrophysiology lab cases (ablations, device implants/extractions/revisions, EP studies)",
               "table_groups": {k: {"n_fit": int(v["count"]), "residual_sd_min": float(v["std"])} for k, v in famtab.iterrows()},
               "table_model": "gradient-boosted model on scheduling/patient/procedure/comorbidity/history table features",
               "current_text_model_fit_cv_R2": round(float(fit_cv(MAIN).mean()), 4),
               "existing_concepts": list(BASE_CONCEPTS) + [k for s_ in st["accepted"].values() for k in s_],
               "rejected_concepts": [k for s_ in st["rejected"].values() for k in s_],
               "residual_terms(term, corr; positive = longer than predicted)": [(names[j], round(float(corr[j]), 3)) for j in ok]}
        prompt = ("You improve a model of electrophysiology-lab case duration. A table model already uses procedure, "
                  "patient, comorbidity and history fields; a text model reads pre-procedure clinical notes (written more "
                  "than 2 days before). CONTEXT (aggregate statistics only) lists note terms most correlated with the "
                  "CURRENT residual (positive = case ran longer than predicted). Propose at most 8 NEW clinical CONCEPTS "
                  "that explain this residual beyond the table: each a Python regex applied case-insensitively to the note "
                  "('binary' = fact present). Use clinically meaningful facts (arrhythmia substrate, prior procedures, "
                  "device/lead history, anatomy, anticoagulation, imaging findings, frailty), robust to abbreviations. "
                  "Do not repeat existing or rejected concepts.\nCONTEXT:\n" + json.dumps(ctx) +
                  '\nReturn ONLY JSON: {"concepts": {"<snake_name>": {"pattern": "<regex>", "kind": "binary", "expect": "+|-", "why": "<short>"}}}')
        rnd = a.arg[0]; open(os.path.join(a.out, f"agent_round{rnd}_prompt.txt"), "w").write(prompt)
        print(json.dumps(ctx)[:3000])

    elif a.stage == "agent_score":
        rnd, pf = a.arg[0], a.arg[1]; C = json.load(open(pf)).get("concepts", {}); st["proposed"] += 1
        X, kept = compile_concepts({k: v["pattern"] for k, v in list(C.items())[:8]}, T)
        rates = X[fit].mean(0); good = [j for j in range(X.shape[1]) if .005 <= rates[j] <= .95]
        X = X[:, good]; kept = {k: kept[k] for j, k in enumerate(kept) if j in good}
        R[f"agent:r{rnd}"] = X; b = fit_cv(MAIN); w = fit_cv(MAIN + [f"agent:r{rnd}"]); g = w - b
        acc = bool(g.mean() > .001 and (g > 0).all())
        (st["accepted"] if acc else st["rejected"])[f"r{rnd}"] = kept; json.dump(st, open(ST, "w"))
        rec = {"round": rnd, "kept": len(kept), "fit_cv_gain": g.round(4).tolist(), "accept": acc, "concepts": list(kept),
               "fit_rates": {k: float(rates[good[j]]) for j, k in enumerate(kept)}}
        save(f"agent_round{rnd}_score", rec); print(json.dumps(rec))

    else:   # eval
        sys.path.insert(0, REPO)
        from pythia.core.tight_bounds import scalar_cert, tune_cert, tame, centre_within
        from scipy.stats import norm, spearmanr
        from sklearn.metrics import roc_auc_score
        from metis.reader_gain import refine_and_certify
        variants = {"table_only": [], "tfidf": ["tfidf"], "concepts": ["concepts"], "MAIN(text, local)": MAIN,
                    "llm11(lineage unverified)": ["llm11(lineage unverified)"], "MAIN+llm11": MAIN + ["llm11(lineage unverified)"]}
        J = len(variants) - 1 + st["proposed"]; alpha = .05 / J; out = {"J": J, "alpha_each": alpha, "cap_min": CAP}
        miss = (np.abs(r) > 30).astype(int); bins = pd.Series(fam).map(pd.Series(miss[fit]).groupby(fam[fit]).mean()).to_numpy()
        cm = pd.Series(fam).map(pd.Series(r[fit]).groupby(fam[fit]).mean()).fillna(0).to_numpy()
        Vres_cal = float(np.mean(centre_within(np.clip(r[cal], -CAP, CAP), fam[cal]) ** 2))
        def rm(ph):
            q = ph[te]; dec = pd.qcut(q, 10, labels=False, duplicates="drop"); obs = pd.Series(miss[te]).groupby(dec).mean(); pr = pd.Series(q).groupby(dec).mean()
            brier = float(((q - miss[te]) ** 2).mean()); prevalence = float(miss[te].mean())
            return {"brier": brier, "observed_miss_prevalence": prevalence,
                    "brier_prevalence_baseline": prevalence * (1 - prevalence),
                    "brier_skill_vs_prevalence": float(1 - brier / (prevalence * (1 - prevalence) + 1e-12)),
                    "case_spearman_risk_vs_miss": float(spearmanr(q, miss[te])[0]),
                    "case_pearson_risk_vs_miss": float(np.corrcoef(q, miss[te])[0, 1]),
                    "auc": float(roc_auc_score(miss[te], q)),
                    "decile_spearman(pred vs observed)": float(spearmanr(pr, obs)[0]), "calib_slope(obs~pred, deciles)": float(np.polyfit(pr, obs, 1)[0]),
                    "observed_miss_rate_top_vs_bottom_decile": [float(obs.iloc[-1]), float(obs.iloc[0])]}
        out["metis_bins_risk"] = rm(bins)
        for k, keys in variants.items():
            h, o = stack(keys); row = {"TEST_MAE_min": float(np.abs(y - f - h)[te].mean()),
                                       "TEST_R2_within_group": float(1 - np.mean((centre(rcap, fam)[te] - h[te]) ** 2) / np.var(centre(rcap, fam)[te]))}
            row["paired_MAE_gain_vs_table_min"] = paired_cluster_bootstrap(
                np.abs(r[te]) - np.abs(r[te] - h[te]), pid[te]
            )
            if keys:
                cfg = tune_cert(centre_within(o[fit], fam[fit]), r[fit], fam[fit], alpha, int(len(set(pid[cal]))), caps=(CAP,))
                hc = tame(h[cal], fam[cal], cfg["q"])
                for meth in ("mp", "betting"):
                    ce = scalar_cert(hc, r[cal], CAP, alpha, method=meth, lam=cfg["lam"] if meth == "betting" else None, groups=fam[cal], entity_ids=pid[cal])
                    row[f"cert_{meth}_min2"] = ce.bound; row["plug_in_min2"] = ce.plug_in
                row["cert_betting_share_of_residual_variance"] = row["cert_betting_min2"] / Vres_cal
            s = np.sqrt(np.mean((r[fit] - cm[fit] - o[fit]) ** 2)); rh = cm + h
            ph_g = norm.cdf((-30 - rh) / s) + 1 - norm.cdf((30 - rh) / s); row["risk_gauss"] = rm(ph_g)
            # empirical conditional residual forecast: FIT out-of-fold errors of the same procedure family
            ph = np.zeros(n); err = r - cm - o
            for c in np.unique(fam):
                e = np.sort(err[fit & (fam == c)]); I = fam == c
                ph[I] = 1 - (np.searchsorted(e, 30 - rh[I], side="right") - np.searchsorted(e, -30 - rh[I], side="left")) / len(e)
            row["risk"] = rm(ph)
            edges = np.unique(np.quantile(ph[fit], [1 / 3, 2 / 3])); bc, bt = np.digitize(ph[cal], edges), np.digitize(ph[te], edges)
            res = refine_and_certify({"cell": fam[cal]}, bc, pid[cal], miss[cal].astype(float), min_entities=250)
            ok = 0; tot = 0; gl = miss[te].mean()
            for key in np.unique(res["region_of_row"]):
                node = res["registry"].nodes.get(f"model|region|{key}"); parts = key.split("|")
                if node is None or not node.eligible or node.contrast_lower is None or parts[0] != "reader":
                    continue
                rb = int(parts[2][1:]) if parts[1] == "cell" else int(parts[1][1:]); sel = bt == rb
                if parts[1] == "cell":
                    sel &= fam[te] == "|".join(parts[3:])
                if node.contrast_lower > 0 or node.contrast_upper < 0:
                    tot += 1; ok += int((miss[te][sel].mean() > gl) == (node.contrast_lower > 0)) if sel.any() else 0
            row["labeler"] = {"certified_share_CAL": float(res["certified_share"]), "certified_regions": tot, "regions_direction_holds_on_TEST": ok}
            out[k] = row
            print(k, json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in row.items()}))
        out["residual_variance_CAL_capped_min2"] = Vres_cal; save("eval", out)
