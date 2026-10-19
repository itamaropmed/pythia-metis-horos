"""Mayo Note Information Atlas -- run LOCALLY in stages (each < 3 min); aggregates only.

Target: duration residual beyond the table, r_A = y - OOF_XGB(Arm A table), patient-grouped 5-fold.
Extra:  r_B = y - OOF_XGB(Arm B = table + 11 LLM concepts)  -> what the raw notes still add (extraction completeness).
Notes:  strict pre-cutoff bundles (note_bundles_recut); the timing rule note_date < scheduled_in_room - 2 days is
        re-checked and any violating bundle is blanked (counted, never shown).
Stages: prep | fit | lens:<letters> | unique:<section index> | null:<b> | beyondB
Intermediate patient-level state lives only in --work (outside the repository); --out receives aggregates.
"""
import sys, os, json, pickle, time, argparse
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from atlas import run_atlas, TextResidualModel, info, mask_sections

ap = argparse.ArgumentParser()
ap.add_argument("stage"); ap.add_argument("--derived", required=True); ap.add_argument("--work", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--alpha", type=float, default=10.0)
ap.add_argument("--max-features", type=int, default=40000)
a = ap.parse_args(); os.makedirs(a.work, exist_ok=True); os.makedirs(a.out, exist_ok=True)
PREP = os.path.join(a.work, "atlas_prep.pkl"); MODEL = os.path.join(a.work, "atlas_model.pkl")
KW = dict(min_patients_term=50, min_patients_header=200, top_sections=12, alpha=a.alpha, max_features=a.max_features)
t0 = time.time()

def save(name, obj):
    obj = {k: v for k, v in obj.items() if k != "_model"} if isinstance(obj, dict) else obj
    json.dump(obj, open(os.path.join(a.out, f"atlas_{name}.json"), "w"), indent=1, default=float)

if a.stage == "prep":
    sys.path.insert(0, os.path.join(HERE, "../../scripts")); sys.path.insert(0, os.path.join(HERE, "../.."))
    from xgboost import XGBRegressor
    from sklearn.model_selection import GroupKFold
    from run_mayo_metis_case_risk import MODEL_KWARGS, procedure_family
    df = pd.read_parquet(f"{a.derived}/df.parquet")
    nb = pd.read_parquet(f"{a.derived}/note_bundles_recut.parquet")
    m = df[["case_durable_id", "scheduled_in_room"]].merge(nb, on="case_durable_id", how="left")
    sched = pd.to_datetime(m.scheduled_in_room, utc=True)
    def last_date(s):
        if not isinstance(s, str) or not s.strip():
            return pd.NaT
        d = pd.to_datetime(pd.Series(s.replace(";", ",").replace("|", ",").split(",")).str.strip(), errors="coerce", utc=True)
        return d.max()
    last = m.note_dates_used.map(last_date)
    viol = (last.notna()) & (last >= sched - pd.Timedelta(days=2))
    texts = m.note_text.fillna("").astype(str).to_numpy(object); texts[viol.to_numpy()] = ""
    y = df.case_actual_duration_time.to_numpy(float); pid = df.patient_durable_id.astype(str).to_numpy()
    fam = df.primary_procedure_name.map(procedure_family).to_numpy()
    kw = dict(MODEL_KWARGS); kw["n_jobs"] = 4
    res = {}
    for arm in ("A", "B"):
        X = pd.read_parquet(f"{a.derived}/X{arm}.parquet").to_numpy(np.float32); o = np.zeros(len(y))
        for k, (tr, va) in enumerate(GroupKFold(5).split(X, y, pid)):
            o[va] = XGBRegressor(random_state=k, **kw).fit(X[tr], y[tr]).predict(X[va])
        res[arm] = y - o
    pickle.dump(dict(texts=list(texts), pid=pid, rA=res["A"], rB=res["B"], fam=fam), open(PREP, "wb"))
    save("prep", dict(cases=len(y), patients=int(len(set(pid))), notes_joined=float(m.note_text.notna().mean()),
                      nonempty_after_rule=float(np.mean([len(t) > 0 for t in texts])), timing_violations_blanked=int(viol.sum()),
                      table_oof_mae_A=float(np.abs(res["A"]).mean()), table_oof_mae_B=float(np.abs(res["B"]).mean()), sec=time.time() - t0))
    print("prep ok", time.time() - t0); sys.exit()

P = pickle.load(open(PREP, "rb")); T, pid, r = P["texts"], P["pid"], P["rA"]
if a.stage == "fit":
    mdl = TextResidualModel(alpha=a.alpha, max_features=a.max_features).fit(T, r, pid)
    pickle.dump(mdl, open(MODEL, "wb")); print("fit ok", time.time() - t0); sys.exit()
mdl = pickle.load(open(MODEL, "rb"))
if a.stage.startswith("lens:"):
    L = a.stage.split(":")[1]
    res = run_atlas(T, pid, r, strata=P["fam"], lenses=L, model=mdl, n_null=0, unique_retrain=0, **KW)
    save("lens_" + L, res); print("lens", L, "ok", time.time() - t0)
elif a.stage.startswith("unique:"):
    j = int(a.stage.split(":")[1])
    from atlas import segment
    from collections import defaultdict
    hp = defaultdict(set)
    for t, p in zip(T, pid):
        for h, _ in segment(t):
            hp[h].add(p)
    secs = [h for h, ps in sorted(hp.items(), key=lambda kv: -len(kv[1])) if len(ps) >= 200 and h != "_PREAMBLE"][:12]
    h = secs[j]; base = info(r, mdl.predict(T))["I"]; mt = [mask_sections(t, drop={h}) for t in T]
    u = base - info(r, TextResidualModel(alpha=a.alpha, max_features=a.max_features).fit(mt, r, pid).predict(mt))["I"]
    save(f"unique_{j}", dict(section=h, unique_I_retrain=u)); print("unique", h, u, time.time() - t0)
elif a.stage.startswith("null:"):
    b = int(a.stage.split(":")[1]); rp = r[np.random.default_rng(100 + b).permutation(len(r))]
    I = info(rp, TextResidualModel(alpha=a.alpha, max_features=a.max_features).fit(T, rp, pid).predict(T))["I"]
    save(f"null_{b}", dict(I=I)); print("null", b, I, time.time() - t0)
elif a.stage == "meta":
    # K: note PRESENCE / VOLUME / DOC-TYPE structure vs CONTENT.  Meta features read no words.
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from scipy.sparse import hstack, csr_matrix
    df = pd.read_parquet(f"{a.derived}/df.parquet", columns=["case_durable_id"])
    nb = pd.read_parquet(f"{a.derived}/note_bundles_recut.parquet", columns=["case_durable_id", "n_pre_cutoff_notes_used", "note_doc_types_used", "note_chars"])
    m = df.merge(nb, on="case_durable_id", how="left")
    types = m.note_doc_types_used.fillna("").astype(str).map(lambda s: [x.strip() for x in s.split(";") if x.strip()])
    tp = {}
    for ts, p in zip(types, pid):
        for t in set(ts):
            tp.setdefault(t, set()).add(p)
    keep_t = sorted([t for t, ps in tp.items() if len(ps) >= 200], key=lambda t: -len(tp[t]))
    M = np.column_stack([np.array([len(t) == 0 for t in T], float), np.log1p([len(t) for t in T]),
                         np.log1p(m.n_pre_cutoff_notes_used.fillna(0).to_numpy(float))] +
                        [np.array([ts.count(t) for ts in types], float) for t in keep_t])
    M = (M - M.mean(0)) / np.maximum(M.std(0), 1e-9)
    gm = np.zeros(len(r)); folds = list(GroupKFold(5).split(M, r, pid))
    for tr, va in folds:
        gm[va] = RidgeCV(alphas=(.1, 1, 10, 100)).fit(M[tr], r[tr]).predict(M[va])
    g_text = mdl.predict(T)
    # combined: text prediction + meta (stacked linearly, cross-fitted)
    S = np.column_stack([M, g_text]); gb = np.zeros(len(r))
    for tr, va in folds:
        gb[va] = RidgeCV(alphas=(.1, 1, 10, 100)).fit(S[tr], r[tr]).predict(S[va])
    per_type = []
    rc = r - r.mean()
    for t in keep_t:
        has = np.array([t in ts for ts in types])
        per_type.append(dict(doc_type=t, patients=len(tp[t]), cases=int(has.sum()), mean_residual_minutes=float(rc[has].mean())))
    save("meta", dict(meta_only=info(r, gm), text_only=info(r, g_text), meta_plus_text=info(r, gb),
                      content_beyond_meta_I=info(r, gb)["I"] - info(r, gm)["I"], doc_types_reported=len(keep_t), per_doc_type=per_type))
    print("meta ok", time.time() - t0)
elif a.stage == "beyondB":
    rB = P["rB"]; mB = TextResidualModel(alpha=a.alpha, max_features=a.max_features).fit(T, rB, pid)
    gB = mB.predict(T); from atlas import cluster_boot
    save("beyondB", dict(**info(rB, gB), I_ci95=cluster_boot(rB ** 2 - (rB - gB) ** 2, pid), residual_var=float((rB ** 2).mean())))
    print("beyondB ok", time.time() - t0)
