"""Metis agentic loop over the symbolic sample space (embedding-free, table-conditioned; testbed v2, known truth).

Each round:
  DIAGNOSE (FIT only)   out-of-fold residual of the current model; rank the symbolic atoms (type:token from the
                        table-conditioned reader) and note n-grams by correlation with it.
  CONTEXT               table summary (cells, expected minutes, spread), current FIT explained share, the ranked atoms
                        and terms, accepted and rejected concepts.  No note text, no CAL/TEST information.
  PROPOSE (LiteLLM)     <= 8 concepts = case-insensitive regexes over the note (binary or numeric), with the
                        expected direction.  Executed locally: no per-note LLM calls (cost ~ one chat per round).
  SCORE (FIT only)      5 x 2-fold cross-fitted gain in within-cell R^2 of the duration residual (continuous target:
                        far more power than the binary miss used before), base vs base + concept reader.
  ACCEPT                mean gain > 0.002 and >= 4 / 5 splits positive; per-concept pruning by leave-one-out on FIT.
Final: every proposed concept set enters the Pythia portfolio count; certify on CAL (betting, alpha / J), audit TEST.
Usage:  agent_concepts.py diagnose ROUND   |   agent_concepts.py score ROUND PROPOSAL_JSON   |   agent_concepts.py final
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

import tb_common as tb
from horos.represent.score_learning import _centre
OUT = tb.HERE / "results/agent_concepts"; OUT.mkdir(parents=True, exist_ok=True)
STATE = OUT / "state.json"


def state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"accepted": {}, "rejected": {}, "proposed_sets": 0, "rounds": []}


def compile_concepts(C: dict, texts=None) -> tuple[np.ndarray, dict]:
    """Validate + execute concepts on the notes (label-free). Returns features and the kept spec."""
    texts = tb.T if texts is None else texts; low = [t.lower() for t in texts]; cols, kept = [], {}
    for name, c in list(C.items())[:8]:
        name = re.sub(r"[^a-z0-9_]", "_", str(name).lower())[:40]
        pat = str(c.get("pattern", ""))[:300]
        try:
            rx = re.compile(pat, re.I)
        except re.error:
            continue
        kind = c.get("kind", "binary")
        if kind == "number" and rx.groups >= 1:
            v = np.array([np.log1p(min(float(m.group(1)), 5000.)) if (m := rx.search(t)) and _isnum(m.group(1)) else 0. for t in low])
        else:
            kind = "binary"; v = np.array([1. if rx.search(t) else 0. for t in low])
        rate = float((v > 0)[tb.fit].mean())
        if not (.01 <= rate <= .95):
            continue
        cols.append(v); kept[name] = {"pattern": pat, "kind": kind, "expect": c.get("expect", "?"), "why": str(c.get("why", ""))[:200], "fit_rate": rate}
    return (np.column_stack(cols) if cols else np.zeros((tb.n, 0))), kept


def _isnum(s):
    try:
        float(s); return True
    except Exception:
        return False


def accepted_extra(st):
    ex = {}
    for setname, spec in st["accepted"].items():
        X, _ = compile_concepts(spec)
        if X.shape[1]:
            ex[f"agent:{setname}"] = X
    return ex


def base_keys(st):
    return tb.NEW + [f"agent:{s}" for s in st["accepted"]]


def fit_cv_r2(keys, extra, seeds=range(5)):
    idx = np.flatnonzero(tb.fit); res = []
    for s in seeds:
        fo = np.random.default_rng(100 + s).permutation(len(idx)) % 2; o = np.zeros(tb.n)
        for q in (0, 1):
            m = np.zeros(tb.n, bool); m[idx[fo != q]] = True
            o[idx[fo == q]] = tb.predictor(keys, m, extra=extra)[idx[fo == q]]
        res.append(tb.r2(o, tb.fit))
    return np.array(res)


def diagnose(rnd):
    st = state(); ex = accepted_extra(st); keys = base_keys(st)
    o = tb.oof(keys, tb.fit, extra=ex); res = (tb.rc - o)[tb.fit]
    # atoms
    atoms = [a for i, a in enumerate(tb.SYM["_atoms"]) if tb.fit[i]]
    from collections import Counter
    dfc = Counter(t for s in atoms for t in s); voc = [t for t, c in dfc.items() if c >= 15]
    Xa = np.array([[1. if t in s else 0. for t in voc] for s in atoms]); rr = res - res.mean()
    m = Xa.mean(0); corr = (Xa.T @ rr / len(rr)) / (np.sqrt(m * (1 - m)) * rr.std() + 1e-12)
    top_atoms = [(voc[j], round(float(corr[j]), 3), int(dfc[voc[j]])) for j in np.argsort(-np.abs(corr))[:40]]
    from metis.agent.reader_search import residual_terms
    terms = residual_terms([tb.T[i] for i in np.flatnonzero(tb.fit)], res, k=30, min_df=20)
    cells = pd.DataFrame({"cell": tb.cell, "f": tb.f}).groupby("cell").f.first().round(0).to_dict()
    sd = pd.Series(tb.r[tb.fit]).groupby(tb.cell[tb.fit]).std().round(0).to_dict()
    cv = fit_cv_r2(keys, ex, seeds=range(3)).mean()
    ctx = {"round": rnd, "table": {c: {"expected_min": cells[c], "spread_sd_min": sd.get(c)} for c in cells},
           "current_model": {"readers": keys, "fit_cv_within_cell_R2": round(float(cv), 4)},
           "residual_atoms(type:token, corr, n_notes)": top_atoms,
           "residual_note_terms(term, corr)": [(t, round(c, 3)) for t, c in terms],
           "accepted_concepts": {k: list(v) for k, v in st["accepted"].items()},
           "rejected_concepts": {k: list(v) for k, v in st["rejected"].items()}}
    prompt = (
        "You improve a surgical-duration model. A table model predicts duration from the specialty group only; a set of "
        "note readers already explains part of what the table misses. Below is the CONTEXT: the table, the current "
        "model's cross-validated explained share of the within-group residual, and the note atoms (typed facts from a "
        "table-conditioned reader) and note terms most correlated with the CURRENT residual (positive = cases run longer "
        "than currently predicted). Propose at most 8 NEW clinical CONCEPTS that would explain this residual. Each concept "
        "is a Python regular expression applied (case-insensitive) to the full note text: either 'binary' (fact present) "
        "or 'number' (exactly one capture group with a number, e.g. blood loss in ml). Concepts must be clinically "
        "meaningful facts (diagnoses, drugs, anesthesia, blood loss, extent of surgery, complications), robust to "
        "spelling variants and abbreviations, and must avoid negated mentions where feasible. Do not repeat accepted or "
        "rejected concepts.\nCONTEXT:\n" + json.dumps(ctx) +
        '\nReturn ONLY JSON: {"concepts": {"<snake_name>": {"pattern": "<regex>", "kind": "binary|number", '
        '"expect": "+|-", "why": "<short>"}}}')
    (OUT / f"round{rnd}_prompt.txt").write_text(prompt); (OUT / f"round{rnd}_context.json").write_text(json.dumps(ctx, indent=1))
    print("prompt chars", len(prompt), "FIT cv R2", round(float(cv), 4)); print(json.dumps(top_atoms[:15]))


def score(rnd, prop_f):
    st = state(); raw = json.loads(Path(prop_f).read_text()); C = (raw.get("concepts") or raw)
    X, kept = compile_concepts(C); setname = f"r{rnd}"; st["proposed_sets"] += 1
    ex = accepted_extra(st); keys = base_keys(st)
    if X.shape[1] == 0:
        st["rejected"][setname] = {}; st["rounds"].append({"round": rnd, "kept": 0}); STATE.write_text(json.dumps(st, indent=1)); print("no valid concepts"); return
    # per-concept pruning on FIT (leave-one-out over concepts, 3 seeds), then the set test on 5 seeds
    ex2 = dict(ex); ex2[f"agent:{setname}"] = X
    full = fit_cv_r2(keys + [f"agent:{setname}"], ex2, seeds=range(3)).mean(); names = list(kept); keep = list(range(len(names)))
    for j in range(len(names)):
        cols = [c for c in keep if c != j]
        if not cols:
            continue
        ex3 = dict(ex); ex3[f"agent:{setname}"] = X[:, cols]
        if fit_cv_r2(keys + [f"agent:{setname}"], ex3, seeds=range(3)).mean() >= full:
            keep = cols; full = fit_cv_r2(keys + [f"agent:{setname}"], ex3, seeds=range(3)).mean()
    kept = {names[j]: kept[names[j]] for j in keep}; X = X[:, keep]
    ex2 = dict(ex); ex2[f"agent:{setname}"] = X
    b = fit_cv_r2(keys, ex); w = fit_cv_r2(keys + [f"agent:{setname}"], ex2); g = w - b
    accept = bool(g.mean() > .002 and (g > 0).sum() >= 4)
    (st["accepted"] if accept else st["rejected"])[setname] = {k: {kk: v[kk] for kk in ("pattern", "kind", "expect")} for k, v in kept.items()}
    # diagnostic only (never used for decisions): agreement of each kept concept with the planted concepts
    planted = {k: tb.d["C"][k] for k in tb.d["C"]}
    diag = {nm: {pk: round(float(np.corrcoef(X[:, j], pv)[0, 1]), 2) for pk, pv in planted.items() if X[:, j].std() > 0}
            for j, nm in enumerate(kept)}
    rec = {"round": rnd, "kept": len(kept), "fit_cv_gain_mean": float(g.mean()), "fit_cv_gain_by_seed": g.round(4).tolist(),
           "accept": accept, "concepts": kept, "diagnostic_corr_with_planted": diag}
    st["rounds"].append(rec); STATE.write_text(json.dumps(st, indent=1)); (OUT / f"round{rnd}_score.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("kept", "fit_cv_gain_mean", "fit_cv_gain_by_seed", "accept")}))
    for nm, dd in diag.items():
        best = max(dd.items(), key=lambda kv: abs(kv[1])) if dd else None
        print(f"  {nm:34s} {kept[nm]['kind']:6s} rate {kept[nm]['fit_rate']:.3f}  best planted match {best}")


def final():
    from pythia.core.tight_bounds import scalar_cert, tune_cert, tame, centre_within
    st = state(); ex = accepted_extra(st)
    models = {"table_only": [], "STACK_old": tb.OLD, "STACK_new(+symbolic)": tb.NEW, "AGENT(+concepts)": base_keys(st),
              "oracle": ["oracle(reference)"]}
    J = 4 + 2 + st["proposed_sets"]           # declared portfolio: model variants + every proposed concept set
    a = .05 / J; CAP = 60.; res = {"J": J, "alpha_each": a}
    eps = np.random.default_rng(3).normal(0, 20, (3000, 1)); m_cap = np.clip(tb.mu - tb.f + eps, -CAP, CAP).mean(0)
    res["truth_CAL_min2"] = float(np.mean(centre_within(m_cap[tb.cal], tb.cell[tb.cal]) ** 2))
    for k, keys in models.items():
        row = {}
        if keys:
            h = tb.predictor(keys, tb.fit, extra=ex); o = tb.oof(keys, tb.fit, extra=ex)
            row["TEST_R2"] = tb.r2(h, tb.te); row["TEST_MAE_min"] = float(np.abs(tb.r[tb.te] - h[tb.te]).mean())
            row["corr_true_direction_TEST"] = float(np.corrcoef(h[tb.te], tb.true_dir[tb.te])[0, 1])
            cfg = tune_cert(centre_within(o[tb.fit], tb.cell[tb.fit]), tb.r[tb.fit], tb.cell[tb.fit], a, int(tb.cal.sum()), caps=(CAP,))
            hc = tame(h[tb.cal], tb.cell[tb.cal], cfg["q"])
            row["cert_betting_min2"] = scalar_cert(hc, tb.r[tb.cal], CAP, a, method="betting", lam=cfg["lam"], groups=tb.cell[tb.cal]).bound
            row["cert_mp_min2"] = scalar_cert(hc, tb.r[tb.cal], CAP, a, method="mp", groups=tb.cell[tb.cal]).bound
        else:
            row["TEST_MAE_min"] = float(np.abs(tb.r[tb.te]).mean())
        ph, _ = tb.risk_forecast(keys, extra=ex); row["risk"] = tb.metrics(ph)
        row["labeler"] = {kk: vv for kk, vv in tb.certified_labeler(ph).items() if kk != "directions"}
        res[k] = row
        print(k, json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in row.items() if kk not in ("risk", "labeler")}),
              "| risk", {kk: round(vv, 4) for kk, vv in row["risk"].items()}, "| labeler", row["labeler"])
    (OUT / "final.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "diagnose":
        diagnose(int(sys.argv[2]))
    elif cmd == "score":
        score(int(sys.argv[2]), sys.argv[3])
    else:
        final()
