"""Instrumented-information certificates on REAL MTSamples notes with a PLANTED binary outcome (known truth).

Truth: P(Y=1 | table, note) = sigmoid(a_cell + 1.2*[ANESTHESIA says general] + 1.0*[EBL >= 100 ml]
                                      + 1.5*[complication present, not negated])
Readers: tfidf (bag of words), oracle (the planted concepts themselves; reference ceiling), and every LLM reader
found in readers.jsonl (produced on the Mac by llm_reader_mtsamples.py).
"""
from __future__ import annotations

import json, os, re, sys
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE.parents[1] / "research/note_atlas_20261018"))
from pythia.core.instrumented import (_groups, information_ledger, loo_centre_groups, projection_bound,  # noqa
                                      reader_portfolio, regression_test, true_spread)
from atlas import segment  # noqa

ALPHA = .05
df = pd.read_csv(os.environ.get("PERAS_DATA", "/home/claude/data") + "/mtsamples.csv").fillna("")
T = df["transcription"].astype(str).tolist(); n = len(T)
rng = np.random.default_rng(20261019)


def sec(t, name):
    return " ".join(b for h, b in segment(t) if name in h).lower()


anes = np.array([("general" in sec(t, "ANESTHESIA")) for t in T], float)
ebl = np.array([(lambda m: float(m.group(1)) if m else 0.)(re.search(r"(\d+)\s*(?:ml|cc)", sec(t, "BLOOD LOSS")))
                for t in T])
comp = np.array([(lambda s: bool(s) and not re.search(r"\b(no|none|without)\b", s))(sec(t, "COMPLICATION"))
                 for t in T], float)
spec = df["medical_specialty"].astype(str).str.strip().to_numpy()
top = pd.Series(spec).value_counts().index[:15]
cell = np.where(np.isin(spec, top), spec, "other")
a_cell = pd.Series(cell).map(pd.Series(rng.normal(-.3, .6, len(np.unique(cell))), index=np.unique(cell))).to_numpy()
p1 = 1 / (1 + np.exp(-(a_cell + 1.2 * anes + 1.0 * (ebl >= 100) + 1.5 * comp)))
P = np.column_stack([1 - p1, p1]); y = (rng.random(n) < p1).astype(int)
folds = rng.permutation(n) % 2
print(f"n={n}; prevalence anes {anes.mean():.3f} ebl>=100 {(ebl >= 100).mean():.3f} comp {comp.mean():.3f}; "
      f"true NLI(T|X) = {true_spread(P, cell):.5f}", flush=True)


def crossfit(Xf):
    Q = np.zeros((n, 2)); Xc = np.column_stack([pd.get_dummies(cell).to_numpy(float), Xf])
    for f in (0, 1):
        tr, te = folds != f, folds == f
        Q[te] = LogisticRegression(C=1., max_iter=2000).fit(Xc[tr], y[tr]).predict_proba(Xc[te])
    return Q


def certify(Q, alpha):
    L = 0.
    for f in (0, 1):
        I = np.flatnonzero(folds == f)
        group_i = _groups(cell[I])
        L += len(I) / n * projection_bound(loo_centre_groups(Q[I], group_i), y[I],
                                            alpha / 2, groups=group_i).bound
    return L


readers, feats = {}, {}
tf = TfidfVectorizer(min_df=5, ngram_range=(1, 2), max_features=20000, sublinear_tf=True).fit_transform(T)
from sklearn.decomposition import TruncatedSVD  # noqa
feats["tfidf_svd200"] = TruncatedSVD(200, random_state=0).fit_transform(tf)
feats["oracle_concepts"] = np.column_stack([anes, ebl >= 100, comp]).astype(float)

# ---- open-source medical readers (local, free): medspaCy ConText rules; OpenMed NER / PII when cached
CACHE = HERE / "results" / "reader_cache"; CACHE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE.parents[1]))
if os.environ.get("FAST_RULES", "1") == "1":
    f = CACHE / "fast_rules.npy"
    if not f.exists():
        from horos.represent.openmed_readers import FastRuleReader
        np.save(f, FastRuleReader().features(T))
    feats["medspacy_context_rules"] = np.load(f)
for f in sorted(list(CACHE.glob("openmed_*.npy")) + list(CACHE.glob("jev_*.npy"))):   # run_openmed_readers / run_jev_reader
    x = np.load(f)
    if len(x) == n:
        feats[f.stem] = x

rj = HERE / "readers.jsonl"
if rj.exists():
    rows = [json.loads(l) for l in open(rj)]
    by = {}
    for r in rows:
        by.setdefault(r["reader"], {})[r["i"]] = r.get("json") or {}
    for name, d in by.items():
        if len(d) < .9 * n:
            continue
        J = [d.get(i, {}) for i in range(n)]
        if "judge" in name:
            x = np.array([[float(j.get("p_long", 50) or 50) / 100] for j in J])
        else:
            def val(j, k, opts):
                return [float(str(j.get(k, "")).lower() == o) for o in opts]
            x = np.array([val(j, "anesthesia", ["general", "regional", "local", "mac", "none"]) +
                          [min(float(j.get("blood_loss_ml") or 0), 2000) / 500] + val(j, "complications", ["yes"]) +
                          val(j, "emergency", ["yes"]) + [float(j.get("complexity_1to5") or 3) / 5] for j in J])
        feats["llm:" + name] = x

ner = [k for k in feats if k.startswith("openmed_ner_")]
if len(ner) > 1:                                           # one reader that reads with all OpenMed NER models
    feats["openmed_ner_all"] = np.column_stack([feats[k] for k in ner])
for k, x in feats.items():
    readers[k] = crossfit(x)
J = len(readers) + 4          # + NER ensemble, NER ledger, LLM ensemble, LLM ledger: one alpha ledger
out = {"n": n, "true_nli": true_spread(P, cell), "alpha": ALPHA, "J": J,
       "single_reader_bounds_alpha_over_J": {k: certify(Q, ALPHA / J) for k, Q in readers.items()}}
best = max(out["single_reader_bounds_alpha_over_J"], key=out["single_reader_bounds_alpha_over_J"].get)
out["portfolio_best"] = best
if ner:
    ens = np.mean([readers[k] for k in ner], 0)
    out["openmed_ner_probability_ensemble"] = certify(ens, ALPHA / J)
    if "medspacy_context_rules" in feats:                   # ledger: rule bins first, then all-NER reader
        bins = _groups(*feats["medspacy_context_rules"].T)
        lg = {"between": 0., "beyond": 0.}
        for f in (0, 1):
            I = np.flatnonzero(folds == f)
            r = information_ledger(cell[I], bins[I], readers["medspacy_context_rules"][I],
                                   readers.get("openmed_ner_all", readers[ner[0]])[I], y[I], ALPHA / (2 * J))
            lg["between"] += len(I) / n * r["between_a"]; lg["beyond"] += len(I) / n * r["beyond_a"]
        out["ledger_rules_then_openmed"] = lg
llm = [k for k in readers if k.startswith("llm:")]
if llm:
    ens = np.mean([readers[k] for k in llm], 0)
    out["llm_prompt_model_ensemble"] = certify(ens, ALPHA / J)
    # ledger: bins of the first extraction reader, second part with tfidf
    ext = [k for k in llm if "extract" in k]
    if ext:
        bins = _groups(*feats[ext[0]].T.round(2))
        lg = {"between": 0., "beyond": 0.}
        for f in (0, 1):
            I = np.flatnonzero(folds == f)
            r = information_ledger(cell[I], bins[I], readers[ext[0]][I], readers["tfidf_svd200"][I], y[I], ALPHA / (2 * J))
            lg["between"] += len(I) / n * r["between_a"]; lg["beyond"] += len(I) / n * r["beyond_a"]
        out["ledger_" + ext[0] + "_then_tfidf"] = lg
print(json.dumps(out, indent=1))
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "mtsamples_instrumented.json").write_text(json.dumps(out, indent=1))
