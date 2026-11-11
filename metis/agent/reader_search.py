"""Metis agentic reader search: an LLM proposes *what to ask the notes*, Jev answers, Pythia decides.

One round:
  1. DIAGNOSE (FIT only)  out-of-fold risk residuals r_i = miss_i - p_hat_i of the current risk model; rank note terms
                          by association with r (where information about the model's errors is still unread).
  2. PROPOSE (LLM via the opmed LiteLLM)  turn the top terms into <= 5 Jev questions (noul / choice / score),
                          strictly schema-validated.  The LLM never sees outcomes, only terms and existing questions.
  3. PROBE (Jev, budgeted) answer the new questions on a FIT sample only.
  4. SCORE (FIT only)     cross-fitted gain of the variational/stacked score with vs without the new reader.
  5. ACCEPT -> full read  only accepted question sets are run on all rows (one more reader in the portfolio).
Safety: selection uses FIT outcomes only, so CAL certificates need alpha / (#final readers); even if CAL were used for
selection, Pythia's portfolio rule (alpha / #question sets ever proposed) would keep every certificate valid.
Budget: a global USD cap over every Jev call (file-based ledger); LiteLLM spend is capped by the LiteLLM key.
"""
from __future__ import annotations

import json, re, time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

SCHEMA_HINT = (
    'Return ONLY JSON: {"questions": {"<snake_case_name>": {"type": "noul"|"choice"|"score", "instructions": "<question>", '
    '"criteria": <for noul: {"true": "...", "false": "..."}; for choice: {"<option>": "<definition>", ...} (2-5 options); '
    'for score: ["<level 0>", "<level 1>", ...] (3-5 ordered levels)>}}}')


# ---------------------------------------------------------------------------------------------------- diagnose
def residual_terms(texts_fit: Sequence[str], resid: np.ndarray, k: int = 25, min_df: int = 20) -> list[tuple[str, float]]:
    """Terms whose presence correlates with the out-of-fold residual (FIT rows only; aggregate statistics)."""
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(min_df=min_df, ngram_range=(1, 2), binary=True, stop_words="english", max_features=20000)
    X = cv.fit_transform(texts_fit).astype(float); r = resid - resid.mean()
    m = np.asarray(X.mean(0)).ravel(); cov = np.asarray(X.T @ r).ravel() / len(r)
    sd = np.sqrt(m * (1 - m)) * r.std() + 1e-12; corr = cov / sd
    names = np.array(cv.get_feature_names_out()); order = np.argsort(-np.abs(corr))[:k]
    return [(names[j], float(corr[j])) for j in order]


# ---------------------------------------------------------------------------------------------------- propose
def proposal_prompt(terms, existing: dict, rejected: Sequence[str] = ()) -> str:
    return ("You design yes/no, multiple-choice and ordinal questions that a clinical-note reader will answer for every "
            "note, to predict whether a surgical case runs much longer or shorter than expected. Terms below are "
            "associated with the current model's errors (positive = cases run longer than predicted, negative = "
            "shorter). Propose at most 5 NEW questions about clinically meaningful facts behind these terms (not about "
            "the words themselves), avoiding the existing questions. Questions must be answerable from the note alone.\n"
            f"TERMS: {json.dumps([(t, round(c, 3)) for t, c in terms])}\nEXISTING: {json.dumps(list(existing))}\n"
            f"REJECTED (asked before, added no information; do not ask similar things): {json.dumps(list(rejected))}\n"
            "Prefer facts that change operative time or risk: patient comorbidities, medications, prior surgery, "
            "anesthesia, blood loss, complications, case urgency or complexity.\n"
            + SCHEMA_HINT)


def validate_questions(obj) -> dict:
    """Keep only well-formed Jev questions; raise if none survive."""
    qs = (obj or {}).get("questions", obj) or {}; ok = {}
    for name, q in list(qs.items())[:5]:
        name = re.sub(r"[^a-z0-9_]", "_", str(name).lower())[:40]
        t = q.get("type"); crit = q.get("criteria"); ins = str(q.get("instructions", ""))[:300]
        if not ins:
            continue
        if t == "noul" and isinstance(crit, dict) and {"true", "false"} <= set(crit):
            ok[name] = {"type": t, "instructions": ins, "criteria": {"true": str(crit["true"])[:200], "false": str(crit["false"])[:200]}}
        elif t == "choice" and isinstance(crit, dict) and 2 <= len(crit) <= 5:
            ok[name] = {"type": t, "instructions": ins,
                        "criteria": {re.sub(r"[^a-z0-9_]", "_", str(k).lower())[:30]: str(v)[:200] for k, v in crit.items()}}
        elif t == "score" and isinstance(crit, list) and 3 <= len(crit) <= 5:
            ok[name] = {"type": t, "instructions": ins, "criteria": [str(c)[:120] for c in crit]}
    if not ok:
        raise ValueError("no valid questions in proposal")
    return ok


# ---------------------------------------------------------------------------------------------------- Jev (budgeted)
class JevLedger:
    """Append-only answers file + global USD cap across all question sets."""

    def __init__(self, path: Path, cap_usd: float):
        self.path, self.cap = Path(path), cap_usd

    def spent(self) -> float:
        if not self.path.exists():
            return 0.0
        return sum((json.loads(l).get("cost") or 0) for l in open(self.path))

    def done(self, qset: str) -> set:
        if not self.path.exists():
            return set()
        return {d["i"] for d in map(json.loads, open(self.path)) if d.get("qset") == qset and d.get("ok")}

    def run(self, texts: dict, qset: str, questions: dict, ask: Callable, workers: int = 8, max_chars: int = 12000) -> dict:
        import concurrent.futures as cf
        todo = [i for i in texts if i not in self.done(qset)]; spent = self.spent()
        if spent >= self.cap:
            return {"stopped": "budget", "spent": spent}

        def one(i):
            for a in range(3):
                try:
                    r = ask((texts[i] or "")[:max_chars], questions)
                    return {"i": int(i), "qset": qset, "ok": True, "answers": r.get("answers"),
                            "cost": (r.get("usage") or {}).get("cost") or 0}
                except Exception as e:  # noqa
                    err = str(e)[:160]; time.sleep(1.5 * (a + 1))
            return {"i": int(i), "qset": qset, "ok": False, "error": err, "cost": 0}
        n = 0
        with open(self.path, "a") as fh, cf.ThreadPoolExecutor(workers) as ex:
            for d in ex.map(one, todo):
                fh.write(json.dumps(d) + "\n"); fh.flush(); spent += d["cost"]; n += 1
                if spent >= self.cap:
                    ex.shutdown(cancel_futures=True)
                    return {"stopped": "budget", "spent": spent, "answered": n}
        return {"stopped": "complete", "spent": spent, "answered": n}


# ---------------------------------------------------------------------------------------------------- score
def fit_gain(X_base, X_new, miss, cell, entity_ids, folds: int = 2, seed: int = 0) -> dict:
    """FIT-only cross-fitted alignment of the stacked score with and without the new reader (no CAL use).
    Returns the plug-in signal C^2/H (an estimate of the certifiable information) for both."""
    from horos.represent.score_learning import variational_scores as vs, oof_scores, stack_weights, stack_scores, _centre
    from horos.represent.score_learning import _entity_fold_labels
    r = _entity_fold_labels(entity_ids, len(miss), folds, seed); res = {}
    for name, Xs in (("base", [X_base]), ("with_new", [X_base, X_new])):
        vals = []
        for f in range(folds):
            tr, te = r != f, r == f
            H = [vs(X[tr], miss[tr], cell[tr], X[te], cell[te], 2) for X in Xs]
            w = stack_weights([oof_scores(vs, X[tr], miss[tr], cell[tr], 2,
                                           entity_ids=np.asarray(entity_ids)[tr], seed=seed + f)
                               for X in Xs], miss[tr], cell[tr]) if len(Xs) > 1 else [1.]
            Hc = _centre(stack_scores(H, w), cell[te]); C = Hc[np.arange(te.sum()), miss[te]].mean()
            vals.append(max(C, 0) ** 2 / max((Hc ** 2).sum(1).mean(), 1e-12))
        res[name] = float(np.mean(vals))
    res["gain"] = res["with_new"] - res["base"]
    return res
