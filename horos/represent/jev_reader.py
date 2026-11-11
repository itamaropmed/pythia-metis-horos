"""Jev (TypeSafe, via OpenRouter Decisions API) as a *probabilistic reader* for Pythia.

Jev is a non-generative "System One" decision model: it answers only the questions you declare, over the
options you declare, and returns a probability per option (no free text, ~0.3 s, priced per input token).
That is exactly a Pythia reader with a pre-declared schema: the questions are fixed before any outcome is
seen, the output is a probability vector per question, and it cannot hallucinate fields that were not asked.

Privacy: this is a REMOTE call.  Use only on public data or de-identified text (pass a de-identifier, e.g.
horos.represent.openmed_readers.OpenMedPII(...).deidentify).  Never on Mayo notes without approval.

  export OPENROUTER_API_KEY=...   (or ~/PycharmProjects/.openrouter_key, never committed)
"""
from __future__ import annotations

import json, os, time, urllib.request
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

ENDPOINT = os.environ.get("JEV_ENDPOINT", "https://openrouter.ai/api/alpha/decisions")
MODEL = os.environ.get("JEV_MODEL", "typesafe/jev-1.14")

# Declared before outcomes (portfolio size counts each question set as one reader).
SURGICAL_QUESTIONS = {
    "anesthesia": {"type": "choice", "instructions": "What anesthesia did the patient receive for the procedure?",
                   "criteria": {"general": "General anesthesia, including general endotracheal or inhalational.",
                                "regional": "Spinal, epidural or a peripheral nerve block as the main anesthetic.",
                                "local_mac": "Local anesthesia or monitored anesthesia care / sedation.",
                                "not_stated": "The note does not describe an anesthetic."}},
    "blood_loss": {"type": "score", "instructions": "How much blood loss does the note report?",
                   "criteria": ["none or minimal (under 50 ml)", "50 to 199 ml", "200 to 499 ml",
                                "500 ml or more"]},
    "complication": {"type": "noul", "instructions": "Did an intra- or post-operative complication occur?",
                     "criteria": {"true": "The note reports a complication that actually happened.",
                                  "false": "No complication, a negated mention such as 'no complications', or not stated."}},
    "emergency": {"type": "noul", "instructions": "Was this an emergency or urgent case?",
                  "criteria": {"true": "Emergent or urgent surgery.", "false": "Elective or not stated."}},
    "complexity": {"type": "score", "instructions": "How complex is the procedure described?",
                   "criteria": ["minor", "moderate", "major", "very complex or multi-stage"]},
}


def api_key() -> str:
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k:
        p = Path.home() / "PycharmProjects" / ".openrouter_key"
        k = p.read_text().strip() if p.exists() else None
    if not k:
        raise RuntimeError("set OPENROUTER_API_KEY or create ~/PycharmProjects/.openrouter_key")
    return k


def ask(text: str, questions: dict = SURGICAL_QUESTIONS, model: str = MODEL, timeout: int = 60) -> dict:
    body = {"model": model, "state": {"clinical_note": text}, "questions": questions}
    req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def to_features(answers: Optional[dict], questions: dict = SURGICAL_QUESTIONS) -> np.ndarray:
    """Probability vector per question, concatenated (choice/score: one entry per option; noul: P(true)).
    Missing answers -> uniform probabilities (an uninformative read, never a fabricated one)."""
    out = []
    for name, q in questions.items():
        a = (answers or {}).get(name) or {}
        if q["type"] == "noul":
            out.append(float(a.get("noul", .5)))
        else:
            keys = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
            p = a.get("probabilities") or {}
            out += [float(p.get(k, 1 / len(keys))) for k in keys]
    return np.asarray(out)


def feature_names(questions: dict = SURGICAL_QUESTIONS) -> list[str]:
    names = []
    for name, q in questions.items():
        if q["type"] == "noul":
            names.append(f"{name}:P(true)")
        else:
            keys = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
            names += [f"{name}:{k}" for k in keys]
    return names


def read_corpus(texts: Sequence[str], out_jsonl: Path, deidentify: Optional[Callable[[str], str]] = None,
                questions: dict = SURGICAL_QUESTIONS, max_chars: int = 12000, workers: int = 8,
                max_usd: float = 1.0) -> list:
    """Resumable corpus run; writes {i, answers, cost} per note (no note text)."""
    import concurrent.futures as cf
    done = {}
    if out_jsonl.exists():
        for l in open(out_jsonl):
            d = json.loads(l)
            if d.get("ok"):
                done[d["i"]] = d

    def one(i):
        t = (texts[i] or "")[:max_chars]
        if deidentify:
            t = deidentify(t)
        for attempt in range(4):
            try:
                r = ask(t, questions)
                return {"i": i, "ok": True, "answers": r.get("answers"), "cost": (r.get("usage") or {}).get("cost")}
            except Exception as e:  # noqa
                err = str(e)[:200]; time.sleep(2 ** attempt)
        return {"i": i, "ok": False, "error": err}

    spent = sum((d.get("cost") or 0) for d in done.values())
    todo = [i for i in range(len(texts)) if i not in done]
    with open(out_jsonl, "a") as fh, cf.ThreadPoolExecutor(workers) as ex:
        for k, d in enumerate(ex.map(one, todo)):
            fh.write(json.dumps(d) + "\n")
            if d["ok"]:
                done[d["i"]] = d; spent += d.get("cost") or 0
            if spent >= max_usd:                       # hard budget stop (resumable later)
                print(f"budget reached: {spent:.4f} USD; stopping", flush=True); ex.shutdown(cancel_futures=True); break
            if k % 200 == 0:
                fh.flush(); print(f"jev {len(done)}/{len(texts)}", flush=True)
    return [done.get(i, {}).get("answers") for i in range(len(texts))]
