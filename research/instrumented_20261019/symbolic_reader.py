"""Embedding-free, table-conditioned representation of a note: the *symbolic sample space*.

The table-conditioned residual reader (residual_reader_vm.py) maps each note, GIVEN the table context, to
  * a scalar information direction   shift_min  ~ E[r | X, T]        (zero-shot: no training, label-free everywhere)
  * two tail probabilities           p_long30, p_short30            (zero-shot risk relative to the table)
  * a finite set of typed, signed atomic propositions  {(fact, type, dir, minutes)}  -- the note's outcome-relevant
    sample space, conditioned on what the table already knows.
Features built here (all label-free; vocabulary frequencies use no outcomes):
  rr_shift   [shift_min]
  rr_typed   per-type signed minutes + per-type counts + shift/p_long/p_short + parsed EBL      (~30 dims)
  rr_atoms   indicators of (type, token) atoms that occur in >= min_docs notes                  (sparse, ~hundreds)
"""
from __future__ import annotations

import json, re
from collections import Counter
from pathlib import Path

import numpy as np

TYPES = ["comorbidity", "medication", "anesthesia", "blood_loss", "procedure_scope", "anatomy", "finding",
         "complication", "urgency", "patient", "other"]
STOP = set("a an the of with and or in on for to no not less than ml cc mild moderate severe".split())
_LOCAL = Path(__file__).resolve().parent / "results/reader_cache/residual_reader.jsonl"
DEFAULT = _LOCAL if _LOCAL.exists() else Path("/mnt/user-data/uploads/PycharmProjects/pythia-metis-horos/research/instrumented_20261019/results/reader_cache/residual_reader.jsonl")


def _num(v, d=0.):
    try:
        return float(v)
    except Exception:
        return d


def load_answers(S, path=DEFAULT):
    A = {}
    for l in open(path):
        d = json.loads(l)
        if d.get("ok") and isinstance(d.get("ans"), dict):
            A[d["i"]] = d["ans"]
    return [A.get(int(i), {}) for i in S]


def drivers(a):
    out = []
    for x in a.get("drivers") or []:
        if not isinstance(x, dict):
            continue
        t = str(x.get("type", "other")).lower(); t = t if t in TYPES else "other"
        m = float(np.clip(_num(x.get("min"), 0.), -60, 60)); dr = str(x.get("dir", "0"))   # winsorised: LLM minutes are heavy-tailed
        if dr == "-" and m > 0:
            m = -m
        out.append((str(x.get("fact", "")).lower(), t, m))
    return out


def ebl_ml(a):
    best = 0.
    for f, t, _ in drivers(a):
        if t == "blood_loss" or "blood loss" in f or "ebl" in f:
            if re.search(r"minimal|negligible|less than (5|10)\b|scant", f):
                best = max(best, 5.)
            m = re.findall(r"(\d+(?:\.\d+)?)\s*(?:ml|cc)", f)
            if m:
                best = max(best, min(float(m[-1]), 2000.))
    return best


def features(S, path=DEFAULT, min_docs=15):
    A = load_answers(S, path)
    shift = np.clip(np.array([_num(a.get("shift_min")) for a in A]), -90, 90)     # label-free winsorisation
    pl = np.array([_num(a.get("p_long30"), .1) for a in A]); ps = np.array([_num(a.get("p_short30"), .1) for a in A])
    typed = np.zeros((len(A), 2 * len(TYPES)))
    toks = []
    for r, a in enumerate(A):
        tk = set()
        for f, t, m in drivers(a):
            j = TYPES.index(t); typed[r, j] += m; typed[r, len(TYPES) + j] += 1
            for w in re.findall(r"[a-z][a-z\-]+", f):
                if w not in STOP and len(w) > 2:
                    tk.add(f"{t}:{w}")
        toks.append(tk)
    ebl = np.array([ebl_ml(a) for a in A])
    typed[:, :len(TYPES)] = np.clip(typed[:, :len(TYPES)], -120, 120)
    rr_typed = np.column_stack([typed, shift, pl, ps, np.log1p(ebl), (ebl >= 100).astype(float)])
    df = Counter(t for s in toks for t in s); vocab = sorted(t for t, c in df.items() if c >= min_docs)
    vi = {t: j for j, t in enumerate(vocab)}; X = np.zeros((len(A), len(vocab)))
    for r, s in enumerate(toks):
        for t in s:
            if t in vi:
                X[r, vi[t]] = 1.
    return {"rr_shift": shift[:, None], "rr_typed": rr_typed, "rr_atoms": np.column_stack([rr_typed, X]),
            "_p_long": pl, "_p_short": ps, "_vocab": vocab, "_atoms": toks, "_answers": A}
