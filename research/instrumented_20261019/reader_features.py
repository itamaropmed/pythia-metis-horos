"""Label-free reader features for the testbed rows (each loader returns None when its cache is absent)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CACHE = HERE / "results/reader_cache"


def _num(v, d):
    try:
        return float(v)
    except Exception:
        return d


def llm_extract(S, reader="gpt-6-luna|extract_v2", path=HERE / "readers.jsonl"):
    if not Path(path).exists():
        return None
    dd = {}
    for l in open(path):
        r = json.loads(l)
        if r.get("reader") == reader and r.get("ok") and r.get("json"):
            dd[r["i"]] = r["json"]
    if np.mean([int(i) in dd for i in S]) < .99:
        return None

    def val(j, k, opts):
        return [float(str(j.get(k, "")).lower() == o) for o in opts]
    return np.array([val(j, "anesthesia", ["general", "regional", "local", "mac", "none"]) +
                     [min(_num(j.get("blood_loss_ml"), 0.), 2000) / 500] + val(j, "complications", ["yes"]) +
                     val(j, "emergency", ["yes"]) + [_num(j.get("complexity_1to5"), 3.) / 5]
                     for j in (dd.get(int(i), {}) for i in S)])


def tfidf(T, k=200, *, fit_rows=None):
    """Fit the TF-IDF/SVD text basis on development rows, then transform all rows.

    Requiring ``fit_rows`` prevents held-out notes from influencing vocabulary,
    IDF weights, or SVD axes. Pass a boolean mask or integer row indices.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = np.asarray(T, dtype=str).reshape(-1)
    if fit_rows is None:
        raise ValueError("fit_rows is required; held-out text must not fit the representation")
    rows = np.asarray(fit_rows)
    if rows.dtype == bool:
        if rows.shape != docs.shape:
            raise ValueError("boolean fit_rows must align with T")
        mask = rows
    else:
        indices = rows.astype(int).reshape(-1)
        if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= len(docs)):
            raise ValueError("fit_rows indices must be nonempty and in range")
        mask = np.zeros(len(docs), dtype=bool)
        mask[indices] = True
    if not mask.any():
        raise ValueError("fit_rows must include at least one row")
    vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1, 2), max_features=20000, sublinear_tf=True)
    X_fit = vectorizer.fit_transform(docs[mask])
    X_all = vectorizer.transform(docs)
    if k > min(X_fit.shape):
        raise ValueError(f"SVD components ({k}) exceed fit-matrix dimensions {X_fit.shape}")
    svd = TruncatedSVD(k, random_state=0).fit(X_fit)
    return svd.transform(X_all)


def ner_bag(S, key, min_docs=30, *, fit_rows=None):
    """Build an entity bag whose vocabulary is selected on development rows only."""
    f = CACHE / f"ner_{key}.jsonl"
    if not f.exists():
        return None
    row_ids = np.asarray(S).reshape(-1)
    if fit_rows is None:
        raise ValueError("fit_rows is required; held-out documents must not define the vocabulary")
    rows = np.asarray(fit_rows)
    if rows.dtype == bool:
        if rows.shape != row_ids.shape:
            raise ValueError("boolean fit_rows must align with S")
        mask = rows
    else:
        indices = rows.astype(int).reshape(-1)
        if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= len(row_ids)):
            raise ValueError("fit_rows indices must be nonempty and in range")
        mask = np.zeros(len(row_ids), dtype=bool)
        mask[indices] = True
    if not mask.any():
        raise ValueError("fit_rows must include at least one row")
    E = {}
    for l in open(f):
        d = json.loads(l); E[d["i"]] = d["v"]
    if not all(int(i) in E for i in row_ids):
        return None
    docs = [[t for _, t in E[int(i)]] for i in row_ids]
    from collections import Counter
    df = Counter(t for d in np.asarray(docs, dtype=object)[mask] for t in set(d))
    vocab = sorted(t for t, c in df.items() if c >= min_docs); vi = {t: j for j, t in enumerate(vocab)}
    X = np.zeros((len(S), len(vocab) + 1))
    for r, d in enumerate(docs):
        X[r, -1] = np.log1p(len(d))
        for t in set(d):
            if t in vi:
                X[r, vi[t]] = 1.
    return X


def jev(S, path=CACHE / "jev_agent.jsonl", qset=None):
    """Jev answers for one question set (qset id) -> probability features; None if incomplete."""
    if not Path(path).exists():
        return None
    from horos.represent.jev_reader import to_features
    dd = {}
    Q = json.loads((CACHE / "jev_qsets.json").read_text()).get(qset) if (CACHE / "jev_qsets.json").exists() else None
    for l in open(path):
        d = json.loads(l)
        if d.get("ok") and d.get("qset") == qset:
            dd[d["i"]] = d["answers"]
    if np.mean([int(i) in dd for i in S]) < .99 or Q is None:
        return None
    return np.vstack([to_features(dd.get(int(i)), Q) for i in S])


def llm_qa(S, qset, path=CACHE / "llm_qa.jsonl"):
    """LLM answers (Jev-style typed questions) for one agent question set -> probability features."""
    if not Path(path).exists() or not (CACHE / "jev_qsets.json").exists():
        return None
    Q = json.loads((CACHE / "jev_qsets.json").read_text()).get(qset)
    if Q is None:
        return None
    dd = {}
    for l in open(path):
        d = json.loads(l)
        if d.get("qset") == qset and d.get("ok"):
            dd[d["i"]] = d["answers"] or {}
    if np.mean([int(i) in dd for i in S]) < .95:
        return None
    rows = []
    for i in S:
        a = dd.get(int(i), {}); r = []
        for name, q in Q.items():
            v = a.get(name)
            if q["type"] == "noul":
                r.append(_num(v if not isinstance(v, dict) else v.get("yes", v.get("true", .5)), .5))
            else:
                keys = list(q["criteria"]) if q["type"] == "choice" else [str(j) for j in range(len(q["criteria"]))]
                v = v if isinstance(v, dict) else {}
                r += [_num(v.get(k, v.get(int(k)) if k.isdigit() else None), 1 / len(keys)) for k in keys]
        rows.append(r)
    return np.array(rows, float)
