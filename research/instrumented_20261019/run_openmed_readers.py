"""Run OpenMed readers over MTSamples and cache label-free features for mtsamples_certify.py.

  python3 run_openmed_readers.py deid  --pii-model /path/or/hf-id      # de-identified text -> tfidf reader
  python3 run_openmed_readers.py ner   --ner disease_tiny,pharma         # entity-bag readers (needs weights)

Outputs results/reader_cache/openmed_*.npy (features only, no text) and a PHI-label count summary.
Resumable: progress is stored in results/reader_cache/*.jsonl.
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np, pandas as pd

HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parents[1]))
from horos.represent.openmed_readers import OPENMED_NER, OpenMedNER, OpenMedPII, entity_bag_features  # noqa

CACHE = HERE / "results" / "reader_cache"; CACHE.mkdir(parents=True, exist_ok=True)
T = pd.read_csv(os.environ.get("PERAS_DATA", "/home/claude/data") + "/mtsamples.csv").fillna("")["transcription"].astype(str).tolist()


def resumable(path: Path, fn, limit):
    done = {}
    if path.exists():
        for l in open(path):
            d = json.loads(l); done[d["i"]] = d["v"]
    with open(path, "a") as fh:
        for i, t in enumerate(T[:limit]):
            if i in done:
                continue
            done[i] = fn(t); fh.write(json.dumps({"i": i, "v": done[i]}) + "\n"); fh.flush()
    return [done[i] for i in range(min(limit, len(T)))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["deid", "ner"])
    ap.add_argument("--pii-model", default="OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1")
    ap.add_argument("--ner", default="disease_tiny"); ap.add_argument("--limit", type=int, default=10 ** 9)
    a = ap.parse_args()
    if a.what == "deid":
        pii = OpenMedPII(a.pii_model, max_chars=20000)
        # keep only what the tfidf reader needs: the de-identified text stays inside the cache folder (public data)
        D = resumable(CACHE / "deid_text.jsonl", pii.deidentify, a.limit)
        if len(D) == len(T):
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            tf = TfidfVectorizer(min_df=5, ngram_range=(1, 2), max_features=20000, sublinear_tf=True).fit_transform(D)
            np.save(CACHE / "openmed_deid_tfidf_svd200.npy", TruncatedSVD(200, random_state=0).fit_transform(tf))
            import re
            tags = pd.Series([t for d in D for t in re.findall(r"\[([a-z_A-Z]+)\]", d)]).value_counts()
            (CACHE / "deid_tag_counts.json").write_text(tags.to_json(indent=1))
    else:
        for key in a.ner.split(","):
            path = os.environ.get("OPENMED_MODEL_DIR", "/home/claude/models") + f"/openmed_ner_{key}"
            ner = OpenMedNER(path if os.path.isdir(path) else OPENMED_NER.get(key, key))
            f = CACHE / f"ner_{key}.jsonl"; done = {}
            if f.exists():
                for l in open(f):
                    d = json.loads(l); done[d["i"]] = d["v"]
            todo = [i for i in range(min(a.limit, len(T))) if i not in done]
            with open(f, "a") as fh:
                for k in range(0, len(todo), 200):
                    idx = todo[k:k + 200]
                    for i, e in zip(idx, ner.entities_batch([T[i] for i in idx])):
                        done[i] = e; fh.write(json.dumps({"i": i, "v": e}) + "\n")
                    fh.flush(); print(key, len(done), flush=True)
            if len(done) == len(T):
                E = [done[i] for i in range(len(T))]
                X, cols = entity_bag_features([[tuple(x) for x in e] for e in E], np.arange(len(T)), min_groups=30)
                np.save(CACHE / f"openmed_ner_{key}.npy", X)
                (CACHE / f"openmed_ner_{key}_columns.json").write_text(json.dumps(cols))

if __name__ == "__main__":
    main()
