"""Run Jev over PUBLIC MTSamples notes and cache its probability features for mtsamples_certify.py.

  export OPENROUTER_API_KEY=...        # or ~/PycharmProjects/.openrouter_key
  python3 run_jev_reader.py [--limit 4999] [--deidentify-with /path/to/openmed_pii_model]

Cost estimate: ~1,500 input tokens per note; at Jev's per-token price the full corpus is well under 1 USD.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np, pandas as pd

HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parents[1]))
from horos.represent.jev_reader import feature_names, read_corpus, to_features  # noqa

ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=10 ** 9)
ap.add_argument("--deidentify-with", default=""); ap.add_argument("--max-usd", type=float, default=1.0); ap.add_argument("--csv", default=os.environ.get(
    "PERAS_DATA", "/home/claude/data") + "/mtsamples.csv")
a = ap.parse_args()
T = pd.read_csv(a.csv).fillna("")["transcription"].astype(str).tolist()[: a.limit]
deid = None
if a.deidentify_with:
    from horos.represent.openmed_readers import OpenMedPII
    deid = OpenMedPII(a.deidentify_with, max_chars=20000).deidentify
CACHE = HERE / "results" / "reader_cache"; CACHE.mkdir(parents=True, exist_ok=True)
A = read_corpus(T, CACHE / "jev_mtsamples.jsonl", deidentify=deid, max_usd=a.max_usd)
ok = sum(x is not None for x in A)
print(f"answered {ok}/{len(T)}")
if ok == len(T) and len(T) >= 4999:
    np.save(CACHE / "jev_surgical.npy", np.vstack([to_features(x) for x in A]))
    (CACHE / "jev_surgical_columns.json").write_text(json.dumps(feature_names()))
    cost = sum((json.loads(l).get("cost") or 0) for l in open(CACHE / "jev_mtsamples.jsonl"))
    print(f"total cost USD {cost:.4f}")
