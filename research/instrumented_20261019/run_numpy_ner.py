"""Run OpenMed NER (torch-free numpy inference) on the testbed rows; resumable jsonl per model."""
import json, sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parents[1]))
from horos.represent.numpy_ner import NumpyDistilBertNER
key = sys.argv[1]; T = pd.read_csv("/home/claude/data/mtsamples.csv").fillna("")["transcription"].astype(str).tolist()
S = json.load(open(HERE / "results/risk_rows.json"))
out = HERE / "results/reader_cache" / f"ner_{key}.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
done = {json.loads(l)["i"] for l in open(out)} if out.exists() else set()
m = NumpyDistilBertNER(f"/home/claude/models/openmed_ner_{key}")
with open(out, "a") as fh:
    for k, i in enumerate(i for i in S if i not in done):
        fh.write(json.dumps({"i": i, "v": m.entities_long(T[i])}) + "\n"); fh.flush()
        if k % 200 == 0: print(key, len(done) + k, flush=True)
print(key, "done", flush=True)
