"""Time-boxed OpenMed NER on the Mac sandbox (numpy, no torch). Processes testbed rows in REVERSE order so it can
run alongside the cloud worker (which goes forward); outputs are merged by row id. Usage: run_ner_vm.py KEY SECONDS"""
import csv, glob, json, os, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from horos.represent.numpy_ner import NumpyDistilBertNER
key, budget = sys.argv[1], float(sys.argv[2]); t0 = time.time()
HUB = Path.home() / "mnt/hub"
name = {"disease": "DiseaseDetect-TinyMed-135M", "pharma": "PharmaDetect-TinyMed-66M", "anatomy": "AnatomyDetect-TinyMed-65M"}[key]
snap = glob.glob(str(HUB / f"models--OpenMed--OpenMed-NER-{name}/snapshots/*"))[0]
csv.field_size_limit(10 ** 8)
T = [r["transcription"] or "" for r in csv.DictReader(open(Path.home() / "mnt/PycharmProjects/_claude_transfer/llm_run/mtsamples.csv", newline=""))]
here = Path(__file__).parent; S = json.load(open(here / "results/risk_rows.json"))
out = here / "results/reader_cache" / f"ner_{key}_vm.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
skip = set()
for f in (out, here / "results/reader_cache" / f"ner_{key}_cloud_done.json"):
    if f.exists():
        skip |= ({json.loads(l)["i"] for l in open(f)} if f.suffix == ".jsonl" else set(json.load(open(f))))
m = NumpyDistilBertNER(snap); k = 0
with open(out, "a") as fh:
    for i in reversed(S):
        if i in skip:
            continue
        if time.time() - t0 > budget:
            break
        fh.write(json.dumps({"i": i, "v": m.entities_long(T[i])}) + "\n"); fh.flush(); k += 1
print(key, "wrote", k, "total", len(skip) + k)
