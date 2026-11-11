"""Embed testbed notes through the opmed LiteLLM (text-embedding-3-small); public MTSamples only.
Variants: note   = the note alone
          joint  = table context serialised in front of the note (specialty + the table model's prediction):
                   one embedding of (table, text) together -- a table-conditioned text representation.
Usage: embed_vm.py VARIANT SECONDS        output: results/reader_cache/emb_<variant>.npz (resumable .jsonl parts)"""
import csv, json, sys, time, urllib.request
from pathlib import Path
t0 = time.time(); variant, budget = sys.argv[1], float(sys.argv[2]); here = Path(__file__).parent
ctx = {r["i"]: r for r in json.load(open(here / "results/agent/table_context.json"))}
csv.field_size_limit(10 ** 8)
T = [r["transcription"] or "" for r in csv.DictReader(open(Path.home() / "mnt/PycharmProjects/_claude_transfer/llm_run/mtsamples.csv", newline=""))]
key = (Path.home() / "mnt/PycharmProjects/.litellm_key").read_text().strip()
out = here / f"results/reader_cache/emb_{variant}.jsonl"; done = set()
if out.exists():
    done = {json.loads(l)["i"] for l in open(out)}
todo = [i for i in ctx if i not in done]; tok = 0


def text(i):
    note = (T[i] or " ")[:7000]
    if variant == "joint":
        c = ctx[i]
        return (f"Structured context: surgical specialty = {c['cell']}; table-based predicted case duration = {c['f']} "
                f"minutes.\nClinical note: {note}")
    return note


with open(out, "a") as fh:
    for k in range(0, len(todo), 48):
        if time.time() - t0 > budget:
            break
        b = todo[k:k + 48]
        body = {"model": "text-embedding-3-small", "input": [text(i) for i in b]}
        req = urllib.request.Request("https://litellm.prod.int.opmed.ai/embeddings", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=120).read()); tok += (r.get("usage") or {}).get("total_tokens", 0)
        for i, e in zip(b, r["data"]):
            fh.write(json.dumps({"i": i, "e": [round(x, 5) for x in e["embedding"]]}) + "\n")
        fh.flush()
print(variant, "done", len(done) + min(len(todo), k + 48 if todo else 0), "of", len(ctx), "tokens this run", tok,
      "est USD", round(tok * 2e-8, 4))
