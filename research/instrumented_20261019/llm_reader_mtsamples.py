#!/usr/bin/env python3
"""Run LLM *readers* over PUBLIC MTSamples notes through the opmed LiteLLM proxy (run this on your Mac / VPN).

Never use this script on Mayo or any PHI.  Output: readers.jsonl (note index, reader id, parsed JSON) --
no note text is written.

  export LITELLM_API_KEY=...            # or put the key in ~/PycharmProjects/.litellm_key (never commit it)
  python3 llm_reader_mtsamples.py --csv mtsamples.csv --models MODEL_A,MODEL_B --limit 2400 --workers 8
  python3 llm_reader_mtsamples.py --list-models

Each (model, prompt) pair is one reader.  Prompts are declared here, before any outcome is computed, so the
portfolio size J = #models x #prompts is fixed in advance (Pythia P2).
"""
from __future__ import annotations

import argparse, concurrent.futures as cf, json, os, re, sys, time, urllib.request
from pathlib import Path

BASE = os.environ.get("LITELLM_BASE", "https://litellm.prod.int.opmed.ai")

SCHEMA = ('{"anesthesia": "general|regional|local|mac|none|unknown", "blood_loss_ml": number or null, '
          '"complications": "yes|no|unknown", "emergency": "yes|no|unknown", "complexity_1to5": integer}')
PROMPTS = {
    "extract_v1": "Read the clinical note and fill this JSON exactly; answer only JSON.\nSchema: " + SCHEMA,
    "extract_v2": ("You are a surgical data abstractor. From the operative/clinical note below, record the anesthesia "
                   "type, the estimated blood loss in ml, whether any complication occurred (negated mentions such as "
                   "'no complications' mean no), whether the case was emergent, and overall procedural complexity "
                   "(1 = trivial, 5 = very complex). Return only JSON with schema: " + SCHEMA),
    "judge_v1": ('Read the note. Estimate the probability (0-100) that this procedure ran LONGER than a typical case of '
                 'the same specialty, using only what the note says. Answer only JSON: {"p_long": integer}'),
}


def api_key() -> str:
    k = os.environ.get("LITELLM_API_KEY")
    if not k:
        p = Path.home() / "PycharmProjects" / ".litellm_key"
        if p.exists():
            k = p.read_text().strip()
    if not k:
        sys.exit("set LITELLM_API_KEY or create ~/PycharmProjects/.litellm_key")
    return k


def call(path, payload=None, timeout=120):
    req = urllib.request.Request(BASE + path, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def parse(txt: str):
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None


MAX_TOKENS = 600          # reasoning models spend hidden tokens first; 200 truncated ~40% of answers
PRICE = {}                # model -> (usd per input token, usd per output token), from /model/info
DEID = None   # set by --deidentify-with: OpenMed PII model (local path or HF id); runs locally before any call


def one(model, pid, idx, note):
    if DEID is not None:
        note = DEID.deidentify(note)
    msg = [{"role": "system", "content": "You extract facts from clinical notes. Output JSON only."},
           {"role": "user", "content": PROMPTS[pid] + "\n\nNOTE:\n" + note[:6000]}]
    for attempt in range(4):
        try:
            r = call("/chat/completions", {"model": model, "messages": msg, "temperature": 0, "max_tokens": MAX_TOKENS})
            out = r["choices"][0]["message"]["content"]
            u = r.get("usage", {})
            return {"i": idx, "reader": f"{model}|{pid}", "json": parse(out), "tokens": u.get("total_tokens"),
                    "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"), "ok": True}
        except Exception as e:  # noqa
            time.sleep(2 ** attempt)
            err = str(e)[:200]
    return {"i": idx, "reader": f"{model}|{pid}", "json": None, "ok": False, "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="mtsamples.csv"); ap.add_argument("--models", default="")
    ap.add_argument("--prompts", default=",".join(PROMPTS)); ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--out", default="readers.jsonl")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--rows", default="", help="JSON list of note indices to read (e.g. results/risk_rows.json)")
    ap.add_argument("--max-calls", type=int, default=4000, help="hard cap on API calls in this run")
    ap.add_argument("--max-usd", type=float, default=1.0, help="stop when estimated spend (all runs in --out) reaches this")
    ap.add_argument("--deidentify-with", default="", help="OpenMed PII model: de-identify locally before sending")
    a = ap.parse_args()
    global DEID
    if a.deidentify_with:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from horos.represent.openmed_readers import OpenMedPII
        DEID = OpenMedPII(a.deidentify_with, max_chars=20000)
    if a.list_models:
        print(json.dumps([m["id"] for m in call("/models")["data"]], indent=1)); return
    import csv
    csv.field_size_limit(10 ** 8)
    notes = [r.get("transcription") or "" for r in csv.DictReader(open(a.csv, newline=""))][: a.limit]
    done = set()
    if Path(a.out).exists():
        for line in open(a.out):
            d = json.loads(line)
            if d.get("ok") and d.get("json") is not None:
                done.add((d["i"], d["reader"]))
    rows = set(json.load(open(a.rows))) if a.rows else None
    jobs = [(m, p, i, t) for m in a.models.split(",") if m for p in a.prompts.split(",") for i, t in enumerate(notes)
            if (i, f"{m}|{p}") not in done and (rows is None or i in rows)][: a.max_calls]
    print(f"{len(notes)} notes, {len(jobs)} calls to run", flush=True)
    try:
        for m in call("/model/info").get("data", []):
            mi = m.get("model_info") or {}
            PRICE[m["model_name"]] = (mi.get("input_cost_per_token") or 0, mi.get("output_cost_per_token") or 0)
    except Exception:
        pass

    def cost(d):
        pi, po = PRICE.get(d["reader"].split("|")[0], (0, 0))
        return (d.get("prompt_tokens") or d.get("tokens") or 0) * pi + (d.get("completion_tokens") or 0) * po
    spent = sum(cost(json.loads(l)) for l in open(a.out)) if Path(a.out).exists() else 0.0
    print(f"estimated spend so far {spent:.4f} USD (cap {a.max_usd})", flush=True)
    if spent >= a.max_usd:
        print("budget cap reached; nothing to do"); return
    with open(a.out, "a") as fh, cf.ThreadPoolExecutor(a.workers) as ex:
        for k, res in enumerate(ex.map(lambda j: one(*j), jobs)):
            fh.write(json.dumps(res) + "\n"); fh.flush(); spent += cost(res)
            if spent >= a.max_usd:
                fh.flush(); print(f"budget cap reached at {spent:.4f} USD; stopping", flush=True)
                ex.shutdown(cancel_futures=True); break
            if k % 200 == 0:
                fh.flush(); print(k, res["reader"], res["ok"], flush=True)


if __name__ == "__main__":
    main()
