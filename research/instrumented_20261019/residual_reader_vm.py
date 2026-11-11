"""Table-conditioned, embedding-free LLM *residual reader* (run on the Mac; PUBLIC MTSamples only, never PHI).

The reader is told what the table already knows about the case (specialty cell, the table model's expected duration
and the typical spread) and is asked only for what the NOTE adds beyond that.  It never sees outcomes.

Output per note (one call):
  shift_min      expected minutes longer (+) / shorter (-) than the table expectation  -> estimate of E[r | X, T]
  p_long30       P(case runs > 30 min longer than the table expectation)
  p_short30      P(case runs > 30 min shorter)
  drivers        up to 10 typed, signed, atomic propositions stated in the note (the symbolic sample space):
                 {"fact": canonical short fact, "type": ..., "dir": "+"|"-"|"0", "min": signed minutes}
Usage: residual_reader_vm.py CONTEXT_JSON MODEL SECONDS MAX_USD [LIMIT] [EFFORT]
"""
import concurrent.futures as cf, csv, json, re, sys, time, urllib.request
from pathlib import Path

ctx_f, model, budget, max_usd = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
limit = int(sys.argv[5]) if len(sys.argv) > 5 else 10 ** 9
effort = sys.argv[6] if len(sys.argv) > 6 else ""
t0 = time.time(); here = Path(__file__).parent
csv.field_size_limit(10 ** 8)
T = [r["transcription"] or "" for r in csv.DictReader(open(Path.home() / "mnt/PycharmProjects/_claude_transfer/llm_run/mtsamples.csv", newline=""))]
ctx = json.load(open(ctx_f))[:limit]
out = here / "results/reader_cache/residual_reader.jsonl"
key = (Path.home() / "mnt/PycharmProjects/.litellm_key").read_text().strip()
PRICE = (1e-7, 5e-7)          # gpt-6-luna list price per token (input, output)
TYPES = "comorbidity|medication|anesthesia|blood_loss|procedure_scope|anatomy|finding|complication|urgency|patient|other"
PROMPT = (
    "A scheduling model predicts surgical case duration from the table only. TABLE CONTEXT for this case: "
    "specialty group = {cell}; the table model expects {f:.0f} minutes; cases in this group typically deviate by "
    "about +/-{sd:.0f} minutes from the table expectation.\n"
    "Read the clinical note and judge ONLY what the note adds beyond the table: facts stated in the note that would "
    "make THIS case run longer or shorter than the table expectation (e.g. anesthesia type, blood loss, comorbidities, "
    "anticoagulants, extent of the procedure, complications, urgency). Do not restate the specialty.\n"
    "Return ONLY JSON: {{\"shift_min\": number, \"p_long30\": number 0-1, \"p_short30\": number 0-1, "
    "\"drivers\": [{{\"fact\": \"<short canonical fact, lowercase, e.g. 'general anesthesia', 'ebl 300 ml', "
    "'type 2 diabetes', 'on warfarin'>\", \"type\": \"" + TYPES + "\", \"dir\": \"+|-|0\", \"min\": number}}]}}\n"
    "Give at most 10 drivers, only facts stated in the note (negated facts are not drivers).")

done = set(); spent = 0.0
if out.exists():
    for l in open(out):
        d = json.loads(l); spent += d.get("cost", 0)
        if d.get("ok"):
            done.add(d["i"])


def one(c):
    body = {"model": model, "temperature": 0, "max_tokens": 1200,
            "messages": [{"role": "system", "content": "You are a surgical outcomes analyst. Output JSON only."},
                         {"role": "user", "content": PROMPT.format(**c) + "\n\nNOTE:\n" + T[c["i"]][:5000]}]}
    if effort:
        body["reasoning_effort"] = effort
    for a in range(3):
        try:
            req = urllib.request.Request("https://litellm.prod.int.opmed.ai/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=90).read()); u = r.get("usage", {})
            cost = u.get("prompt_tokens", 0) * PRICE[0] + u.get("completion_tokens", 0) * PRICE[1]
            m = re.search(r"\{.*\}", r["choices"][0]["message"]["content"] or "", re.S)
            ans = json.loads(m.group(0)) if m else None
            return {"i": c["i"], "ok": ans is not None, "ans": ans, "cost": cost,
                    "pt": u.get("prompt_tokens"), "ct": u.get("completion_tokens")}
        except Exception as e:  # noqa
            err = str(e)[:120]; time.sleep(2 * (a + 1))
    return {"i": c["i"], "ok": False, "error": err, "cost": 0}


todo = [c for c in ctx if c["i"] not in done]; n = 0
with open(out, "a") as fh, cf.ThreadPoolExecutor(10) as ex:
    futs = {}; it = iter(todo)
    while True:
        while len(futs) < 10 and time.time() - t0 < budget and spent < max_usd:
            try:
                c = next(it)
            except StopIteration:
                break
            futs[ex.submit(one, c)] = c
        if not futs:
            break
        f = next(cf.as_completed(futs)); futs.pop(f); d = f.result()
        fh.write(json.dumps(d) + "\n"); fh.flush(); spent += d["cost"]; n += d["ok"]
print("answered", n, "remaining", len(todo) - n, "LiteLLM est. spend (residual reader)", round(spent, 4))
