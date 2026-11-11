"""ANSWER step on the Mac sandbox: the LiteLLM model answers a proposed question set in Jev's typed format
(noul -> P(true), choice -> option probabilities, score -> level probabilities).  Public MTSamples only.
Usage: agent_answer_vm.py QSET ROWS_JSON MODEL SECONDS [MAX_USD]"""
import concurrent.futures as cf, csv, json, re, sys, time, urllib.request
from pathlib import Path
qset, rows_f, model, budget = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
max_usd = float(sys.argv[5]) if len(sys.argv) > 5 else 0.6
t0 = time.time(); here = Path(__file__).parent
Q = json.loads((here / "results/reader_cache/jev_qsets.json").read_text())[qset]
csv.field_size_limit(10 ** 8)
T = [r["transcription"] or "" for r in csv.DictReader(open(Path.home() / "mnt/PycharmProjects/_claude_transfer/llm_run/mtsamples.csv", newline=""))]
rows = json.load(open(rows_f)); out = here / "results/reader_cache/llm_qa.jsonl"
key = (Path.home() / "mnt/PycharmProjects/.litellm_key").read_text().strip()
PRICE = (1e-7, 5e-7)          # gpt-6-luna list price per token (input, output) from /model/info
done = set(); spent = 0.0
if out.exists():
    for l in open(out):
        d = json.loads(l); spent += d.get("cost", 0)
        if d.get("qset") == qset and d.get("ok"):
            done.add(d["i"])
spec = {k: (q["type"], list(q["criteria"]) if q["type"] == "choice" else list(range(len(q["criteria"])))) for k, q in Q.items()}
ask = ("Answer each question about the clinical note. Return ONLY JSON mapping each question name to: for yes/no "
       "questions the probability (0-1) that the answer is yes; for choice/score questions an object giving a probability "
       "for every option key (summing to 1).\nQUESTIONS: " + json.dumps(Q))


def one(i):
    body = {"model": model, "temperature": 0, "max_tokens": 900,
            "messages": [{"role": "system", "content": "You read clinical notes and answer typed questions. Output JSON only."},
                         {"role": "user", "content": ask + "\n\nNOTE:\n" + T[i][:6000]}]}
    try:
        req = urllib.request.Request("https://litellm.prod.int.opmed.ai/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=60).read()); u = r.get("usage", {})
        cost = u.get("prompt_tokens", 0) * PRICE[0] + u.get("completion_tokens", 0) * PRICE[1]
        m = re.search(r"\{.*\}", r["choices"][0]["message"]["content"] or "", re.S)
        ans = json.loads(m.group(0)) if m else None
        return {"i": i, "qset": qset, "ok": ans is not None, "answers": ans, "cost": cost}
    except Exception as e:  # noqa
        return {"i": i, "qset": qset, "ok": False, "error": str(e)[:120], "cost": 0}


todo = [i for i in rows if i not in done]; n = 0
with open(out, "a") as fh, cf.ThreadPoolExecutor(8) as ex:
    futs = {}
    it = iter(todo)
    while True:
        while len(futs) < 8 and time.time() - t0 < budget and spent < max_usd:
            try:
                i = next(it)
            except StopIteration:
                break
            futs[ex.submit(one, i)] = i
        if not futs:
            break
        f = next(cf.as_completed(futs)); futs.pop(f); d = f.result()
        fh.write(json.dumps(d) + "\n"); fh.flush(); spent += d["cost"]; n += d["ok"]
print(qset, "answered", n, "remaining", len(todo) - n, "LiteLLM est. spend (all qsets)", round(spent, 4))
