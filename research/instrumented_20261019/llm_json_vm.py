"""One LiteLLM chat call on the Mac: PROMPT_FILE (aggregates only, no note text) -> JSON reply in OUT_FILE.
Spend is appended to results/reader_cache/agent_spend.jsonl.  Usage: llm_json_vm.py PROMPT_FILE OUT_FILE MODEL"""
import json, re, sys, urllib.request
from pathlib import Path
pf, of, model = sys.argv[1], sys.argv[2], sys.argv[3]
key = (Path.home() / "mnt/PycharmProjects/.litellm_key").read_text().strip()
body = {"model": model, "temperature": 0, "max_tokens": 6000,
        "messages": [{"role": "system", "content": "You are a clinical informatics and surgical outcomes expert. Output JSON only."},
                     {"role": "user", "content": Path(pf).read_text()}]}
req = urllib.request.Request("https://litellm.prod.int.opmed.ai/chat/completions", data=json.dumps(body).encode(),
                             headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=170).read()); u = r.get("usage", {})
txt = r["choices"][0]["message"]["content"] or ""; m = re.search(r"\{.*\}", txt, re.S)
Path(of).write_text(m.group(0) if m else "{}")
cost = u.get("prompt_tokens", 0) * 1e-7 + u.get("completion_tokens", 0) * 5e-7
with open(Path(__file__).parent / "results/reader_cache/agent_spend.jsonl", "a") as fh:
    fh.write(json.dumps({"prompt": Path(pf).name, "model": model, "usage": u, "cost_est": cost}) + "\n")
print("finish", r["choices"][0].get("finish_reason"), "tokens", u.get("total_tokens"), "cost_est", round(cost, 5))
