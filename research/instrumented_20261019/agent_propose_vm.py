"""PROPOSE step on the Mac sandbox: LiteLLM turns residual terms into new Jev questions (no outcomes, no note text).
Usage: agent_propose_vm.py results/agent/round_R_terms.json QSET_NEW MODEL"""
import json, os, sys, urllib.request
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from metis.agent.reader_search import proposal_prompt, validate_questions
terms_f, qset, model = sys.argv[1], sys.argv[2], sys.argv[3]
here = Path(__file__).parent; qsets_f = here / "results/reader_cache/jev_qsets.json"
qsets = json.loads(qsets_f.read_text()) if qsets_f.exists() else {}
existing = {k: v for qs in qsets.values() for k, v in qs.items()}
terms = json.loads(Path(terms_f).read_text())["terms"]
key = (Path.home() / "mnt/PycharmProjects/.litellm_key").read_text().strip()
body = {"model": model, "temperature": 0, "max_tokens": 2500,
        "messages": [{"role": "system", "content": "You are a clinical informatics expert. Output JSON only."},
                     {"role": "user", "content": proposal_prompt(terms, existing)}]}
req = urllib.request.Request("https://litellm.prod.int.opmed.ai/chat/completions", data=json.dumps(body).encode(),
                             headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=170).read())
txt = r["choices"][0]["message"]["content"] or ""
import re
m = re.search(r"\{.*\}", txt, re.S)
Q = validate_questions(json.loads(m.group(0)) if m else None)
qsets[qset] = Q; qsets_f.write_text(json.dumps(qsets, indent=1))
(here / f"results/agent/{qset}_proposal.json").write_text(json.dumps({"model": model, "usage": r.get("usage"), "questions": Q}, indent=1))
print(json.dumps(Q, indent=1)[:3000]); print("usage", r.get("usage", {}).get("total_tokens"))
