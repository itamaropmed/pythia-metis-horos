"""Run one Jev question set over testbed rows with the global budget ledger.  Usage:
   agent_jev.py QSET {all|fitprobe:N}     questions are read from results/reader_cache/jev_qsets.json"""
import json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
from metis.agent.reader_search import JevLedger
from horos.represent.jev_reader import ask
from testbed_v2 import load
CAP = float(os.environ.get("JEV_CAP_USD", "1.0"))
qset, scope = sys.argv[1], sys.argv[2]
Q = json.loads((HERE / "results/reader_cache/jev_qsets.json").read_text())[qset]
d = load(); S, split = d["S"], d["split"]
if scope == "all":
    idx = list(range(d["n"]))
else:
    n = int(scope.split(":")[1]); fit_idx = np.flatnonzero(split == "fit")
    idx = list(np.random.default_rng(5).choice(fit_idx, min(n, len(fit_idx)), replace=False))
texts = {int(S[k]): d["T"][k] for k in idx}
L = JevLedger(HERE / "results/reader_cache/jev_agent.jsonl", CAP)
print(qset, scope, "spent before", round(L.spent(), 4), "->", L.run(texts, qset, Q, lambda t, q: ask(t, q)), flush=True)
