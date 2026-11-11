"""SCORE step (FIT probe rows only): does the proposed question set add certifiable risk information beyond the
current readers?  Cross-fitted plug-in signal C^2/E|H|^2 with vs without the new reader (no CAL/TEST rows)."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
from testbed_v2 import load
import reader_features as rf
from metis.agent.reader_search import fit_gain
qset = sys.argv[1]; base_keys = sys.argv[2].split(",")
d = load(); S = d["S"]; pos = {int(s): k for k, s in enumerate(S)}
fit_rows = np.flatnonzero(d["split"] == "fit")
probe = [i for i in json.load(open(HERE / (sys.argv[3] if len(sys.argv) > 3 else "results/agent/probe_rows.json")))]
Xnew = rf.llm_qa(np.array(probe), qset); rows = np.array([pos[i] for i in probe])
lib = {"llm": rf.llm_extract(S), "disease": rf.ner_bag(S, "disease", fit_rows=fit_rows),
       "pharma": rf.ner_bag(S, "pharma", fit_rows=fit_rows),
       "anatomy": rf.ner_bag(S, "anatomy", fit_rows=fit_rows)}
Xbase = np.column_stack([lib[k][rows] for k in base_keys])
res = {"qset": qset, "probe_rows": len(rows), "base_readers": base_keys}
for seed in range(5):
    r = fit_gain(Xbase, Xnew, d["miss"][rows], d["cell"][rows], entity_ids=S[rows], seed=seed)
    for k, v in r.items():
        res.setdefault(k, []).append(v)
res["mean_gain"] = float(np.mean(res["gain"])); res["gain_positive_seeds"] = int(np.sum(np.array(res["gain"]) > 0))
res["accept"] = res["mean_gain"] > 0 and res["gain_positive_seeds"] >= 4
(HERE / f"results/agent/{qset}_score{'_' + Path(sys.argv[3]).stem if len(sys.argv) > 3 else ''}.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res))
