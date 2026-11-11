"""DIAGNOSE step (FIT rows only): out-of-fold residuals of the current risk model -> terms where error information
is still unread.  Writes results/agent/round_<R>_terms.json (aggregate term statistics only)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
from testbed_v2 import load
import reader_features as rf
from metis.agent.reader_search import residual_terms
from horos.represent.score_learning import variational_scores_cv as vs, oof_scores, stack_weights, stack_scores
rnd = sys.argv[1]; readers = sys.argv[2].split(",")
d = load(); S, cell, miss, split = d["S"], d["cell"], d["miss"], d["split"]; fit = np.flatnonzero(split == "fit")
lib = {"llm": lambda: rf.llm_extract(S), "disease": lambda: rf.ner_bag(S, "disease", fit_rows=fit), "pharma": lambda: rf.ner_bag(S, "pharma", fit_rows=fit),
       "anatomy": lambda: rf.ner_bag(S, "anatomy", fit_rows=fit), "tfidf": lambda: rf.tfidf(d["T"], fit_rows=fit)}
for q in ("q1", "q2", "q3"):
    lib[q] = lambda q=q: rf.llm_qa(S, q)
Xs = [lib[k]()[fit] for k in readers]; m, g = miss[fit], cell[fit]
base = pd.Series(g).map(pd.Series(m).groupby(g).mean()).to_numpy()       # in-sample cell rate (FIT only)
# The synthetic testbed rows are generated as independent sampling units.
oof = [oof_scores(vs, X, m, g, 2, entity_ids=np.arange(len(m))) for X in Xs]; w = stack_weights(oof, m, g) if len(Xs) > 1 else np.array([1.])
h = stack_scores(oof, w)[:, 1]; a = float((h * (m - base)).sum() / max((h ** 2).sum(), 1e-12))
resid = m - np.clip(base + a * h, 0, 1)
terms = residual_terms([d["T"][i] for i in fit], resid, k=30, min_df=25)
(HERE / f"results/agent/round_{rnd}_terms.json").write_text(json.dumps({"round": rnd, "base_readers": readers,
                                                                          "terms": terms}, indent=1))
print(terms)
