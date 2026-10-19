"""The ten known-noise designs (signal/noise as in Peras-RTX) with HARD notes and an LLM reader.
Gold Bayes risks are exact (gold.pointwise); notes fully determine the text concepts."""
import json, sys, numpy as np
from pathlib import Path
import datasets as D, notes
LAB = json.load(open(Path(__file__).with_name("llm_bank.json")))

def llm_read(cs):
    o = [None, None, None]
    for s in cs:
        l = LAB[s]
        for j, f in enumerate(("complexity", "prior", "access_risk")):
            if l[f] is not None: o[j] = l[f]
    return [(-1 if v is None else v) for v in o]

def make(design, n, seed):
    d = D.make(design, n, seed)
    rng = np.random.default_rng(10_000 + seed * 13 + design)
    c, r, k = d["concepts"].T
    nl = [notes.note(rng, int(c[i]), int(r[i]), int(k[i])) for i in range(n)]
    d["notes"] = [t for t, _ in nl]
    d["llm"] = np.array([llm_read(cs) for _, cs in nl], float)
    return d
