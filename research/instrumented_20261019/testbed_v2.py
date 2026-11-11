"""Testbed v2: real MTSamples notes, planted surgical-duration outcome driven by SEVERAL kinds of note content,
so different readers own different parts of the signal (known truth for every claim).

Ground-truth concepts (regex on the note, fixed before any reader is run):
  anes_general  'general' in the ANESTHESIA section          -> LLM / Jev questions
  ebl           estimated blood loss number (ml, capped 500) -> LLM / Jev questions
  diabetes, hypertension, obesity, malignancy                -> OpenMed DiseaseDetect
  anticoag      warfarin / heparin / apixaban / clopidogrel  -> OpenMed PharmaDetect
duration = 90 + table(cell) + 25 anes + .08 ebl + 15 diabetes + 10 htn + 15 obesity + 20 malignancy + 20 anticoag + N(0, 20^2)
Deployed model: table-only cell means fitted on FIT.  Risk target: W30 miss.  Exact P(miss | table, note) is known.
Split (fixed, outcome-free): FIT 40% / CAL 30% / TEST 30% of the 3,423 testbed rows (results/risk_rows.json).
"""
from __future__ import annotations

import json, re, sys
from pathlib import Path

import numpy as np, pandas as pd
from scipy.stats import norm

HERE = Path(__file__).parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research/note_atlas_20261018"))
from atlas import segment  # noqa

CONCEPTS = {
    "diabetes": r"\bdiabet|\bdm2\b|\bniddm\b|\biddm\b",
    "hypertension": r"\bhypertension\b|\bhtn\b",
    "obesity": r"\bobes|\bmorbid(ly)? obes",
    "malignancy": r"carcinoma|malignan|\bcancer\b|metasta",
    "anticoag": r"warfarin|coumadin|heparin|lovenox|enoxaparin|eliquis|apixaban|xarelto|rivaroxaban|plavix|clopidogrel",
}
EFFECT = {"anes_general": 25., "diabetes": 15., "hypertension": 10., "obesity": 15., "malignancy": 20., "anticoag": 20.}


def _sec(t, name):
    return " ".join(b for h, b in segment(t) if name in h).lower()


def load(data_csv="/home/claude/data/mtsamples.csv", seed=20261023):
    df = pd.read_csv(data_csv).fillna("")
    T_all = df["transcription"].astype(str).tolist()
    S = np.array(json.load(open(HERE / "results/risk_rows.json")))
    T = [T_all[i] for i in S]; n = len(S)
    spec = df["medical_specialty"].astype(str).str.strip().to_numpy()[S]
    top = pd.Series(spec).value_counts().index[:12]
    cell = np.where(np.isin(spec, top), spec, "other")
    C = {"anes_general": np.array([("general" in _sec(t, "ANESTHESIA")) for t in T], float)}
    ebl = []
    for t in T:
        m = re.search(r"(\d+)\s*(?:ml|cc)", _sec(t, "BLOOD LOSS")); ebl.append(min(float(m.group(1)), 500) if m else 0.)
    C["ebl"] = np.array(ebl)
    low = [t.lower() for t in T]
    for k, rx in CONCEPTS.items():
        C[k] = np.array([bool(re.search(rx, t)) for t in low], float)
    rng = np.random.default_rng(seed)
    tab = pd.Series(cell).map(pd.Series(rng.normal(0, 15, len(np.unique(cell))), index=np.unique(cell))).to_numpy()
    mu = 90 + tab + .08 * C["ebl"] + sum(EFFECT[k] * C[k] for k in EFFECT)
    y = mu + rng.normal(0, 20, n)
    u = np.random.default_rng(seed + 1).random(n)                     # outcome-free split
    split = np.where(u < .4, "fit", np.where(u < .7, "cal", "test"))
    fit = split == "fit"
    f = pd.Series(cell).map(pd.Series(y[fit]).groupby(cell[fit]).mean()).fillna(y[fit].mean()).to_numpy()
    miss = (np.abs(y - f) > 30).astype(int)
    p_miss = norm.cdf((-30 - (mu - f)) / 20) + 1 - norm.cdf((30 - (mu - f)) / 20)
    return dict(S=S, T=T, cell=cell, C=C, y=y, f=f, miss=miss, p_miss=p_miss, split=split, n=n)


if __name__ == "__main__":
    d = load()
    print("n", d["n"], {k: round(float(v.mean() if k != "ebl" else (v >= 100).mean()), 3) for k, v in d["C"].items()},
          "miss rate", d["miss"].mean().round(3), "split", pd.Series(d["split"]).value_counts().to_dict())
