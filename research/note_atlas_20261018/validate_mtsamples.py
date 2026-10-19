"""Validate the Note Information Atlas on REAL MTSamples notes with a PLANTED duration-like outcome
(known truth): table = note type x age band x sex; text effects planted in specific sections / token classes."""
import sys, os, json, re, runpy, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from atlas import run_atlas, segment
R = runpy.run_path(os.environ.get("AFI_REAL", "/home/claude/repo/research/peras_v5/afi_20261017/run_afi_real.py"), run_name="lib")
T = [t if isinstance(t, str) else "" for t in R["T"].tolist()]; n = len(T)
cell = R["LADDER"]["T2_type_age_sex"]
rng = np.random.default_rng(20261018)

def sec(t, name):
    return " ".join(b for h, b in segment(t) if name in h).lower()

f_anes = np.array([("general" in sec(t, "ANESTHESIA")) for t in T], float)                 # section-specific term
ebl = []
for t in T:
    m = re.search(r"(\d+)\s*(?:ml|cc)", sec(t, "BLOOD LOSS")); ebl.append(min(float(m.group(1)), 500) if m else 0.0)
ebl = np.array(ebl)                                                                          # number in a section
comp = np.array([(lambda s: bool(s) and not re.search(r"\b(no|none|without)\b", s))(sec(t, "COMPLICATION")) for t in T], float)  # negation-sensitive
text_effect = 25 * f_anes + .08 * ebl + 30 * comp
table_effect = pd.Series(cell).map(pd.Series(rng.normal(0, 15, cell.max() + 1))).to_numpy()
y = 90 + table_effect + text_effect + rng.normal(0, 20, n)
# table model: cross-fitted cell means (patient = note here)
from sklearn.model_selection import KFold
fx = np.zeros(n)
for tr, va in KFold(5, shuffle=True, random_state=0).split(T):
    mu = pd.Series(y[tr]).groupby(cell[tr]).mean(); fx[va] = pd.Series(cell[va]).map(mu).fillna(y[tr].mean()).to_numpy()
r = y - fx
truth = text_effect - pd.Series(text_effect).groupby(cell).transform("mean").to_numpy()
print("planted prevalence: anes-general %.3f, EBL>0 %.3f, complication-present %.3f; true text var beyond table %.1f"
      % (f_anes.mean(), (ebl > 0).mean(), comp.mean(), truth.var()), flush=True)
t0 = time.time()
res = run_atlas(T, np.arange(n).astype(str), r, strata=R["LADDER"]["T1_note_type"], truth=truth,
                min_patients_term=30, min_patients_header=60, verbose=True)
res["validation"] = dict(true_I_beyond_table=float(truth.var()), planted=["ANESTHESIA:general (+25)", "BLOOD LOSS number (+0.08/ml)",
                         "COMPLICATIONS present & not negated (+30)"], sec=time.time() - t0)
# negative control
rp = r[rng.permutation(n)]
res["negative_control"] = run_atlas(T, np.arange(n).astype(str), rp, lenses="B")["B_global"]
res.pop("_model", None); res["negative_control"].pop("_model", None) if isinstance(res.get("negative_control"), dict) else None
json.dump(res, open(os.path.join(os.path.dirname(__file__), "mtsamples_validation.json"), "w"), indent=1, default=float)
print(json.dumps({k: res[k] for k in ("B_global", "D_sections", "E_token_classes", "F_melting", "G_beam", "G_annealing", "H_diffusion", "J_topology", "validation")}, indent=1, default=float)[:6000])
