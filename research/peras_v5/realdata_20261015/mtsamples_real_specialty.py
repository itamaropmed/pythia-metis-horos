"""REAL labels: specialty (40 classes) given the exact transcription text, MTSamples.  Cells = exact (whitespace/
case-normalised) duplicate notes; a note listed under several specialties is a replicate draw of the label.
Development/certification split by cell (random halves, 20 splits).  Compares the accepted min-count / hard-label
route with the pair (Thm A) and linear-minorant (Thm C) certificates.  Also K collapsed to the top-10 specialties
+ 'other' (K=11).  No gold: the finite-corpus floor 0.525 is a reference for the population floor's order."""
import sys, os, json, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from peras.core.replicate_pair_certificates import (cells_from_keys, cell_counts, pair_certificate,
    mincount_certificate, fit_g_minorant, choose_rule, linear_certificate, within_cell_independence_audit)

df = pd.read_csv(__import__("os").environ.get("PERAS_DATA", "/home/claude/data") + "/mtsamples.csv").dropna(subset=["transcription"]).reset_index(drop=True)
key = df.transcription.str.lower().str.replace(r"\s+", " ", regex=True).str.strip().map(lambda s: hashlib.md5(s.encode()).hexdigest())
cell = cells_from_keys(key)
spec = df.medical_specialty.str.strip()
res = []
for Kname in ("all", "top10"):
    lab = spec if Kname == "all" else spec.where(spec.isin(spec.value_counts().index[:10]), "other")
    y = cells_from_keys(lab); K = int(y.max() + 1)
    for s in range(20):
        r = np.random.default_rng(s); dev = r.random(cell.max() + 1) < .5
        di = dev[cell]; ci = ~di
        cnt_d = cell_counts(cells_from_keys(cell[di]), y[di], K)
        coefs = {d: fit_g_minorant(cnt_d, K, d) for d in (1, 2)}; rule = choose_rule(cnt_d, K, coefs, .025)
        cc, yc = cell[ci], y[ci]
        o = dict(K=K, labels=Kname, split=s, n=int(ci.sum()),
                 mincount=mincount_certificate(cc, yc, K, .05)["bound"],
                 plug_in=mincount_certificate(cc, yc, K, .05)["plug_in"],
                 pair=pair_certificate(cc, yc, K, .025)["bound"],
                 linear=linear_certificate(cc, yc, K, rule, coefs, .025)["bound"], rule=rule)
        aud = within_cell_independence_audit(cc, yc, K); o["audit_p"] = aud["p"]; o["mean_pair_agreement"] = aud["mean_agree"]
        o["peras_unaudited"] = max(o["pair"], o["linear"])
        o["peras"] = o["peras_unaudited"] if not aud["reject"] else o["mincount"]; res.append(o)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()}), flush=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mtsamples_real_specialty.json"), "w"), indent=1)
