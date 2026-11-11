"""Certified duration information from the notes (testbed v2, known truth), with the taming quantile and the
outcome cap chosen on FIT only (out-of-fold scores), then certified once on CAL at alpha / J."""
import json, sys
from pathlib import Path
import numpy as np
__file__ = str(Path(__file__).resolve())
HERE = Path(__file__).parent
exec(open(HERE / "cre_eval2.py").read().split("J = len(R) + 1; preds = {}")[0])
from pythia.core.instrumented import eb_scalar_bound, tune_on_fit, loo_centre_scalar
keys = list(R) + ["STACK(all real readers)"]; J = len(keys); res = {}
idx = np.flatnonzero(fit); fo = np.random.default_rng(1).permutation(len(idx)) % 2
for k in keys:
    ks = real if k.startswith("STACK") else [k]
    h_all = stacked_pred(ks, rr)                                   # FIT-trained, used for CAL
    h_oof = np.zeros(len(idx))                                     # FIT out-of-fold scores for tuning
    for q in (0, 1):
        m = np.zeros(len(S), bool); m[idx[fo != q]] = True
        preds_q = [ridge_dir(R[kk], rr, m) for kk in ks]
        h_oof[fo == q] = (np.mean(preds_q, 0) if len(ks) > 1 else preds_q[0])[idx[fo == q]]
    qq, cap = tune_on_fit(h_oof, (y - f)[idx], cell[idx], .05 / J)
    lo, hi = np.quantile(h_all[cal], [qq, 1 - qq]); hc = loo_centre_scalar(np.clip(h_all[cal], lo, hi), cell[cal])
    # truth for the capped residual on CAL (Monte Carlo over the planted noise)
    eps = np.random.default_rng(3).normal(0, 20, (3000, 1)); m_cap = np.clip(mu - f + eps, -cap, cap).mean(0)
    tv = float(np.var(_centre(m_cap[:, None], cell)[cal]))
    b = eb_scalar_bound(hc, (y - f)[cal], cap, .05 / J, groups=cell[cal], entity_ids=S[cal])
    res[k] = {"certified_min2": b, "truth_min2": tv, "share": b / tv, "tamed_q": qq, "cap_min": cap}
    print(f"{k:32s} certified {b:7.1f} min^2 of {tv:6.1f} ({b / tv:.2f})  [q={qq}, cap={cap:.0f}]  bound>truth: {b > tv}")
(HERE / "results/cre_bound_final.json").write_text(json.dumps(res, indent=1))
