"""Testbed v2: held-out (TEST) duration MAE in minutes, table only vs table + note readers (ridge on the
within-cell residual, FIT-trained; stack weights on FIT out-of-fold).  Floor: the true mean (noise only)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
exec(open(HERE / "cre_eval2.py").read().split("J = len(R) + 1; preds = {}")[0])
out = {}
def mae(pred): return float(np.abs(y[te] - pred[te]).mean())
out["table_only"] = mae(f)
for k in list(R) + ["STACK(all real readers)"]:
    h = stacked_pred(real if k.startswith("STACK") else [k], y - f)
    out[k] = mae(f + h)
out["true_mean_floor"] = mae(mu)
for k, v in out.items():
    print(f"{k:32s} TEST MAE {v:6.2f} min   gain vs table {out['table_only'] - v:+.2f}")
(HERE / "results/testbed_mae.json").write_text(json.dumps(out, indent=1))
