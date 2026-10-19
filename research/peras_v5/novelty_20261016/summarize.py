import json, sys, pandas as pd, numpy as np
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f) if l.startswith('{')]
d = pd.DataFrame(rows)
cols = ['peras_x', 'gamma_1.0', 'gamma_0.5', 'gamma_0.25', 'lip_oracle_x0.5', 'lip_oracle_x1.0', 'lip_oracle_x2.0', 'nn_estimate', 'gamma_test_estimate']
out = []
for nm, g in d.groupby('design', sort=False):
    r = {'design': nm, 'gold': round(g.gold.mean(), 3), 'Lor': round(g.L_oracle.mean(), 2) if 'L_oracle' in g and g.L_oracle.notna().any() else None}
    for c in cols:
        if c in g and g[c].notna().any():
            r[c] = f"{(g[c]/g.gold).mean():.2f}/{int((g[c] > g.gold + 1e-12).sum())}"
    r['held'] = f"{int(g.assumption_held_at_choice.sum())}/{len(g)}"
    r['X>lip1'] = int((g.peras_x > g.get('lip_oracle_x1.0', 0)).sum()) if 'lip_oracle_x1.0' in g else None
    out.append(r)
print(pd.DataFrame(out).to_string(index=False))
print("cells: mean ratio to floor / #runs above floor (violations; for estimates = overshoots)")
