import json, sys, pandas as pd
d = pd.DataFrame([json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')])
rows = []
for nm, g in d.groupby('dataset', sort=False):
    r = dict(dataset=nm[:22], runs=len(g), true=round(g.true_dI.mean(), 3))
    for c in ('static', 'diffusion', 'posterior', 'nca', 'portfolio'):
        r[c] = f"{g[c].mean():.3f} ({int((g[c] > 0).sum())}/{int((g[c] > g.true_dI + 1e-12).sum())})"
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False)); print("cells: mean certified dI (#positive / #violations)")
