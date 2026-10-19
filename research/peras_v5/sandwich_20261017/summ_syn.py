import json, sys, pandas as pd
d = pd.DataFrame([json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')])
rows = []
for nm, g in d.groupby('design', sort=False):
    rows.append(dict(design=nm, G=round(g.G.mean(), 3), I=round(g.I.mean(), 3),
        GU=f"{(g.GU/g.G).mean():.2f}/{int((g.GU<g.G).sum())}", modelGU=f"{(g.model_GU/g.G).mean():.2f}/{int((g.model_GU<g.G).sum())}",
        GL=f"{(g.GL/g.G).mean():.2f}/{int((g.GL>g.G).sum())}",
        RU=f"{(g.RU/g.R).mean():.2f}/{int((g.RU<g.R).sum())}",
        IL=f"{g.IL.mean():.3f}/{int((g.IL>g.I).sum())}", IU=f"{g.IU.mean():.3f}/{int((g.IU<g.I).sum())}",
        BSS=round(g.BSS.mean(), 3), BSS_L=f"{g.BL.mean():.3f}/{int((g.BL>g.BSS).sum())}"))
print(pd.DataFrame(rows).to_string(index=False))
print("upper cells: ratio U/true / #runs U < true (violations); lower cells: ratio or value / #violations")
