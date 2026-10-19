import json, sys, pandas as pd, numpy as np
d = pd.DataFrame([json.loads(l) for f in sys.argv[1:] for l in open(f) if l.startswith('{')])
rows = []
for nm, g in d.groupby('dataset', sort=False):
    rows.append(dict(dataset=nm[:22], runs=len(g),
        IX_true=round(g.true_IX.mean(), 3), IX_cov=f"{int(((g.IX_L <= g.true_IX) & (g.true_IX <= g.IX_U)).sum())}",
        IX_int=f"[{g.IX_L.mean():.3f},{g.IX_U.mean():.3f}]",
        dI_true=round(g.true_dI.mean(), 3), dI_L=round(g.dI_L.mean(), 3), dI_pos=int((g.dI_L > 0).sum()), dI_viol=int((g.dI_L > g.true_dI + 1e-12).sum()),
        GX_cov=int(((g.GX_L <= g.true_GX) & (g.true_GX <= g.GX_U)).sum()), GXT_U_viol=int((g.GXT_U < g.true_GXT).sum()),
        p_lt05=int((g.p_text_nothing < .05).sum()), coinfo=round(g.coinfo_point.mean(), 3),
        modelAV=round(g.model_added_value.mean(), 3), modelAV_neg=int((g.model_added_value < 0).sum())))
print(pd.DataFrame(rows).to_string(index=False))
