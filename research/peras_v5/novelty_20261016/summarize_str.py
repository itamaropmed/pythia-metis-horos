import json, sys, pandas as pd, numpy as np
d = pd.DataFrame([json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')])
out = []
for nm, g in d.groupby('design', sort=False):
    r = {'design': nm}
    for x in (1.0, .5, .25):
        r[f'01 g{x}'] = f"{(g[f'x01_g{x}']/g.gold01).mean():.2f}/{int((g[f'x01_g{x}']>g.gold01).sum())}"
        r[f'Brier g{x}'] = f"{(g[f'xB_g{x}']/g.gold_brier).mean():.2f}/{int((g[f'xB_g{x}']>g.gold_brier).sum())}"
        r[f'expOv g{x}'] = f"{g[f'exp_overshoot_g{x}'].mean():+.4f}"
        r[f'anchor rej g{x}'] = int((g[f'anchor_p_g{x}'] < .05).sum())
        r[f'inc rej g{x}'] = int((g[f'inc_p_g{x}'] < .05).sum())
    r['substudy'] = f"{(g.substudy_bound/g.gold01).mean():.2f}/{int((g.substudy_bound>g.gold01).sum())}"
    for k in ('pipeline_cap1', 'pipeline_cap_half'):
        r[k] = f"{(g[k]/g.gold01).mean():.2f}/{int((g[k]>g.gold01).sum())}"
    out.append(r)
print(pd.DataFrame(out).set_index('design').T.to_string())
if 'tau_g1.0' in d:
    rows=[]
    for nm, g in d.groupby('design', sort=False):
        r={'design':nm}
        for x in (1.0,.5):
            r[f'tau g{x}']=round(g[f'tau_g{x}'].mean(),3)
            r[f'overshoot g{x}']=round(g[f'overshoot_rest_g{x}'].mean(),3)
            ok=(g[f'anchor_p_g{x}']>=.05)
            r[f'undetected overshoot>tau g{x}']=int(((g[f'overshoot_rest_g{x}']>g[f'tau_g{x}'])&ok).sum())
        rows.append(r)
    print(pd.DataFrame(rows).set_index('design').T.to_string())
