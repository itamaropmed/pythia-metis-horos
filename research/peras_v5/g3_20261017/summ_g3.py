import json, sys, pandas as pd, numpy as np
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')]
losses = [k for k in ('01', 'brier', 'log', 'mse', 'mae', 'pinball') if k in rows[0]]
T = []
for L in losses:
    for o in rows:
        c = o[L]; T.append(dict(dataset=o['dataset'][:20], loss=L, jL=c['joint_floor'][0], jU=c['joint_floor'][1], jT=c.get('true_joint'),
                               xL=c['table_floor'][0], xU=c['table_floor'][1], xT=c.get('true_table'),
                               dgL=c['data_gap_lower'], dgT=c.get('true_data_gap'), egL=c['estimation_gap_lower'], egU=c['estimation_gap_upper'],
                               egT=c.get('true_est_gap'), mL=c['model_risk'][0], mU=c['model_risk'][1], mT=c.get('true_model')))
d = pd.DataFrame(T)
g = d.groupby(['dataset', 'loss'], sort=False)
agg = g[['jL', 'jU', 'jT', 'xL', 'xU', 'xT', 'dgL', 'dgT', 'egL', 'egU', 'egT']].mean().round(3)
if d.jT.notna().any():
    agg['viol'] = g.apply(lambda x: int(((x.jL > x.jT + 1e-9) | (x.jU < x.jT - 1e-9) | (x.xL > x.xT + 1e-9) | (x.xU < x.xT - 1e-9) |
                                          (x.dgL > x.dgT + 1e-9) | (x.egL > x.egT + 1e-9) | (x.egU < x.egT - 1e-9)).sum()))
print(agg.to_string())
if 'per_sample' in rows[0]:
    print(pd.DataFrame([dict(dataset=o['dataset'][:20], **{k: v for k, v in o['per_sample'].items() if not isinstance(v, dict)}) for o in rows]).groupby('dataset', sort=False).mean(numeric_only=True).round(3).to_string())
