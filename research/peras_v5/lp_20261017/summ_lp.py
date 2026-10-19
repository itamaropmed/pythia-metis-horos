import json, sys, pandas as pd
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f) if l.startswith('{')]
out = []
for o in rows:
    r = dict(dataset=o['dataset'][:18], G=o['true_G'], GL1=o['G_lower_g1.0'], GLh=o['G_lower_g0.5'], GU=o['G_upper_free'],
             V=o['true_V'], VLfree=o['V_lower_free'], VLh=o['V_lower_g0.5'], VUh=o['V_upper_g0.5'], tip1=o['tipping_eps_g1.0'], tiph=o['tipping_eps_g0.5'])
    if 'true_table01' in o:
        r.update(t01=o['true_table01'], LPJ2lo=o['table01_LP_J2'][0], LPJ3lo=o['table01_LP_J3'][0], LPJ3hi=o['table01_LP_J3'][1])
    if 'map_spearman_point_vs_trueV' in o:
        r.update(map_rho=o['map_spearman_point_vs_trueV'], map_cert=o['map_cells_certified'], map_false=o['map_false_cert'])
    out.append(r)
d = pd.DataFrame(out); g = d.groupby('dataset', sort=False)
print(g.mean(numeric_only=True).round(3).T.to_string())
viol = g.apply(lambda x: pd.Series(dict(
    G_low_g1_viol=int((x.GL1 > x.G + 1e-9).sum()), G_low_half_viol=int((x.GLh > x.G + 1e-9).sum()), G_up_viol=int((x.GU < x.G - 1e-9).sum()),
    V_lowfree_viol=int((x.VLfree > x.V + 1e-9).sum()), V_up_half_viol=int((x.VUh < x.V - 1e-9).sum()),
    LP01_viol=int(((x.get('LPJ3lo', pd.Series(0, index=x.index)) > x.get('t01', pd.Series(1, index=x.index)) + 1e-9) |
                   (x.get('LPJ3hi', pd.Series(1, index=x.index)) < x.get('t01', pd.Series(0, index=x.index)) - 1e-9)).sum()))))
print(viol.to_string())
