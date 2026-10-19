import runpy, numpy as np, pandas as pd
L = runpy.run_path("run_anchored.py", run_name="lib")
from peras.core.anchored_information import anchored_information, _greedy_pairs
n=L['n']; r=np.random.default_rng(1); ageb,sex,lab=L['make_table'](r); cells=ageb*4+sex*2+lab
q=L['law']("S0_table_only",ageb,sex,lab); y=(r.random(n)<q).astype(int)
s=np.random.default_rng(5).random(n)            # pure noise score
a=anchored_information(cells, s[:,None], y, 2, .05)
print('noise score', a['Q_all'], a['Q_txt'])
s2=np.round(np.random.default_rng(5).random(n)*15)/15
a=anchored_information(cells, s2[:,None], y, 2, .05)
print('tied noise score', a['Q_all'], a['Q_txt'])
I=np.flatnonzero(cells==cells[0]); P=_greedy_pairs(s2[I][:,None]); print(len(I), len(P), len(set(np.array(P).ravel())))
