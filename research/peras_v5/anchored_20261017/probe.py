import runpy, numpy as np, pandas as pd, sys
L = runpy.run_path("run_anchored.py", run_name="lib")
from peras.core.anchored_information import anchored_information
from sklearn.neighbors import NearestNeighbors
E, n = L["E"], L["n"]
def knn_score(cells, y, folds, f, k=15):
    s = np.zeros(n)
    for c in np.unique(cells):
        tr = np.flatnonzero((cells == c) & (folds != f)); te = np.flatnonzero((cells == c) & (folds == f))
        if len(te) == 0: continue
        kk = min(k, len(tr)); idx = NearestNeighbors(n_neighbors=kk).fit(E[tr]).kneighbors(E[te], return_distance=False)
        s[te] = y[tr][idx].mean(1)
    return s
for scen in ("S0_table_only", "S2_unique", "S3_synergy"):
    for rep in range(3):
        r = np.random.default_rng(4242 + rep + 1000 * len(scen))
        ageb, sex, lab = L["make_table"](r); cells = ageb * 4 + sex * 2 + lab
        q = L["law"](scen, ageb, sex, lab); y = (r.random(n) < q).astype(int)
        qbar = pd.Series(q).groupby(cells).transform("mean").to_numpy(); dI = float((2*qbar*(1-qbar)).mean() - (2*q*(1-q)).mean())
        a = anchored_information(cells, E, y, 2, .05, seed=rep)
        folds = np.random.default_rng(rep).integers(0, 2, n); parts = []
        for f in (0, 1):
            s = knn_score(cells, y, folds, f); te = folds == f
            parts.append(anchored_information(cells[te], s[te][:, None], y[te], 2, .025, seed=rep))
        print(scen, rep, 'true %.3f' % dI, '| text-NN point %.3f L %.3f' % (a['dI_point'], a['dI_lower']),
              '| guided point %.3f L %.3f' % (np.mean([p['dI_point'] for p in parts]), np.mean([p['dI_lower'] for p in parts])))
