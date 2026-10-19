"""Rolling-origin validation of the schedule ablation, inside the training partition
only (the held-out block is never touched), plus the NOCAL arm that completes the
decomposition of the schedule block into booked slot vs calendar/time-of-day."""
import json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

SEED = 0
t0 = time.time()
def log(*a): print('[%6.1fs]' % (time.time() - t0), *a, flush=True)

df = pd.read_parquet('df.parquet'); X = pd.read_parquet('XB.parquet')
y = df['case_actual_duration_time'].values.astype(float)
pid = df['patient_durable_id'].astype(str).values
tsched = pd.to_datetime(df['scheduled_in_room']).values

SLOT = ['scheduled_room_minutes', 'scheduled_setup_lead_minutes', 'scheduled_cleanup_minutes']
SCHED = [c for c in X.columns if c.startswith('scheduled_')]
CAL = [c for c in SCHED if c not in SLOT]
ALL = list(X.columns)
ARMS = {'FULL': ALL,
        'NOSLOT': [c for c in ALL if c not in SLOT],
        'NOCAL': [c for c in ALL if c not in CAL],
        'NOSCHED': [c for c in ALL if c not in SCHED]}
log('slot block   :', SLOT)
log('calendar blk :', CAL)

o = np.argsort(tsched, kind='stable')
cut = int(round(0.793 * len(o)))
tr = o[:cut]                      # training partition only
ytr, ltr = y[tr], np.log(np.maximum(y[tr], 1.0))

CFG = [dict(max_iter=600, learning_rate=0.045, max_leaf_nodes=63),
       dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31),
       dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=45, l2_regularization=1.0)]
WIN = [(0.00, 0.55, 0.70), (0.00, 0.70, 0.85), (0.00, 0.85, 1.00)]

res = {k: [] for k in ARMS}
n = len(tr)
for wi, (a, b, c) in enumerate(WIN, 1):
    ia = tr[int(a * n):int(b * n)]
    ib = tr[int(b * n):int(c * n)]
    # patient separation inside the window
    ps = set(pid[ia])
    ib = ib[[p not in ps for p in pid[ib]]]
    assert tsched[ia].max() <= tsched[ib].min()
    for k, cols in ARMS.items():
        Xa, Xb = X[cols].values[ia], X[cols].values[ib]
        la = np.log(np.maximum(y[ia], 1.0))
        P = []
        for cfg in CFG:
            m = HGB(loss='absolute_error', random_state=SEED, **cfg)
            m.fit(Xa, la)
            P.append(m.predict(Xb))
        pred = np.exp(np.median(np.vstack(P), axis=0))
        v = float(np.mean(np.abs(y[ib] - pred)))
        res[k].append(v)
        log('window %d  n_fit=%5d n_val=%5d  %-8s  MAE %.4f' % (wi, len(ia), len(ib), k, v))

log('=' * 60)
out = {}
for k in ARMS:
    out[k] = dict(folds=res[k], mean=float(np.mean(res[k])))
    log('%-8s  folds %s   mean %.4f' % (k, np.round(res[k], 3).tolist(), np.mean(res[k])))
for k in ['NOSLOT', 'NOCAL', 'NOSCHED']:
    d = np.mean(res[k]) - np.mean(res['FULL'])
    per = np.array(res[k]) - np.array(res['FULL'])
    out['delta_' + k] = dict(mean=float(d), per_window=per.round(4).tolist())
    log('cost of removing %-8s  %+.4f min   per window %s' % (k, d, per.round(3).tolist()))
json.dump(out, open('sched_rolling.json', 'w'), indent=1)
log('done')
