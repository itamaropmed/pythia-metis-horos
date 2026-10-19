"""
Schedule-derivation ablation on the Mayo HRS duration task.

Replicates the incumbent construction (temporal split with train strictly before
held-out, patient separation, log target, absolute-error base learners,
patient-grouped out-of-fold cross-fitting, simplex MAE stacking in log space,
margin-gated family specialisation, held-out scored once) and re-runs it with the
schedule-derived block removed.

Arms
  FULL     all features
  NOSLOT   drop the booked-slot quantities only
           (scheduled_room_minutes, scheduled_setup_lead_minutes, scheduled_cleanup_minutes)
  NOSCHED  drop every scheduled_* derived feature (slot + calendar/time-of-day)
"""
import json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import (HistGradientBoostingRegressor as HGB,
                              ExtraTreesRegressor as ET, RandomForestRegressor as RF)
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize

SEED = 0
rng = np.random.default_rng(SEED)
t0 = time.time()


def log(*a):
    print('[%6.1fs]' % (time.time() - t0), *a, flush=True)


# ---------------------------------------------------------------- data
df = pd.read_parquet('df.parquet')
X = pd.read_parquet('XB.parquet')
assert len(df) == len(X)

y = df['case_actual_duration_time'].values.astype(float)
pid = df['patient_durable_id'].astype(str).values
proc = df['primary_procedure_name'].fillna('').astype(str).str.lower().values
tsched = pd.to_datetime(df['scheduled_in_room']).values

SLOT = ['scheduled_room_minutes', 'scheduled_setup_lead_minutes', 'scheduled_cleanup_minutes']
SCHED = [c for c in X.columns if c.startswith('scheduled_')]
log('schedule-derived columns present:', SCHED)
assert set(SLOT) <= set(SCHED)

# ---------------------------------------------------------------- split
o = np.argsort(tsched, kind='stable')
cut = int(round(0.793 * len(o)))
tr_idx, ho_idx = o[:cut], o[cut:]
tr_pat = set(pid[tr_idx])
keep = np.array([p not in tr_pat for p in pid[ho_idx]])
dropped = int((~keep).sum())
ho_idx = ho_idx[keep]
log('temporal split  train n=%d  held-out n=%d  (patient-separation dropped %d)'
    % (len(tr_idx), len(ho_idx), dropped))
log('train  %s  ->  %s' % (pd.Timestamp(tsched[tr_idx].max()), ''))
log('train ends   %s' % pd.Timestamp(tsched[tr_idx]. max()))
log('held-out begins %s' % pd.Timestamp(tsched[ho_idx].min()))
assert tsched[tr_idx].max() <= tsched[ho_idx].min()
assert not (set(pid[tr_idx]) & set(pid[ho_idx]))

ytr, yho = y[tr_idx], y[ho_idx]
ltr = np.log(np.maximum(ytr, 1.0))

# ---------------------------------------------------------------- families
PATTERNS = [('extraction', ('extract', 'removal', 'explant')),
            ('revision', ('revision', 'generator', 'upgrade', 'pocket')),
            ('af_pvi', ('af ', 'fibrillation', 'pvi', 'pulmonary vein', 'pfa')),
            ('vt_pvc', ('vt', 'ventricular tach', 'pvc')),
            ('other_ablation', ('ablat',)),
            ('implant', ('implant', 'insert', 'pacemaker', 'icd', 'crt', 'leadless', 'loop recorder')),
            ('diagnostic', ('study', 'test', 'diagnostic', 'tilt', 'cardiovers'))]


def family(s):
    for name, pats in PATTERNS:
        if any(p in s for p in pats):
            return name
    return 'other'


fam = np.array([family(s) for s in proc])
log('family counts (train):', dict(pd.Series(fam[tr_idx]).value_counts()))

# ---------------------------------------------------------------- learners
def base_models(seed=SEED):
    return [
        ('hgb600', HGB(loss='absolute_error', max_iter=600, learning_rate=0.045,
                       max_leaf_nodes=63, random_state=seed)),
        ('hgb400', HGB(loss='absolute_error', max_iter=400, learning_rate=0.06,
                       max_leaf_nodes=31, random_state=seed)),
        ('hgb500', HGB(loss='absolute_error', max_iter=500, learning_rate=0.05,
                       max_leaf_nodes=45, l2_regularization=1.0, random_state=seed)),
        ('et', ET(n_estimators=250, max_features=0.55, min_samples_leaf=8,
                  random_state=seed, n_jobs=-1)),
        ('rf', RF(n_estimators=200, max_features=0.4, min_samples_leaf=6,
                  random_state=seed, n_jobs=-1)),
    ]


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def simplex_stack(P, yv, seed=SEED):
    """min over the simplex of MAE(exp(P w), y); Nelder-Mead, 4 restarts."""
    B = P.shape[1]
    r = np.random.default_rng(seed)

    def obj(w):
        w = np.abs(w); s = w.sum()
        w = w / s if s > 0 else np.full(B, 1 / B)
        return mae(np.exp(P @ w), yv)

    starts = [np.full(B, 1 / B)] + [r.dirichlet(np.ones(B)) for _ in range(3)]
    best, bw = np.inf, starts[0]
    for s0 in starts:
        res = minimize(obj, s0, method='Nelder-Mead',
                       options=dict(maxiter=4000, xatol=1e-6, fatol=1e-8))
        if res.fun < best:
            best, bw = res.fun, np.abs(res.x) / np.abs(res.x).sum()
    return bw, best


ALPHAS = np.round(np.arange(0, 1.01, 0.1), 2)


def run_arm(cols, tag):
    log('=' * 62)
    log('ARM %s   p = %d features' % (tag, len(cols)))
    Xtr = X[cols].values[tr_idx]
    Xho = X[cols].values[ho_idx]

    # ---- cross-fitting, patient-grouped
    gkf = GroupKFold(n_splits=3)
    folds = list(gkf.split(Xtr, ltr, groups=pid[tr_idx]))
    B = len(base_models())
    P_oof = np.zeros((len(tr_idx), B))
    P_app = np.zeros((len(ho_idx), B))
    for b, (name, _) in enumerate(base_models()):
        for k, (ia, ib) in enumerate(folds):
            m = base_models()[b][1]
            m.fit(Xtr[ia], ltr[ia])
            P_oof[ib, b] = m.predict(Xtr[ib])
        m = base_models()[b][1]
        m.fit(Xtr, ltr)
        P_app[:, b] = m.predict(Xho)
        log('  %-7s oof MAE %.4f   holdout MAE %.4f'
            % (name, mae(np.exp(P_oof[:, b]), ytr), mae(np.exp(P_app[:, b]), yho)))

    # ---- stacking
    w, oof_mae = simplex_stack(P_oof, ytr)
    med_tr = np.exp(np.median(P_oof, axis=1))
    med_ho = np.exp(np.median(P_app, axis=1))
    g_tr = np.exp(P_oof @ w)
    G_ho = np.exp(P_app @ w)
    log('  stack w = %s' % np.round(w, 3).tolist())
    log('  median-combined holdout MAE %.4f | stacked holdout MAE %.4f'
        % (mae(med_ho, yho), mae(G_ho, yho)))

    # ---- margin-gated family specialisation
    yhat_ho = G_ho.copy()
    alphas = {}
    for f in sorted(set(fam)):
        m_tr = fam[tr_idx] == f
        if m_tr.sum() < 250:
            alphas[f] = 0.0
            continue
        Xf, lf, yf, pf = Xtr[m_tr], ltr[m_tr], ytr[m_tr], pid[tr_idx][m_tr]
        s_oof = np.zeros(m_tr.sum())
        ns = min(3, len(set(pf)))
        for ia, ib in GroupKFold(n_splits=ns).split(Xf, lf, groups=pf):
            sm = HGB(loss='absolute_error', max_iter=400, learning_rate=0.05,
                     max_leaf_nodes=31, random_state=SEED)
            sm.fit(Xf[ia], lf[ia])
            s_oof[ib] = sm.predict(Xf[ib])
        s_oof = np.exp(s_oof)
        gf = g_tr[m_tr]
        base = mae(gf, yf)
        scores = [mae((1 - a) * gf + a * s_oof, yf) for a in ALPHAS]
        a_star = float(ALPHAS[int(np.argmin(scores))])
        if min(scores) + 0.05 < base and a_star > 0:
            alphas[f] = a_star
            sm = HGB(loss='absolute_error', max_iter=400, learning_rate=0.05,
                     max_leaf_nodes=31, random_state=SEED)
            sm.fit(Xf, lf)
            m_ho = fam[ho_idx] == f
            if m_ho.sum():
                S = np.exp(sm.predict(Xho[m_ho]))
                yhat_ho[m_ho] = (1 - a_star) * G_ho[m_ho] + a_star * S
        else:
            alphas[f] = 0.0
    log('  accepted family weights: %s' % {k: v for k, v in alphas.items() if v > 0})
    log('  FINAL stacked + family holdout MAE  %.4f' % mae(yhat_ho, yho))
    return dict(tag=tag, p=len(cols), w=w.tolist(), alphas=alphas,
                oof_mae=oof_mae,
                mae_median=mae(med_ho, yho), mae_stack=mae(G_ho, yho),
                mae_final=mae(yhat_ho, yho),
                pred=yhat_ho, pred_stack=G_ho)


ALL = list(X.columns)
arms = {}
arms['FULL'] = run_arm(ALL, 'FULL')
arms['NOSLOT'] = run_arm([c for c in ALL if c not in SLOT], 'NOSLOT')
arms['NOSCHED'] = run_arm([c for c in ALL if c not in SCHED], 'NOSCHED')

# ---------------------------------------------------------------- inference
def boot_paired(a, b, groups, reps=3000, seed=SEED):
    """patient-clustered bootstrap of mean(|y-a|) - mean(|y-b|)"""
    da = np.abs(yho - a); db = np.abs(yho - b)
    d = da - db
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    r = np.random.default_rng(seed)
    out = np.empty(reps)
    for i in range(reps):
        pick = r.choice(ug, size=len(ug), replace=True)
        sel = np.concatenate([idx[g] for g in pick])
        out[i] = d[sel].mean()
    return float(d.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def boot_mae(a, groups, reps=3000, seed=SEED):
    e = np.abs(yho - a)
    ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}
    r = np.random.default_rng(seed)
    out = np.empty(reps)
    for i in range(reps):
        pick = r.choice(ug, size=len(ug), replace=True)
        sel = np.concatenate([idx[g] for g in pick])
        out[i] = e[sel].mean()
    return float(e.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


gho = pid[ho_idx]
log('=' * 62)
res = {}
for k, a in arms.items():
    m, lo, hi = boot_mae(a['pred'], gho)
    res[k] = dict(mae=m, lo=lo, hi=hi, p=a['p'], alphas=a['alphas'], w=a['w'],
                  mae_median=a['mae_median'], mae_stack=a['mae_stack'])
    log('%-8s p=%3d   held-out MAE %.4f  [%.3f, %.3f]' % (k, a['p'], m, lo, hi))

for k in ['NOSLOT', 'NOSCHED']:
    d, lo, hi = boot_paired(arms[k]['pred'], arms['FULL']['pred'], gho)
    res['delta_%s' % k] = dict(delta=d, lo=lo, hi=hi)
    log('cost of removing (%s − FULL):  %+.4f min   95%% CI [%+.3f, %+.3f]' % (k, d, lo, hi))

# ---------------------------------------------------------------- leak diagnostics
slot = df['scheduled_room_minutes'].values.astype(float)
diag = dict(
    n=int(len(y)),
    corr=float(np.corrcoef(slot, y)[0, 1]),
    exact=int((slot == y).sum()),
    within1=float(np.mean(np.abs(slot - y) <= 1)),
    within5=float(np.mean(np.abs(slot - y) <= 5)),
    within10=float(np.mean(np.abs(slot - y) <= 10)),
    median_signed=float(np.median(y - slot)),
    mae_slot_alone=float(np.mean(np.abs(y - slot))),
    frac_slot_over=float(np.mean(slot > y)),
    round_mult_15=float(np.mean(slot % 15 == 0)),
    round_mult_15_y=float(np.mean(y % 15 == 0)),
)
res['slot_diagnostics'] = diag
log('slot diagnostics: %s' % json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                         for k, v in diag.items()}))

json.dump(res, open('sched_ablation.json', 'w'), indent=1)
np.save('sched_pred.npy', np.vstack([arms[k]['pred'] for k in ['FULL', 'NOSLOT', 'NOSCHED']]))
np.save('sched_yho.npy', yho)
log('done')
