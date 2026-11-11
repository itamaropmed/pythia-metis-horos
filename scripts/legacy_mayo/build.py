import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
D='/mnt/user-data/uploads/Clinical_notes/'
TARGET='case_actual_duration_time'; CASE='case_durable_id'; PAT='patient_durable_id'
SURG='surgeon_durable_id'; PROC='primary_procedure_name'
SCHED_IN='scheduled_in_room'; ACTUAL_IN='case_actual_patient_in_room'
ACTUAL_OUT='case_actual_patient_out_of_room'
POSTHOC=[TARGET,'case_actual_prep_time','case_actual_procedure_time','case_actual_wrapup_time',ACTUAL_IN,ACTUAL_OUT]
NONFEAT=[CASE,PAT,'patient_birth_date','scheduled_in_room','scheduled_setup_start','scheduled_out_of_room','scheduled_cleanup_complete','all_procs_not_performed']
SHIPPED=['lag_sp_mean','lag_sp_std','lag_sp_count','lag_proc_mean','lag_proc_std','lag_proc_count','prior_same_proc_duration','redo_time_gap_days']

def load():
    df=pd.read_parquet(D+'data/mayo/mayo_hrs_tabular_features_with_lags_and_durations.parquet')
    for c in (SCHED_IN,ACTUAL_IN): df[c]=pd.to_datetime(df[c],errors='coerce',utc=True)
    return df.sort_values(ACTUAL_IN).reset_index(drop=True)

def exclusions(df):
    log={}; keep=pd.Series(True,index=df.index)
    bad=df[TARGET].isna()|(df[TARGET].astype(float)<=0); log['null_or_nonpositive_target']=int(bad.sum()); keep&=~bad
    sd=(pd.to_datetime(df['scheduled_out_of_room'],errors='coerce',utc=True)-df[SCHED_IN]).dt.total_seconds()/60.
    bf=((sd-df[TARGET]).abs()<1/60.).fillna(False); log['backfilled']=int((bf&keep).sum()); keep&=~bf
    if 'all_procs_not_performed' in df:
        ab=df['all_procs_not_performed'].fillna(False).astype(bool); log['aborted']=int((ab&keep).sum()); keep&=~ab
    npr=df[PROC].isna(); log['null_procedure']=int((npr&keep).sum()); keep&=~npr
    return df[keep].reset_index(drop=True), log

def history(df,window=50):
    """Build strictly as-of aggregates from cases completed before scheduling.

    Ordering by actual start and shifting the target leaks outcomes from cases
    that are still in progress. Require a trusted actual-out timestamp and
    admit a historical target only when its completion precedes this case's
    scheduled start. The completion timestamp itself is dropped from X below.
    """
    if ACTUAL_OUT not in df:
        raise ValueError(f"history features require completion timestamp {ACTUAL_OUT!r}")
    if SCHED_IN not in df or TARGET not in df or CASE not in df:
        raise ValueError("history features require case ID, scheduled time, and duration target")
    if df[CASE].isna().any() or df[CASE].duplicated().any():
        raise ValueError("history features require complete unique case IDs")
    if window < 1:
        raise ValueError("history window must be positive")
    out=df.copy()
    scheduled=pd.to_datetime(out[SCHED_IN],errors='coerce',utc=True)
    completed=pd.to_datetime(out[ACTUAL_OUT],errors='coerce',utc=True)
    y=pd.to_numeric(out[TARGET],errors='coerce').to_numpy(float)
    if scheduled.isna().any() or completed.isna().any():
        raise ValueError("history feature timestamps must be complete and parseable")
    sched_ns=scheduled.astype('int64').to_numpy()
    end_ns=completed.astype('int64').to_numpy()
    case_ids=out[CASE].astype(str).to_numpy()

    specifications={
        'sp':([SURG,PROC],True), 'proc':([PROC],True), 'surgeon':([SURG],True),
        'patient_prior':([PAT],False),
    }
    for name,(keys,limited) in specifications.items():
        if any(key not in out for key in keys):
            raise ValueError(f"history feature {name!r} lacks grouping columns {keys}")
        mean=np.full(len(out),np.nan); median=np.full(len(out),np.nan)
        std=np.full(len(out),np.nan); count=np.zeros(len(out),float)
        grouped=out.groupby(keys,observed=True,dropna=False,sort=False).indices
        for indices in grouped.values():
            rows=np.asarray(indices,dtype=int)
            # Stable ties are allowed, but a case is only visible strictly
            # before the current scheduled instant (never at the same time).
            order=rows[np.argsort(end_ns[rows],kind='stable')]
            ordered_end=end_ns[order]
            for row in rows:
                stop=np.searchsorted(ordered_end,sched_ns[row],side='left')
                prior=order[:stop]
                prior=prior[case_ids[prior]!=case_ids[row]]
                prior=prior[np.isfinite(y[prior])]
                if limited:
                    prior=prior[-window:]
                values=y[prior]
                count[row]=len(values)
                if len(values):
                    mean[row]=float(np.mean(values)); median[row]=float(np.median(values))
                if len(values)>1:
                    std[row]=float(np.std(values,ddof=1))
        out[f'hist_{name}_mean']=mean
        if limited:
            out[f'hist_{name}_median']=median
            out[f'hist_{name}_std']=std
        out[f'hist_{name}_count']=count
    out['hist_sp_is_first']=(out['hist_sp_count']==0).astype(int)
    return out

def schedule(df):
    o=df.copy(); s=o[SCHED_IN]
    o['scheduled_year']=s.dt.year; o['scheduled_month']=s.dt.month
    o['scheduled_dayofweek']=s.dt.dayofweek; o['scheduled_hour']=s.dt.hour
    o['scheduled_is_weekend']=s.dt.dayofweek.isin([5,6]).astype(int)
    m=lambda a,b:(pd.to_datetime(o[a],errors='coerce',utc=True)-pd.to_datetime(o[b],errors='coerce',utc=True)).dt.total_seconds()/60.
    o['scheduled_room_minutes']=m('scheduled_out_of_room','scheduled_in_room')
    o['scheduled_setup_lead_minutes']=m('scheduled_in_room','scheduled_setup_start')
    o['scheduled_cleanup_minutes']=m('scheduled_cleanup_complete','scheduled_out_of_room')
    o['bmi_is_missing']=o['bmi'].isna().astype(int) if 'bmi' in o else 0
    return o

def frame(df,rare_min=25):
    drop=set(POSTHOC)|set(NONFEAT)|set(SHIPPED)|{c for c in df.columns if c.endswith('_date')}
    X=df[[c for c in df.columns if c not in drop]].copy()
    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]): X[c]=X[c].astype(int)
    cats=[c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    for c in cats:
        s=X[c].astype('string'); vc=s.value_counts(); keep=set(vc[vc>=rare_min].index)
        X[c]=s.where(s.isin(keep),'__rare__').fillna('__missing__')
    return X,cats

if __name__=='__main__':
    df,log=exclusions(load()); print('exclusions',log,'->',len(df))
    df=schedule(history(df))
    df=df[df[SCHED_IN]>=pd.Timestamp('2023-01-01',tz='UTC')].reset_index(drop=True)
    print('after burn-in',len(df),' patients',df[PAT].nunique())
    X,cats=frame(df); print('X',X.shape,'cats',len(cats))
    print('levels per cat:',{c:X[c].nunique() for c in cats})
    Xd=pd.get_dummies(X,columns=cats,dummy_na=False).astype(float)
    print('one-hot width',Xd.shape[1])
    df.to_parquet('df.parquet'); Xd.to_parquet('Xtab.parquet')
    print('y: n=%d med=%.1f MAD=%.2f'%(len(df),df[TARGET].median(),np.abs(df[TARGET]-df[TARGET].median()).mean()))
