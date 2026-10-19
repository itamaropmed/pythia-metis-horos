import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
D='/mnt/user-data/uploads/Clinical_notes/'
TARGET='case_actual_duration_time'; CASE='case_durable_id'; PAT='patient_durable_id'
SURG='surgeon_durable_id'; PROC='primary_procedure_name'
SCHED_IN='scheduled_in_room'; ACTUAL_IN='case_actual_patient_in_room'
POSTHOC=[TARGET,'case_actual_prep_time','case_actual_procedure_time','case_actual_wrapup_time',ACTUAL_IN]
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
    out=df.sort_values([ACTUAL_IN,CASE]).copy(); y=out[TARGET].astype(float)
    for name,keys in {'sp':[SURG,PROC],'proc':[PROC],'surgeon':[SURG]}.items():
        g=y.groupby([out[k] for k in keys],observed=True); sh=g.transform(lambda s:s.shift(1))
        r=sh.groupby([out[k] for k in keys],observed=True)
        out[f'hist_{name}_mean']=r.transform(lambda s:s.rolling(window,min_periods=1).mean())
        out[f'hist_{name}_median']=r.transform(lambda s:s.rolling(window,min_periods=1).median())
        out[f'hist_{name}_std']=r.transform(lambda s:s.rolling(window,min_periods=2).std())
        out[f'hist_{name}_count']=r.transform(lambda s:s.rolling(window,min_periods=1).count()).fillna(0.)
    gp=y.groupby(out[PAT],observed=True)
    out['hist_patient_prior_mean']=gp.transform(lambda s:s.shift(1).expanding().mean())
    out['hist_patient_prior_count']=gp.transform(lambda s:s.shift(1).expanding().count()).fillna(0.)
    out['hist_sp_is_first']=(out['hist_sp_count']==0).astype(int)
    return out.sort_index()

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
