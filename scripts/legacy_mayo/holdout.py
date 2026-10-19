import numpy as np, pandas as pd, warnings, json
warnings.filterwarnings('ignore')
from xgboost import XGBRegressor
df=pd.read_parquet('df.parquet')
y=df['case_actual_duration_time'].astype(float).values
t=df['scheduled_in_room']; pat=df['patient_durable_id'].astype(str)
cut=t.quantile(0.8); tp=set(pat[t>=cut])
te=np.where((t>=cut)&pat.isin(tp))[0]; tr=np.where((t<cut)&~pat.isin(tp))[0]
out={}
for arm in ['A','B','C']:
    X=pd.read_parquet('X%s.parquet'%arm).values.astype(np.float32)
    m=XGBRegressor(objective='reg:absoluteerror',n_estimators=400,max_depth=6,learning_rate=0.05,
        subsample=0.8,colsample_bytree=0.8,min_child_weight=8,reg_lambda=1.0,tree_method='hist',
        random_state=0,n_jobs=2)
    m.fit(X[tr],y[tr]); p=m.predict(X[te])
    ae=np.abs(y[te]-p)
    # patient-clustered bootstrap
    g=pat.values[te]; rng=np.random.default_rng(0)
    ug=np.unique(g); blocks=[np.where(g==u)[0] for u in ug]
    bs=[ae[np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))])].mean() for _ in range(1000)]
    out[arm]=dict(mae=float(ae.mean()),lo=float(np.percentile(bs,2.5)),hi=float(np.percentile(bs,97.5)),n_test=len(te))
    print('%s temporal-group holdout MAE %.2f  (95%% CI %.2f-%.2f)  n_test=%d'%(arm,out[arm]['mae'],out[arm]['lo'],out[arm]['hi'],len(te)),flush=True)
json.dump(out,open('holdout.json','w'),indent=1)
