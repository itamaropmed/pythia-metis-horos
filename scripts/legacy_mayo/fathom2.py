import numpy as np, pandas as pd, json, time, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor as HGB, ExtraTreesRegressor as ET
from xgboost import XGBRegressor
def members(seed):
    return [HGB(loss='absolute_error',max_iter=400,learning_rate=0.05,max_depth=6,
                min_samples_leaf=20,l2_regularization=1.0,random_state=seed),
            XGBRegressor(objective='reg:absoluteerror',n_estimators=400,max_depth=6,
                learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,min_child_weight=8,
                reg_lambda=1.0,tree_method='hist',random_state=seed,n_jobs=2),
            ET(n_estimators=300,min_samples_leaf=5,n_jobs=2,random_state=seed)]
def oof(X,y,groups,seed):
    rng=np.random.default_rng(seed); gu=pd.factorize(groups)[0]
    gp=rng.permutation(gu.max()+1)[gu]; P=[]
    for m in members(seed):
        p=np.empty(len(y))
        for tr,te in GroupKFold(5).split(X,y,groups=gp):
            mm=m.__class__(**m.get_params()); mm.fit(X[tr],y[tr]); p[te]=mm.predict(X[te])
        P.append(p)
    return P
df=pd.read_parquet('df.parquet')
y=df['case_actual_duration_time'].astype(float).values
g=df['patient_durable_id'].astype(str).values
for arm in ['A','B','C']:
    X=pd.read_parquet('X%s.parquet'%arm).values.astype(np.float32)
    t=time.time(); P=np.vstack(oof(X,y,g,11)+oof(X,y,g,112))
    np.save('P%s.npy'%arm,P); print(arm,'saved',P.shape,'%.0fs'%(time.time()-t),
        'MAEs',np.round(np.abs(y-P).mean(1),3),flush=True)
