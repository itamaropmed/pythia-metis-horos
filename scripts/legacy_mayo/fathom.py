import numpy as np, pandas as pd, json, time, sys, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor as HGB, ExtraTreesRegressor as ET
from xgboost import XGBRegressor
G,raw,maj,H,OC=np.load('env.npy')
mad=lambda x: float(np.abs(x-np.median(x)).mean())
def members(seed):
    return [HGB(loss='absolute_error',max_iter=400,learning_rate=0.05,max_depth=6,
                min_samples_leaf=20,l2_regularization=1.0,random_state=seed),
            XGBRegressor(objective='reg:absoluteerror',n_estimators=400,max_depth=6,
                learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,min_child_weight=8,
                reg_lambda=1.0,tree_method='hist',random_state=seed,n_jobs=2),
            ET(n_estimators=300,min_samples_leaf=5,n_jobs=2,random_state=seed)]
def oof(X,y,groups,seed):
    idx=np.arange(len(y)); rng=np.random.default_rng(seed)
    # patient-grouped 5-fold; the shuffle of group order is what makes the two partitions distinct
    gu=pd.factorize(groups)[0]; perm=rng.permutation(gu.max()+1); gp=perm[gu]
    P=[]
    for m in members(seed):
        p=np.empty(len(y))
        for tr,te in GroupKFold(5).split(X,y,groups=gp):
            mm=m.__class__(**m.get_params()); mm.fit(X[tr],y[tr]); p[te]=mm.predict(X[te])
        P.append(p)
    return P
def fathom(X,y,groups,seed=11):
    P=oof(X,y,groups,seed)+oof(X,y,groups,seed+101)
    Ls=np.array([np.abs(y-p).mean() for p in P]); L=float(Ls.min())
    Ds=[mad(P[i]-P[j]) for i in range(len(P)) for j in range(i+1,len(P))]
    Mp=float(max(Ds)); Mbar=float(np.mean(Ds))
    r=y-P[int(Ls.argmin())]; rho=mad(r)/r.std()
    qh=float(np.clip(np.interp(rho,G,maj),1.0,np.sqrt(2))); qm=float(maj.max())
    f=lambda q: float(np.sqrt(max((L/q)**2-Mp**2,0.0)))
    return dict(L=L,Ls=[float(v) for v in Ls],Mbar=Mbar,Mp=Mp,rho=float(rho),qhat=qh,
                B=f(1.0),Bg=f(qh),Bsf=f(qm),Bs2=f(np.sqrt(2)),
                E=float(np.sqrt(max(L**2-(Mbar/np.sqrt(2))**2,0.0))),
                protected=bool(True))
df=pd.read_parquet('df.parquet')
y=df['case_actual_duration_time'].astype(float).values
groups=df['patient_durable_id'].astype(str).values
res={}
for arm,lab in [('A','tabular only'),('B','tabular + 11 note-derived features'),('C','tabular + notes (features + text)')]:
    X=pd.read_parquet('X%s.parquet'%arm).values.astype(np.float32)
    t=time.time(); r=fathom(X,y,groups); r['label']=lab; r['p']=X.shape[1]
    res[arm]=r
    print('%s %-38s p=%3d  L=%.3f  M+=%.3f  Mbar=%.3f  rho=%.3f  qhat=%.4f  [%.0fs]'%(
        arm,lab,X.shape[1],r['L'],r['Mp'],r['Mbar'],r['rho'],r['qhat'],time.time()-t),flush=True)
    print('    member OOF MAEs:',np.round(r['Ls'],2))
    print('    B=%.2f  B*=%.2f  Bsf=%.2f  Bs2=%.2f   E=%.2f'%(r['B'],r['Bg'],r['Bsf'],r['Bs2'],r['E']),flush=True)
json.dump(res,open('fathom_arms.json','w'),indent=1)
