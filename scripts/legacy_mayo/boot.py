"""Patient-clustered bootstrap of B* and E2 (out-of-fold predictions held fixed)."""
import numpy as np, pandas as pd, json
from build2 import q2, Qcap
from med2 import q50
S2=np.sqrt(2); mad=lambda x: np.abs(x-np.median(x)).mean()
df=pd.read_parquet('df.parquet'); y=df['case_actual_duration_time'].astype(float).values
pat=pd.factorize(df['patient_durable_id'].astype(str))[0]
P=np.load('PB.npy')                                   # tabular + note features
best=P[int(np.abs(y[None,:]-P).mean(1).argmin())]
cut=np.quantile(best,[.2,.4,.6,.8]); ST=np.searchsorted(cut,best)   # partition fixed a priori
blocks=[np.where(pat==u)[0] for u in np.unique(pat)]
def one(ix):
    yy=y[ix]; PP=P[:,ix]; ss=ST[ix]; out={'B':0.,'E':0.}
    for s in range(5):
        m=ss==s; w=m.mean()
        if m.sum()<30: continue
        yv=yy[m]; Pv=PP[:,m]
        Ls=np.abs(yv[None,:]-Pv).mean(1); k=int(Ls.argmin()); L=Ls[k]
        r=yv-Pv[k]; Mr=mad(r); rho=Mr/r.std()
        D=[mad(Pv[i]-Pv[j]) for i in range(len(Pv)) for j in range(i+1,len(Pv))]
        Mp=max(D); Mb=float(np.mean(D)); tau=(Mp/S2)/Mr
        out['B']+=w*np.sqrt(max((Mr/q2(rho,tau))**2-Mp**2,0))
        out['E']+=w*np.sqrt(max((L/q50(rho,tau))**2-(Mb/S2)**2,0))
    # global (unstratified) estimator, the validated form
    Ls=np.abs(yy[None,:]-PP).mean(1); k=int(Ls.argmin()); L=Ls[k]
    r=yy-PP[k]; Mr=mad(r); rho=Mr/r.std()
    D=[mad(PP[i]-PP[j]) for i in range(len(PP)) for j in range(i+1,len(PP))]
    Mp=max(D); Mb=float(np.mean(D)); tau=(Mp/S2)/Mr
    out['Eg']=np.sqrt(max((L/q50(rho,tau))**2-(Mb/S2)**2,0))
    out['Bg']=np.sqrt(max((Mr/q2(rho,tau))**2-Mp**2,0))
    out['Bcap']=np.sqrt(max((Mr/Qcap(tau))**2-Mp**2,0))
    out['L']=L
    return out
base=one(np.arange(len(y)))
rng=np.random.default_rng(0); n=len(blocks); B=[]
for b in range(800):
    ix=np.concatenate([blocks[i] for i in rng.integers(0,n,n)])
    B.append(one(ix))
K={k:np.array([d[k] for d in B]) for k in base}
print('%-28s %8s %8s %8s %8s'%('quantity','point','p5','p2.5','p97.5'))
for k,lab in [('B','B* stratified (bound)'),('Bg','B* global (bound)'),('Bcap','Qcap rung (bound)'),
              ('Eg','E2 global (estimator)'),('E','E2 stratified (estimator)'),('L','L (best OOF MAE)')]:
    v=K[k]; print('%-28s %8.2f %8.2f %8.2f %8.2f'%(lab,base[k],np.percentile(v,5),np.percentile(v,2.5),np.percentile(v,97.5)))
json.dump({k:dict(point=float(base[k]),p5=float(np.percentile(K[k],5)),lo=float(np.percentile(K[k],2.5)),
                  hi=float(np.percentile(K[k],97.5))) for k in base},open('boot.json','w'),indent=1)
