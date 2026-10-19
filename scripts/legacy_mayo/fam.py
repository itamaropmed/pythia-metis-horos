import numpy as np, pandas as pd, json, warnings
warnings.filterwarnings('ignore')
from build2 import q2, Qcap
from med2 import q50
S2=np.sqrt(2); mad=lambda x: float(np.abs(x-np.median(x)).mean())
G,raw,maj1,H,OC=np.load('env.npy'); g1=lambda r: float(np.clip(np.interp(r,G,maj1),1,S2))
df=pd.read_parquet('df.parquet'); y=df['case_actual_duration_time'].astype(float).values
P=np.load('PB.npy')                      # arm B: tabular + 11 note-derived features
best=P[int(np.abs(y[None,:]-P).mean(1).argmin())]

def cell(y,P):
    Ls=np.abs(y[None,:]-P).mean(1); k=int(Ls.argmin()); L=float(Ls[k])
    r=y-P[k]; Mr=mad(r); rho=float(Mr/r.std())
    D=[mad(P[i]-P[j]) for i in range(len(P)) for j in range(i+1,len(P))]
    Mp=float(max(D)); Mb=float(np.mean(D)); tau=(Mp/S2)/Mr
    q2v=q2(rho,tau); q50v=q50(rho,tau)
    f=lambda q: float(np.sqrt(max((Mr/q)**2-Mp**2,0.0)))
    return dict(n=len(y),L=L,MADr=Mr,rho=rho,tau=float(tau),Mp=Mp,Mbar=Mb,q1=g1(rho),q2=q2v,q50=q50v,
                B1=f(g1(rho)),B2=f(q2v),Bcap=f(Qcap(tau)),
                E1=float(np.sqrt(max(L**2-(Mb/S2)**2,0.0))),
                E2=float(np.sqrt(max((L/q50v)**2-(Mb/S2)**2,0.0))))

def merge_small(lab,minn=400):
    lab=pd.Series(lab).astype(str); vc=lab.value_counts()
    return lab.where(lab.isin(vc[vc>=minn].index),'other').values

def partition(name,lab):
    lab=merge_small(lab); cs=[]
    for v in pd.unique(lab):
        m=lab==v; c=cell(y[m],P[:,m]); c['w']=float(m.mean()); c['name']=str(v); cs.append(c)
    cs=sorted(cs,key=lambda c:-c['w'])
    agg={k:float(sum(c['w']*c[k] for c in cs)) for k in ['B1','B2','Bcap','E1','E2','L','MADr','Mp']}
    agg.update(part=name,K=len(cs),cells=cs); return agg
