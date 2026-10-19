import numpy as np, pandas as pd, json
from build2 import q2, Qcap
S2=np.sqrt(2); mad=lambda x: float(np.abs(x-np.median(x)).mean())
G,raw,maj1,H,OC=np.load('env.npy')
g1=lambda rho: float(np.clip(np.interp(rho,G,maj1),1,S2))
df=pd.read_parquet('df.parquet'); y=df['case_actual_duration_time'].astype(float).values

def cell(y,P):
    Ls=np.abs(y[None,:]-P).mean(1); k=int(Ls.argmin()); L=float(Ls[k])
    r=y-P[k]; Mr=mad(r); rho=Mr/r.std()
    D=[mad(P[i]-P[j]) for i in range(len(P)) for j in range(i+1,len(P))]
    Mp=float(max(D)); Mb=float(np.mean(D)); tau=(Mp/S2)/Mr
    f=lambda q: float(np.sqrt(max((Mr/q)**2-Mp**2,0.0)))
    return dict(n=len(y),L=L,MADr=Mr,rho=float(rho),tau=float(tau),Mp=Mp,Mbar=Mb,
                q1=g1(rho),q2=q2(rho,tau),qcap=float(Qcap(tau)),
                B=f(1.0),B1=f(g1(rho)),B2=f(q2(rho,tau)),Bcap=f(Qcap(tau)),Bs2=f(S2),
                E=float(np.sqrt(max(L**2-(Mb/S2)**2,0.0))))
def strat(y,P,s):
    cs=[]
    for v in np.unique(s):
        m=s==v; c=cell(y[m],P[:,m]); c['w']=float(m.mean()); cs.append(c)
    agg={k:float(sum(c['w']*c[k] for c in cs)) for k in ['B','B1','B2','Bcap','Bs2','E','L','MADr','Mp']}
    agg['cells']=cs; return agg
out={}
for arm,lab in [('A','tabular only'),('B','tabular + 11 note features'),('C','tabular + notes (features + text)')]:
    P=np.load('P%s.npy'%arm); gl=cell(y,P)
    best=P[int(np.abs(y[None,:]-P).mean(1).argmin())]
    st5=pd.qcut(best,5,labels=False,duplicates='drop').astype(int)
    sr=strat(y,P,st5)
    out[arm]=dict(label=lab,glob=gl,strat5=sr,
                  sens={str(K):strat(y,P,pd.qcut(best,K,labels=False,duplicates='drop').astype(int))['B2'] for K in [2,3,5,10]})
    print('%s %-34s rho=%.3f tau=%.3f | q: 1D %.4f  2D %.4f  cap %.4f'%(arm,lab,gl['rho'],gl['tau'],gl['q1'],gl['q2'],gl['qcap']))
    print('    global   B=%.2f  B1=%.2f  B2=%.2f  Bcap=%.2f  Bs2=%.2f | E=%.2f  L=%.2f'%(gl['B'],gl['B1'],gl['B2'],gl['Bcap'],gl['Bs2'],gl['E'],gl['L']))
    print('    strat-5  B=%.2f  B1=%.2f  B2=%.2f  Bcap=%.2f  Bs2=%.2f | E=%.2f'%(sr['B'],sr['B1'],sr['B2'],sr['Bcap'],sr['Bs2'],sr['E']))
    print('    K-sens (B2): '+'  '.join('K=%s:%.2f'%(k,v) for k,v in out[arm]['sens'].items()))
    if arm=='B':
        print('    per-stratum: '+' | '.join('n=%d MADr=%.1f rho=%.3f tau=%.3f q2=%.4f M+=%.1f B2=%.1f'%(
            c['n'],c['MADr'],c['rho'],c['tau'],c['q2'],c['Mp'],c['B2']) for c in sr['cells']))
json.dump({k:{'label':v['label'],'glob':v['glob'],'strat5':{a:b for a,b in v['strat5'].items() if a!='cells'},
              'strat5_cells':v['strat5']['cells'],'sens':v['sens']} for k,v in out.items()},open('final.json','w'),indent=1)
