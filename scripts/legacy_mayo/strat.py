"""Stratified Fathom.

R* = E[MAD(Y|X)] is additive over any measurable partition S=s(X):
    R* = sum_s w_s R*_s,   w_s = P(S=s),   R*_s = E[|eps| | S=s].
So if B_s <= R*_s for every stratum, sum_s w_s B_s <= R*.

Within a stratum we use MAD_s(r) rather than the mean absolute residual, which
makes the chain rigorous rather than approximate:
    (MAD_s(r)/qhat_s)^2 - MAD_s(e)^2 <= MAD_s(eps)^2 <= R*_s^2
the last step because MAD_s(eps) = min_t E[|eps-t| | s] <= E[|eps| | s] = R*_s.
"""
import numpy as np, pandas as pd, json
G,raw,maj,H,OC=np.load('env.npy')
mad=lambda x: float(np.abs(x-np.median(x)).mean())
guard=lambda rho: float(np.clip(np.interp(rho,G,maj),1.0,np.sqrt(2)))
S2=np.sqrt(2)

def cell(y,P,use_mad=True):
    """Fathom on one stratum (or the whole sample) from stored OOF predictions."""
    Ls=np.abs(y[None,:]-P).mean(1); k=int(Ls.argmin()); L=float(Ls[k])
    r=y-P[k]; Mr=mad(r); rho=Mr/r.std(); q=guard(rho)
    D=[mad(P[i]-P[j]) for i in range(len(P)) for j in range(i+1,len(P))]
    Mp=float(max(D)); Mb=float(np.mean(D))
    top=Mr if use_mad else L
    f=lambda qq: float(np.sqrt(max((top/qq)**2-Mp**2,0.0)))
    return dict(n=len(y),L=L,MADr=Mr,rho=float(rho),q=q,Mp=Mp,Mbar=Mb,
                B=f(1.0),Bg=f(q),Bsf=f(float(maj.max())),Bs2=f(S2),
                E=float(np.sqrt(max(L**2-(Mb/S2)**2,0.0))))

def stratified(y,P,strata,use_mad=True):
    out=[];n=len(y)
    for s in pd.unique(strata):
        m=strata==s
        c=cell(y[m],P[:,m],use_mad); c['w']=m.mean(); c['s']=str(s); out.append(c)
    agg={k:float(sum(c['w']*c[k] for c in out)) for k in ['B','Bg','Bsf','Bs2','E','L','MADr','Mp']}
    agg['cells']=out; agg['K']=len(out)
    return agg

def qcut_strata(v,K):
    return pd.qcut(v,K,labels=False,duplicates='drop').astype(int)
