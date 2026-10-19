"""Re-validate the 2-D guard on suites K and P, from stored per-design quantities."""
import numpy as np, json, pandas as pd
S2=np.sqrt(2); Qcap=lambda tau: 1.0/np.sqrt((1-tau)**2+tau**2)
# ---- suite K: published Table 12 (ratios to R*) + Table 4 (rho) ----
NAME=['K%02d'%i for i in range(1,21)]
Bg =np.array([0.611,0,0.696,0.854,0.851,0,0.915,0.911,0.769,0.778,0.840,0.879,0.776,0.917,0.886,0.836,0.564,0.887,0,0.918])
B  =np.array([0.744,0.396,0.949,0.935,1.008,0,0.983,0.951,0.998,0.987,0.988,1.072,0.975,0.984,1.010,0.981,0.789,0.960,0,0.998])
E  =np.array([1.016,0.931,1.028,1.028,1.061,1.304,1.002,0.980,1.037,1.016,1.042,1.178,1.004,1.015,1.036,1.016,0.999,0.980,1.004,1.034])
L  =np.array([1.145,1.138,1.082,1.075,1.100,2.662,1.014,0.998,1.066,1.038,1.088,1.258,1.024,1.044,1.055,1.037,1.085,0.992,1.415,1.059])
RHO=np.array([0.793,0.797,0.441,0.800,0.731,0.671,0.800,0.826,0.386,0.537,0.756,0.736,0.599,0.802,0.770,0.716,0.674,0.797,0.765,0.799])
Mp_K=np.where(B>0,np.sqrt(np.maximum(L**2-B**2,0)),L)          # M+/R*
tau_K=(Mp_K/S2)/L
# ---- suite P: recomputed run ----
R=sorted([json.loads(l) for l in open('/home/claude/pxB/P.jsonl')],key=lambda r:r['name'])
Rs=np.array([r['R'] for r in R]); L_P=np.array([r['L'] for r in R])/Rs
Mp_P=np.array([r['Mp'] for r in R])/Rs; RHO_P=np.array([r['rho'] for r in R])
tau_P=(Mp_P/S2)/L_P
def report(name,L,Mp,rho,tau,q2fun,maj1,G):
    q1=np.clip(np.interp(rho,G,maj1),1,S2)
    q2=np.array([q2fun(r,t) for r,t in zip(rho,tau)])
    f=lambda q: np.sqrt(np.maximum((L/q)**2-Mp**2,0))
    b1,b2,bc,bs2=f(q1),f(q2),f(Qcap(tau)),f(S2)
    u=lambda v: float(np.where(v<=1,v,0).mean())
    print('%s  n=%d   tau: min %.3f med %.3f max %.3f'%(name,len(L),tau.min(),np.median(tau),tau.max()))
    print('   guard         valid   usable   worst')
    for lab,v in [('1-D  B*',b1),('2-D  B*',b2),('Qcap(tau)',bc),('sqrt2',bs2)]:
        print('   %-12s %2d/%2d   %.4f   %.4f'%(lab,(v<=1).sum(),len(v),u(v),v.max()))
    return dict(q1=q1,q2=q2,b1=b1,b2=b2,bc=bc,bs2=bs2,tau=tau)
