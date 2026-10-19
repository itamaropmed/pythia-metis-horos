"""Two-dimensional guard envelope q(rho, tau).

The published envelope is indexed by rho = MAD(U+V)/sd(U+V) alone, and every one
of the 496 library pairs was measured with BOTH components standardised, i.e. at
component scale ratio 1. But the quadrature excess also depends on how unequal
the components are, and that is observable:

    tau = MAD(V) / MAD(U+V)   ~   (M+/sqrt2) / MAD(r)

Two facts, one proved and one measured:
  (i)  MAD(U+V) <= MAD(U)+MAD(V) gives MAD(U) >= MAD(U+V)-MAD(V), hence
           q <= [ (1-tau)^2 + tau^2 ]^{-1/2}  =: Qcap(tau),
       which peaks at sqrt2 at tau=1/2 and falls away on both sides. This is a
       strict refinement of the universal sqrt2 cap.
  (ii) empirically q -> 1 as tau -> 0 much faster than Qcap.
"""
import numpy as np, itertools, json, time
from shapes import SHAPES, make, mad
N=1_000_000
TS=[1.0,0.8,0.65,0.5,0.4,0.3,0.22,0.16,0.11,0.07,0.04,0.02]
Qcap=lambda tau: 1.0/np.sqrt((1-tau)**2+tau**2)
if __name__=='__main__':
    rows=[]; t0=time.time()
    for k,(a,b) in enumerate(itertools.combinations_with_replacement(SHAPES,2)):
        U=make(a,np.random.default_rng(10_000+k),N); V0=make(b,np.random.default_rng(90_000+k),N)
        mu=mad(U); s0=mad(V0)
        for t in TS:
            V=V0*(t*mu/s0); mv=mad(V); W=U+V; mw=mad(W)
            rows.append((a,b,t,float(mw/np.hypot(mu,mv)),float(mw/W.std()),float(mv/mw)))
        if k%100==0: print(k,'%.0fs'%(time.time()-t0),flush=True)
    json.dump(rows,open('lib2.json','w'))
    q=np.array([r[3] for r in rows]); tau=np.array([r[5] for r in rows])
    print('grid points',len(rows))
    for lo,hi in [(0,.05),(.05,.10),(.10,.15),(.15,.20),(.20,.25),(.25,.35),(.35,.45),(.45,.55)]:
        m=(tau>=lo)&(tau<hi)
        if m.sum(): print('  tau[%.2f,%.2f) n=%5d  med %.4f  p95 %.4f  max %.4f  Qcap(%.2f)=%.4f'%(
            lo,hi,m.sum(),np.median(q[m]),np.percentile(q[m],95),q[m].max(),(lo+hi)/2,Qcap((lo+hi)/2)))
