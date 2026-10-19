"""Median envelope qtil50(rho,tau) from the same 2-D library.
For a BOUND the right summary of the quadrature excess is an upper percentile;
for an ESTIMATOR it is the central one. The published estimator implicitly uses
q=1, which the library says is wrong in the unsafe direction on 77% of pairs."""
import numpy as np, json
rows=json.load(open('lib2.json'))
q=np.array([r[3] for r in rows]); rho=np.array([r[4] for r in rows]); tau=np.array([r[5] for r in rows])
RG=np.round(np.arange(0.20,1.0001,0.01),3); TG=np.round(np.arange(0.0,1.0001,0.01),3)
med=np.empty((len(RG),len(TG)))
for i,r in enumerate(RG):
    for j,t in enumerate(TG):
        h=0.05
        while True:
            m=(np.abs(rho-r)<=h)&(np.abs(tau-t)<=h)
            if m.sum()>=30 or h>1.5: break
            h+=0.01
        med[i,j]=np.median(q[m]) if m.sum() else np.nan
np.save('med2.npy',med)
def q50(r,t):
    i=int(np.clip(round((r-0.20)/0.01),0,len(RG)-1)); j=int(np.clip(round((t-0.0)/0.01),0,len(TG)-1))
    return float(med[i,j])
if __name__=='__main__':
    print('median envelope, qtil50(rho,tau):')
    print('%6s'%'rho'+''.join('%9s'%('tau=%.2f'%t) for t in [0.05,0.15,0.25,0.35,0.50]))
    for r in [0.45,0.55,0.67,0.75,0.80,0.85]:
        print('%6.2f'%r+''.join('%9.4f'%q50(r,t) for t in [0.05,0.15,0.25,0.35,0.50]))
