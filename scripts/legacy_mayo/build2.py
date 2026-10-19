import numpy as np, json
S2=np.sqrt(2); Qcap=lambda t: 1.0/np.sqrt((1-t)**2+t**2)
rows=json.load(open('lib2.json'))
q=np.array([r[3] for r in rows]); rho=np.array([r[4] for r in rows]); tau=np.array([r[5] for r in rows])
RG=np.round(np.arange(0.20,1.0001,0.01),3); TG=np.round(np.arange(0.0,1.0001,0.01),3)
raw=np.empty((len(RG),len(TG)))
for i,r in enumerate(RG):
    for j,t in enumerate(TG):
        h=0.05
        while True:
            m=(np.abs(rho-r)<=h)&(np.abs(tau-t)<=h)
            if m.sum()>=30 or h>1.5: break
            h+=0.01
        raw[i,j]=np.percentile(q[m],95) if m.sum() else np.nan
# majorant: non-increasing in rho (sup over x>=rho), non-decreasing in tau (sup over u<=tau)
maj=np.array(raw)
for j in range(len(TG)):                      # sup over x >= rho
    run=-np.inf
    for i in range(len(RG)-1,-1,-1):
        if np.isfinite(maj[i,j]): run=max(run,maj[i,j])
        maj[i,j]=run
for i in range(len(RG)):                      # sup over u <= tau
    run=-np.inf
    for j in range(len(TG)):
        if np.isfinite(maj[i,j]): run=max(run,maj[i,j])
        maj[i,j]=run
maj=np.maximum(maj,1.0)
maj=np.minimum(maj,Qcap(TG)[None,:])          # provable cap, per tau
np.save('env2.npy',maj); np.save('env2_axes.npy',np.array([RG[0],RG[-1],len(RG),TG[0],TG[-1],len(TG)]))
def q2(r,t):
    i=int(np.clip(round((r-RG[0])/0.01),0,len(RG)-1)); j=int(np.clip(round((t-TG[0])/0.01),0,len(TG)-1))
    return float(maj[i,j])
if __name__=='__main__':
    print('2-D envelope built. Slices (rho -> qhat) at fixed tau:')
    print('%6s'%'rho'+''.join('%9s'%('tau=%.2f'%t) for t in [0.05,0.10,0.15,0.20,0.25,0.35,0.50]))
    for r in [0.40,0.50,0.60,0.67,0.70,0.75,0.80,0.85]:
        print('%6.2f'%r+''.join('%9.4f'%q2(r,t) for t in [0.05,0.10,0.15,0.20,0.25,0.35,0.50]))
    print('\nMayo arm B query: rho=0.669 tau=0.230 -> qhat2 = %.4f   (1-D was 1.1611, Qcap=%.4f)'%(q2(0.669,0.230),Qcap(0.230)))
