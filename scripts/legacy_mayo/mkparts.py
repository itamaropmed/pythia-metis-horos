import numpy as np, pandas as pd, json, time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import networkx as nx
df=pd.read_parquet('df.parquet'); y=df['case_actual_duration_time'].astype(float).values
P=np.load('PB.npy'); best=P[int(np.abs(y[None,:]-P).mean(1).argmin())]
XC=pd.read_parquet('XC.parquet')
svd=XC[[c for c in XC.columns if c.startswith('note_svd_')]].values
has=(np.abs(svd).sum(1)>1e-9)
parts={}
parts['P1 predicted-duration quintiles']=np.asarray(pd.qcut(best,5,labels=False,duplicates='drop')).astype(str).tolist()
parts['P2 procedure family']=df['primary_procedure_name'].astype(str).tolist()
Z=StandardScaler().fit_transform(svd[has]); km=KMeans(8,n_init=10,random_state=0).fit(Z)
l3=np.array(['nonote']*len(y),dtype=object); l3[has]=np.array(['note_c%d'%i for i in km.labels_])
parts['P3 note-content communities (k-means, 8)']=l3.tolist()
t=time.time()
F=np.hstack([StandardScaler().fit_transform(svd),StandardScaler().fit_transform(best[:,None])])
_,idx=NearestNeighbors(n_neighbors=16).fit(F).kneighbors(F); np.save('knnidx.npy',idx)
G=nx.Graph(); G.add_nodes_from(range(len(y)))
G.add_edges_from((int(i),int(j)) for i in range(len(y)) for j in idx[i,1:])
comm=nx.community.louvain_communities(G,seed=0)
l4=np.empty(len(y),dtype=object)
for ci,c in enumerate(sorted(comm,key=len,reverse=True)):
    for i in c: l4[i]='knn_c%02d'%ci
parts['P4 kNN-graph communities (Louvain)']=l4.tolist()
print('louvain: %d communities, sizes %s, modularity %.3f, %.0fs'%(
    len(comm),sorted([len(c) for c in comm],reverse=True)[:10],nx.community.modularity(G,comm),time.time()-t))
parts['P5 anesthesia x urgency']=(df['case_anesthesia'].astype(str)+' | '+df['case_urgency_level'].astype(str)).tolist()
json.dump(parts,open('parts.json','w'))
print('saved', {k:len(set(v)) for k,v in parts.items()})
