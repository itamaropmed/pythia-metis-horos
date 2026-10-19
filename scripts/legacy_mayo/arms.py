import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
D='/mnt/user-data/uploads/Clinical_notes/'
df=pd.read_parquet('df.parquet'); Xtab=pd.read_parquet('Xtab.parquet')
cid=df['case_durable_id'].values
# --- arm B: 11 note-derived LLM features ---
llm=pd.read_parquet(D+'data/mayo/mayo_hrs_llm_features_top11.parquet')
llm=llm.drop_duplicates('case_durable_id').set_index('case_durable_id')
Xllm=llm.reindex(cid).reset_index(drop=True)
cov=Xllm.notna().mean()
print('LLM feature coverage on the %d modelled cases:'%len(df)); print(cov.round(3).to_string())
Xllm.columns=['llm_'+c for c in Xllm.columns]
# --- arm C: note text -> tfidf -> svd ---
nb=pd.read_parquet(D+'mayo_residual_distiller_study/data/phrase_dir/note_bundles.parquet')
nb=nb.drop_duplicates('case_durable_id').set_index('case_durable_id')
txt=nb['note_text'].reindex(cid).fillna('').values
print('cases with note text: %d/%d (%.1f%%)'%((np.array([len(t) for t in txt])>0).sum(),len(txt),100*(np.array([len(t) for t in txt])>0).mean()))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
tf=TfidfVectorizer(max_features=60000,min_df=10,max_df=0.9,ngram_range=(1,2),sublinear_tf=True,strip_accents='unicode')
Z=tf.fit_transform(txt); print('tfidf',Z.shape)
sv=TruncatedSVD(n_components=100,random_state=0); S=sv.fit_transform(Z)
print('svd explained var %.3f'%sv.explained_variance_ratio_.sum())
Xsvd=pd.DataFrame(S,columns=['note_svd_%03d'%i for i in range(S.shape[1])])
nchar=pd.DataFrame({'note_chars':nb['note_chars'].reindex(cid).fillna(0).values,
                    'note_n_used':nb['n_pre_cutoff_notes_used'].reindex(cid).fillna(0).values})
A=Xtab
B=pd.concat([Xtab.reset_index(drop=True),Xllm],axis=1)
C=pd.concat([B,Xsvd,nchar],axis=1)
for n,X in [('A',A),('B',B),('C',C)]:
    X.astype(float).to_parquet('X%s.parquet'%n); print(n,X.shape)
