"""Pre-registration audit of a known-truth benchmark package (run BEFORE any reader development).

Uses: observed files fully (label-free checks), oracle files ONLY on the train split for shortcut audits,
and never CAL/TEST outcomes.  Prints one JSON line per benchmark.
Usage: python3 -I audit_package.py PACKAGE_DIR
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

D = Path(sys.argv[1]); man = json.load(open(D / "manifest.json"))
for b in man["benchmarks"]:
    o = pd.read_csv(D / b["observed_file"]); q = pd.read_csv(D / b["oracle_file"]); assert (o.row_id == q.row_id).all()
    tr, te = o.split == "train", o.split == "test"
    nt, pt = pd.to_datetime(o.note_time_utc), pd.to_datetime(o.prediction_time_utc)
    leak = {c: float(np.abs(o[c] - q[k]).max()) for c in o.select_dtypes("number").columns for k in q.select_dtypes("number").columns
            if k not in ("seed",) and np.abs(o[c] - q[k]).max() < 1e-3}
    X = CountVectorizer(binary=True, min_df=5).fit_transform(o.note_text_signal[tr])
    auc = cross_val_score(LogisticRegression(max_iter=3000), X, q.latent_state[tr], cv=5, scoring="roc_auc").mean()
    rec = {"benchmark": b["name"], "observed_columns_equal_to_oracle_columns": leak,
           "notes_before_prediction": float((nt < pt).mean()), "midnight_timestamps": float((nt.dt.hour == 0).mean()),
           "split_rows": o.split.value_counts().to_dict(), "unique_notes": int(o.note_text_signal.nunique()),
           "test_notes_verbatim_in_train": float(o.note_text_signal[te].isin(set(o.note_text_signal[tr])).mean()),
           "null_is_within_cell_permutation": all(sorted(g.note_text_signal) == sorted(g.note_text_null_permuted_within_cell) for _, g in o.groupby("cell")),
           "train_bag_of_words_auc_for_latent": float(auc), "latent_levels": int(q.latent_state.nunique()),
           "signal_rate": float(q.true_text_signal_nonzero.mean())}
    print(json.dumps(rec))
