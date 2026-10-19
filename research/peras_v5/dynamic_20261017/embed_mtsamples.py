"""e5-small-v2 embeddings (mean of 512-token chunk embeddings, 'passage: ' prefix) for the MTSamples unique notes, in
the SAME shuffled order as run_afi_real.py.  Unsupervised -> outcome-blind geometry."""
import hashlib, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel
torch.set_num_threads(2)
df = pd.read_csv(__import__("os").environ.get("PERAS_DATA", "/home/claude/data") + "/mtsamples.csv").fillna("")
key = df.transcription.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
df = df[(key != "") & ~key.map(lambda s: hashlib.md5(s.encode()).hexdigest()).duplicated()]
df = df.sample(frac=1, random_state=20261017).reset_index(drop=True)
M = __import__("os").environ.get("PERAS_MODELS", "/home/claude/models") + "/e5-small-v2"; tok = AutoTokenizer.from_pretrained(M); mdl = AutoModel.from_pretrained(M).eval()
out = []
with torch.no_grad():
    for i, t in enumerate(df.transcription):
        ids = tok("passage: " + t, add_special_tokens=False)["input_ids"]
        chunks = [ids[s:s + 500] for s in range(0, max(len(ids), 1), 500)][:6]
        batch = tok.pad({"input_ids": [[tok.cls_token_id] + c + [tok.sep_token_id] for c in chunks]}, return_tensors="pt")
        h = mdl(**batch).last_hidden_state; m = batch["attention_mask"].unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1); e = torch.nn.functional.normalize(e, dim=-1).mean(0)
        out.append(torch.nn.functional.normalize(e, dim=0).numpy())
        if i % 300 == 0: print(i, flush=True)
np.save(__import__("os").environ.get("PERAS_REPO", __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), "../../.."))) + "/research/peras_v5/dynamic_20261017/mtsamples_e5.npy", np.array(out))
print("done", len(out))
