"""Known-signal / known-noise plasmodes on real clinical notes (MTSamples) for nuisance-quotient replicate
certificates.  Each unique note is a 'base' with a planted label law p (binary or K=4); copies of the base are
re-issued with outcome-irrelevant nuisance (header with name/MRN/date, dates, abbreviations, whitespace/case,
section order).  The canonicaliser declares that nuisance group and collapses copies into exact replicate cells.

Scenarios
  T1  nuisance-only copies (quotient exact)
  T2  copy-forward addenda that shift p by at most eps0 in TV (declared slack eps; tipping curve)
  T3  canonicaliser wrongly also removes dose numbers that DO change p (invariance violated) -> audit must flag
Methods: raw exact cells vs canonical cells; accepted plug-in/min-count (hard-label averaging) vs pair
certificate (Thm A), stratified (Thm B, strata from a dev-split TF-IDF model), polynomial minorant (Thm C);
FeeBee-style 1-NN Cover-Hart estimate on TF-IDF (an estimate, not a bound)."""
import sys, os, re, json, hashlib, string
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from peras.core.replicate_pair_certificates import (cells_from_keys, cell_counts, pair_certificate,
    mincount_certificate, fit_g_minorant, choose_rule, linear_certificate, invariance_audit, f_K,
    within_cell_independence_audit)
NUM = re.compile(r"\d+(?:\.\d+)?")

df = pd.read_csv(__import__("os").environ.get("PERAS_DATA", "/home/claude/data") + "/mtsamples.csv").dropna(subset=["transcription"])
key = df.transcription.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
BASE = df.transcription[~key.map(lambda s: hashlib.md5(s.encode()).hexdigest()).duplicated()].reset_index(drop=True)
B = len(BASE); low = BASE.str.lower()
HDR = re.compile(r"([A-Z][A-Z /&-]{3,}:)")
DATE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
DOSE = re.compile(r"(\d+(?:\.\d+)?)\s?mg")
ABBR = [("hypertension", "HTN"), ("diabetes mellitus", "DM"), ("history of", "h/o"), ("shortness of breath", "SOB"),
        ("blood pressure", "BP"), ("coronary artery disease", "CAD")]

# ---- planted signal (uses dictionary-like terms AND out-of-dictionary channels: 'smok', length, dose values)
z = np.column_stack([low.str.contains("diabet"), low.str.contains("hypertens"), low.str.contains("smok"),
                     low.str.count(r"\d+\s?mg").clip(upper=5) / 5, (np.log(low.str.len()) - 7.7),
                     low.str.contains("pain"), low.str.contains("surgery|operative")]).astype(float)
dose_base = low.map(lambda s: np.mean([float(v) for v in DOSE.findall(s)]) if DOSE.search(s) else 0.0).to_numpy()
rng0 = np.random.default_rng(20261015)
beta_bin = np.array([1.6, 1.2, 1.4, 1.5, 1.1, -.9, -1.0]); lin_bin = z @ beta_bin - .4 + rng0.normal(0, 1.0, B)
beta_K = rng0.normal(0, 1.3, (z.shape[1], 4)); lin_K = z @ beta_K + rng0.normal(0, .9, (B, 4))


def softmax(a):
    e = np.exp(a - a.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def nuisance_copy(t, r):
    secs = [s for s in HDR.split(t)]
    head, rest = secs[0], secs[1:]
    pairs = [rest[i] + rest[i + 1] for i in range(0, len(rest) - 1, 2)]
    if len(pairs) > 1 and r.random() < .7:
        pairs = [pairs[i] for i in r.permutation(len(pairs))]
    t = head + "".join(pairs)
    t = DATE.sub(lambda m: f"{r.integers(1,13)}/{r.integers(1,29)}/{r.integers(2000,2025)}", t)
    for full, ab in ABBR:
        if r.random() < .5:
            t = re.sub(full, ab, t, flags=re.I)
    if r.random() < .5:
        t = t.replace(" ", "  ", int(r.integers(1, 20)))
    if r.random() < .3:
        t = t.upper()
    name = "".join(r.choice(list(string.ascii_uppercase), 6))
    hdr = f"Patient: {name}  MRN: {r.integers(10**6, 10**7)}  Date of service: {r.integers(1,13)}/{r.integers(1,29)}/2024\n"
    return hdr + t


def canonical(t, drop_numbers=False):
    t = re.sub(r"^Patient:.*?\n", "", t)
    t = "\n".join(l for l in t.split("\n") if not l.startswith("ADDENDUM:"))
    for full, ab in ABBR:
        t = re.sub(r"\b" + re.escape(ab) + r"\b", full, t, flags=re.I)
    secs = HDR.split(t.upper()); head, rest = secs[0], secs[1:]
    pairs = sorted(rest[i] + rest[i + 1] for i in range(0, len(rest) - 1, 2))
    t = (head + "".join(pairs)).lower()
    t = DATE.sub("<date>", t)
    if drop_numbers:
        t = re.sub(r"\d+(?:\.\d+)?", "<num>", t)
    return re.sub(r"\s+", " ", t).strip()


def raw_key(t):
    return re.sub(r"\s+", " ", t.lower()).strip()


def make_rows(bases, scen, r, K, eps0=.03):
    sizes = r.choice([1, 2, 3, 4, 6, 8], len(bases), p=[.35, .30, .15, .10, .06, .04])
    texts, P, bid, resid = [], [], [], []
    for b, m in zip(bases, sizes):
        for c in range(m):
            t = nuisance_copy(BASE[b], r) if r.random() < .8 else BASE[b]   # 20% verbatim re-issues
            lin = lin_bin[b] if K == 2 else lin_K[b].copy()
            rd = 0.0
            if scen == "T2" and r.random() < .5:
                other = BASE[int(r.integers(B))][:200].replace("\n", " ")
                t = t + "\nADDENDUM: " + other
                shift = r.uniform(-eps0, eps0)
            else:
                shift = 0.0
            if scen == "T3" and DOSE.search(t) and r.random() < .6:
                f = r.choice([.25, .5, 2., 4.])
                t = DOSE.sub(lambda m_: f"{float(m_.group(1))*f:g} mg", t)
                rd = np.log(f)
                lin = lin + 1.5 * rd if K == 2 else lin + 1.5 * rd * beta_K[3]
            if K == 2:
                p1 = 1 / (1 + np.exp(-lin)); p1 = float(np.clip(p1 + shift, 0, 1)); P.append([1 - p1, p1])
            else:
                pk = softmax(lin[None])[0]
                if shift:
                    pk = np.clip(pk + shift * (np.arange(4) == 0) - shift * (np.arange(4) == 1), 0, 1); pk /= pk.sum()
                P.append(pk)
            texts.append(t); bid.append(b); resid.append(rd)
    P = np.array(P); y = np.array([r.choice(len(p), p=p) for p in P])
    return texts, P, np.array(bid), np.array(resid), y


def run(scen, K, seed):
    r = np.random.default_rng(1000 * K + 7 * seed + {"T1": 0, "T2": 1, "T3": 2}[scen])
    perm = r.permutation(B); dev_b, cert_b = perm[:int(.4 * B)], perm[int(.4 * B):]
    Td, Pd, bd, _, yd = make_rows(dev_b, scen, r, K)
    Tc, Pc, bc, rc, yc = make_rows(cert_b, scen, r, K)
    gold = float((1 - Pc.max(1)).mean())
    drop = scen == "T3"
    Cd = [canonical(t, drop) for t in Td]; Cc = [canonical(t, drop) for t in Tc]
    raw_cells = cells_from_keys([raw_key(t) for t in Tc]); can_cells = cells_from_keys(Cc)
    slack = .06 if scen == "T2" else 0.0   # declared pairwise TV (two shifts of at most .03)
    out = dict(scen=scen, K=K, seed=seed, gold=gold, n=len(yc),
               n_raw_cells=int(raw_cells.max() + 1), n_can_cells=int(can_cells.max() + 1))
    a = .05
    out["raw_mincount"] = mincount_certificate(raw_cells, yc, K, a)["bound"]
    out["raw_pair"] = pair_certificate(raw_cells, yc, K, a)["bound"]
    mcc = mincount_certificate(can_cells, yc, K, a, slack=slack)
    out["can_mincount"] = mcc["bound"]; out["can_hardlabel_hoeffding"] = mcc["hardlabel_hoeffding"]
    out["indep_audit_p"] = within_cell_independence_audit(can_cells, yc, K)["p"]
    out["can_pair"] = pair_certificate(can_cells, yc, K, a, slack=slack)["bound"]
    # linear cell certificate (Thm C): minorant polynomials + per-size rule chosen on the development split
    cd = cell_counts(cells_from_keys(Cd), yd, K)
    coefs = {d: fit_g_minorant(cd, K, d) for d in (1, 2)}
    rule = choose_rule(cd, K, coefs, a / 2)
    cands = {"pair": pair_certificate(can_cells, yc, K, a / 2, slack=slack)["bound"],
             "linear": linear_certificate(can_cells, yc, K, rule, coefs, a / 2, slack=slack)["bound"]}
    out.update({"can_" + k: v for k, v in cands.items()}); out["rule"] = rule
    out["peras"] = max(cands.values())
    # FeeBee-style estimate: LOO 1-NN error on TF-IDF of raw text, Cover-Hart lower inversion
    Xr = TfidfVectorizer(min_df=3, max_features=20000, sublinear_tf=True).fit_transform(Tc)
    nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(Xr); _, idx = nn.kneighbors(Xr)
    e1 = float((yc[idx[:, 1]] != yc).mean())
    out["knn_estimate"] = float(f_K(min(e1, (K - 1) / K), K))
    if scen in ("T1", "T3") and seed < 6:
        nums = [frozenset(NUM.findall(t)) - frozenset(NUM.findall(c)) for t, c in zip(Tc, Cc)]   # what was removed
        jd = lambda i, j: 1 - len(nums[i] & nums[j]) / max(len(nums[i] | nums[j]), 1)
        out["audit_p"] = invariance_audit(can_cells, jd, yc, n_perm=99, seed=seed)["p"]
    if scen == "T2":
        out["tipping_delta_pair"] = tipping(can_cells, yc, K, a)
    return out


def tipping(cells, y, K, a):
    lo, hi = 0.0, .5
    for _ in range(30):
        mid = (lo + hi) / 2
        if pair_certificate(cells, y, K, a, slack=mid)["bound"] > 0: lo = mid
        else: hi = mid
    return lo


if __name__ == "__main__":
    reps = int(sys.argv[1]); scens = sys.argv[2].split(","); Ks = [int(k) for k in sys.argv[3].split(",")]
    res = []
    for scen in scens:
        for K in Ks:
            for s in range(reps):
                o = run(scen, K, s); res.append(o)
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()}), flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"text_quotient_{'_'.join(scens)}.json"), "w"), indent=1)
