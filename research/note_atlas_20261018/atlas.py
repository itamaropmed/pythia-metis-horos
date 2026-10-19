"""Note Information Atlas: a broad, dynamic analysis of what clinical free text carries about an outcome
BEYOND a table, at the corpus, section, token-class, case, network and geometric level.

Input
  texts      : list[str] (one note bundle per case)
  patient    : patient id per case (all folds are patient-grouped)
  r          : outcome residual after a cross-fitted TABLE model (the information target is r; anything the
               table already explains is excluded by construction)
  strata     : optional coarse table variable (e.g. procedure family) for stratified summaries
Lenses
  A structure        corpus anatomy: length, sections, boilerplate, copy-forward, numbers, negation
  B global info      cross-fitted text->residual model: I = E r^2 - E (r - g)^2 (minutes^2), R^2|table, dMAE
  C per-case info    pointwise movement |g_i| and gain_i; concentration (Gini, top-10% share), by stratum
  D section masking  drop-section (reliance), keep-only (sufficiency), leave-section-out retrain (unique info)
  E token classes    mask numbers / negation scopes / medication-like / units / dates-times
  F melting curve    "temperature" = random token-dropout rate p; I(p), heat capacity -dI/dp, half-melting p_1/2
  G metaheuristics   beam search for a globally sufficient section set; simulated annealing for minimal
                     per-case explanations (how many sections a case's information lives in)
  H diffusion        heat diffusion of the residual over the case kNN graph (leave-self-out random walk):
                     information as a function of diffusion time t; Louvain note communities
  I term network     co-occurrence graph of frequent terms; residual association diffused over it;
                     term communities ranked by information
  J topology         MST (H0 persistence) of note space; Dirichlet energy of r along the MST vs permutation null
Privacy: only aggregates; a term is reportable only if it occurs for >= min_patients_term distinct patients and
passes a proper-noun / title-adjacency filter; section headers need >= min_patients_header patients.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

HEADER_RE = re.compile(r"(?:(?<=[\n\.,;])|^)\s*([A-Z][A-Z0-9 /&\-\(\)]{2,48}[A-Z\)])\s*:")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']+|\d+(?:\.\d+)?")
NEG_CUES = {"no", "not", "denies", "denied", "without", "negative", "absent", "none", "never", "nor"}
MED_SUFFIX = ("olol", "pril", "sartan", "statin", "parin", "xaban", "gatran", "azole", "mycin", "cillin", "floxacin",
              "dipine", "amide", "oxin", "tidine", "prazole", "lukast", "afil", "umab", "tinib")
MED_WORDS = {"warfarin", "aspirin", "insulin", "heparin", "metformin", "amiodarone", "digoxin", "clopidogrel",
             "furosemide", "lasix", "coumadin", "eliquis", "xarelto", "plavix", "prednisone", "morphine", "fentanyl"}
UNITS = {"mg", "mcg", "ml", "cc", "mmhg", "kg", "cm", "mm", "units", "bpm", "min", "minutes", "hours", "hrs", "%", "lbs"}
DATE_RE = re.compile(r"\b\d{1,2}[/\-]\d{1,2}(?:[/\-]\d{2,4})?\b|\b\d{1,2}:\d{2}\b|\b(?:19|20)\d{2}\b")
NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
TITLES = {"dr", "md", "mr", "mrs", "ms", "do", "rn", "np", "pa"}


# ---------------------------------------------------------------- text utilities
def norm_header(h: str) -> str:
    return re.sub(r"\s+", " ", h.strip().upper())


def segment(text: str) -> list[tuple[str, str]]:
    """[(header, body)], first chunk has header '_PREAMBLE'."""
    out, last, head = [], 0, "_PREAMBLE"
    for m in HEADER_RE.finditer(text):
        out.append((head, text[last:m.start()]))
        head, last = norm_header(m.group(1)), m.end()
    out.append((head, text[last:]))
    return [(h, b) for h, b in out if b.strip() or h != "_PREAMBLE"]


def mask_sections(text: str, drop: set | None = None, keep: set | None = None) -> str:
    parts = []
    for h, b in segment(text):
        if drop is not None and h in drop:
            continue
        if keep is not None and h not in keep:
            continue
        parts.append(("" if h == "_PREAMBLE" else h + ": ") + b)
    return " ".join(parts)


def mask_tokens(text: str, kind: str) -> str:
    if kind == "numbers":
        return NUM_RE.sub(" ", text)
    if kind == "dates_times":
        return DATE_RE.sub(" ", text)
    toks = text.split()
    if kind == "negation_scope":
        out, k = [], 0
        for w in toks:
            lw = w.lower().strip(".,;:()")
            if lw in NEG_CUES:
                k = 5; continue
            if k > 0:
                k -= 1; continue
            out.append(w)
        return " ".join(out)
    if kind == "negation_cues_only":          # remove the cue but keep the scope: flips meaning for a BoW model
        return " ".join(w for w in toks if w.lower().strip(".,;:()") not in NEG_CUES)
    if kind == "medication_like":
        return " ".join(w for w in toks if not (w.lower().strip(".,;:()") in MED_WORDS or
                                                 w.lower().strip(".,;:()").endswith(MED_SUFFIX)))
    if kind == "units":
        return " ".join(w for w in toks if w.lower().strip(".,;:()") not in UNITS)
    raise ValueError(kind)


def random_dropout(text: str, p: float, rng) -> str:
    toks = text.split()
    if not toks or p <= 0:
        return text
    keep = rng.random(len(toks)) >= p
    return " ".join(t for t, k in zip(toks, keep) if k)


# ---------------------------------------------------------------- reportability filter (privacy)
def reportable_terms(texts, patient, min_patients=50):
    """Lower-case terms seen for >= min_patients distinct patients, mostly NOT capitalised mid-sentence
    (proper-noun heuristic) and never directly after a title (dr, mr, ...)."""
    pats = defaultdict(set); cap = Counter(); tot = Counter(); titled = Counter()
    for t, p in zip(texts, patient):
        prev = ""
        for m in TOKEN_RE.finditer(t):
            w = m.group(0); lw = w.lower()
            if lw.isdigit():
                prev = lw; continue
            tot[lw] += 1
            if w[0].isupper() and not w.isupper():
                cap[lw] += 1
            if prev in TITLES:
                titled[lw] += 1
            pats[lw].add(p); prev = lw
    return {w for w, s in pats.items() if len(s) >= min_patients and cap[w] / max(tot[w], 1) < .5 and titled[w] == 0
            and len(w) > 2}


# ---------------------------------------------------------------- models
class TextResidualModel:
    """Patient-grouped cross-fitted TF-IDF -> ridge model of the table residual (Horos-side fitting)."""

    def __init__(self, folds=5, alpha=None, max_features=60000, min_df=5, seed=0):
        self.folds, self.alpha, self.max_features, self.min_df, self.seed = folds, alpha, max_features, min_df, seed

    def fit(self, texts, r, patient):
        self.split = list(GroupKFold(self.folds).split(texts, r, patient))
        self.vec, self.reg = [], []
        for tr, _ in self.split:
            v = TfidfVectorizer(ngram_range=(1, 2), min_df=self.min_df, max_features=self.max_features,
                                sublinear_tf=True, token_pattern=r"(?u)\b[A-Za-z][A-Za-z\-']+\b|\b\d+(?:\.\d+)?\b")
            Xtr = v.fit_transform([texts[i] for i in tr])
            a = self.alpha or self._choose_alpha(Xtr, r[tr], patient[tr])
            self.vec.append(v); self.reg.append(Ridge(alpha=a).fit(Xtr, r[tr]))
        return self

    def _choose_alpha(self, X, y, groups):
        best = None
        for a in (1.0, 3.0, 10.0, 30.0):
            err = 0.0
            for tr, va in GroupKFold(3).split(X, y, groups):
                err += ((Ridge(alpha=a).fit(X[tr], y[tr]).predict(X[va]) - y[va]) ** 2).sum()
            best = (err, a) if best is None or err < best[0] else best
        return best[1]

    def predict(self, texts):
        """Cross-fitted prediction: case i uses the model of the fold in which it was held out."""
        g = np.zeros(len(texts))
        for (tr, va), v, m in zip(self.split, self.vec, self.reg):
            g[va] = m.predict(v.transform([texts[i] for i in va]))
        return g


def info(r, g):
    """Information in minutes^2 the text adds beyond the table: E r^2 - E (r - g)^2, plus R^2 and dMAE."""
    i = float((r ** 2).mean() - ((r - g) ** 2).mean())
    return dict(I=i, R2=i / float((r ** 2).mean()), dMAE=float(np.abs(r).mean() - np.abs(r - g).mean()))


def cluster_boot(gain, patient, B=500, seed=0):
    rng = np.random.default_rng(seed); u, inv = np.unique(patient, return_inverse=True)
    s = np.bincount(inv, weights=gain); c = np.bincount(inv)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, len(u), len(u)); vals.append(s[idx].sum() / c[idx].sum())
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def gini(x):
    x = np.sort(np.clip(np.asarray(x, float), 0, None)); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if x.sum() > 0 else 0.0


# ---------------------------------------------------------------- the atlas
def run_atlas(texts, patient, r, strata=None, *, min_patients_term=50, min_patients_header=200, top_sections=12,
              seed=0, truth=None, lenses="ABCDEFGHIJ", verbose=False, model=None, n_null=2, unique_retrain=5,
              alpha=None, max_features=60000):
    texts = [t if isinstance(t, str) else "" for t in texts]
    patient = np.asarray(patient).astype(str); r = np.asarray(r, float); n = len(texts)
    rng = np.random.default_rng(seed); out = {"n_cases": n, "n_patients": int(len(np.unique(patient)))}
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    segs = [segment(t) for t in texts]
    hdr_pat = defaultdict(set)
    for s, p in zip(segs, patient):
        for h, _ in s:
            hdr_pat[h].add(p)
    sections = [h for h, ps in sorted(hdr_pat.items(), key=lambda kv: -len(kv[1]))
                if len(ps) >= min_patients_header and h != "_PREAMBLE"][:top_sections]
    rep = reportable_terms(texts, patient, min_patients_term)

    # ---- A structure
    if "A" in lenses:
        lens = np.array([len(t) for t in texts]); toks = [t.split() for t in texts]
        line_pat = defaultdict(set); per_pat_lines = defaultdict(Counter)
        for t, p in zip(texts, patient):
            for ln in set(x.strip().lower() for x in re.split(r"[\n\.]", t) if len(x.strip()) > 25):
                line_pat[hash(ln)].add(p); per_pat_lines[p][hash(ln)] += 1
        boiler = np.mean([np.mean([len(line_pat[hash(ln)]) >= 20 for ln in set(x.strip().lower() for x in re.split(r"[\n\.]", t) if len(x.strip()) > 25)] or [0]) for t in texts])
        copyfwd = np.mean([np.mean([c > 1 for c in cnt.values()]) for cnt in per_pat_lines.values() if sum(cnt.values()) > 0])
        out["A_structure"] = dict(
            empty_share=float(np.mean(lens == 0)), chars_median=float(np.median(lens)), chars_p90=float(np.quantile(lens, .9)),
            tokens_median=float(np.median([len(x) for x in toks])),
            sections_per_case_median=float(np.median([len(s) for s in segs])),
            top_sections={h: len(hdr_pat[h]) for h in sections},
            boilerplate_line_share=float(boiler), copy_forward_line_share_within_patient=float(copyfwd),
            numbers_per_1k_tokens=float(1000 * np.mean([len(NUM_RE.findall(t)) / max(len(x), 1) for t, x in zip(texts, toks)])),
            negation_cues_per_1k_tokens=float(1000 * np.mean([sum(w.lower() in NEG_CUES for w in x) / max(len(x), 1) for x in toks])),
            reportable_vocabulary=len(rep))
        log("A done")

    # ---- B global info
    model = model or TextResidualModel(seed=seed, alpha=alpha, max_features=max_features).fit(texts, r, patient)
    out["_model"] = model
    g = model.predict(texts)
    base = info(r, g); gain = r ** 2 - (r - g) ** 2
    out["B_global"] = dict(**base, I_ci95=cluster_boot(gain, patient), residual_var=float((r ** 2).mean()),
                           residual_mae=float(np.abs(r).mean()))
    null = []
    for b in range(n_null):                                # outcome-permutation null of the whole pipeline
        rp = r[rng.permutation(n)]
        null.append(info(rp, TextResidualModel(seed=seed + b, alpha=model.reg[0].alpha, max_features=max_features)
                         .fit(texts, rp, patient).predict(texts))["I"])
    out["B_global"]["permutation_null_I"] = null
    log("B done", base)

    # ---- C per-case
    if "C" in lenses:
        mv = np.abs(g); order = np.argsort(-gain)
        top10 = float(gain[order[: max(1, n // 10)]].sum() / max(gain[gain > 0].sum(), 1e-12))
        out["C_per_case"] = dict(
            movement_minutes_quantiles={q: float(np.quantile(mv, q)) for q in (.1, .25, .5, .75, .9, .99)},
            share_cases_positive_gain=float(np.mean(gain > 0)), gini_positive_gain=gini(gain),
            top10pct_share_of_positive_gain=top10,
            by_length_decile=[dict(decile=int(d), **info(r[m], g[m])) for d, m in
                              ((d, np.digitize([len(t) for t in texts], np.quantile([len(t) for t in texts], np.linspace(.1, .9, 9))) == d)
                               for d in range(10)) if m.sum() > 30],
            empty_notes=info(r[np.array([len(t) == 0 for t in texts])], g[np.array([len(t) == 0 for t in texts])])
            if any(len(t) == 0 for t in texts) else None)
        if strata is not None:
            st = np.asarray(strata).astype(str)
            out["C_per_case"]["by_stratum"] = {s: dict(n=int((st == s).sum()), **info(r[st == s], g[st == s]))
                                               for s in np.unique(st) if (st == s).sum() >= 100}
        if truth is not None:
            from scipy.stats import spearmanr
            out["C_per_case"]["spearman_movement_vs_true_effect"] = float(spearmanr(g, truth).correlation)
        log("C done")

    # ---- D sections
    if "D" in lenses:
        D = {}
        for h in sections:
            drop = info(r, model.predict([mask_sections(t, drop={h}) for t in texts]))
            keep = info(r, model.predict([mask_sections(t, keep={h}) for t in texts]))
            D[h] = dict(patients=len(hdr_pat[h]), reliance_drop_I=base["I"] - drop["I"], keep_only_I=keep["I"])
        # leave-section-out retrain for the top 5 by reliance (unique information)
        for h in sorted(D, key=lambda k: -D[k]["reliance_drop_I"])[:unique_retrain]:
            mt = [mask_sections(t, drop={h}) for t in texts]
            D[h]["unique_I_retrain"] = base["I"] - info(r, TextResidualModel(seed=seed, alpha=model.reg[0].alpha, max_features=max_features)
                                                        .fit(mt, r, patient).predict(mt))["I"]
        out["D_sections"] = D
        log("D done")

    # ---- E token classes
    if "E" in lenses:
        out["E_token_classes"] = {k: base["I"] - info(r, model.predict([mask_tokens(t, k) for t in texts]))["I"]
                                  for k in ("numbers", "dates_times", "negation_scope", "negation_cues_only",
                                            "medication_like", "units")}
        log("E done")

    # ---- F melting curve (temperature = token dropout rate)
    if "F" in lenses:
        ps = [0, .1, .25, .5, .75, .9, .97]
        curve = [info(r, model.predict([random_dropout(t, p, rng) for t in texts]))["I"] for p in ps]
        frac = [c / base["I"] if base["I"] > 0 else 0 for c in curve]
        heat = [-(frac[i + 1] - frac[i]) / (ps[i + 1] - ps[i]) for i in range(len(ps) - 1)]
        half = next((ps[i] for i in range(len(ps)) if frac[i] <= .5), None)
        out["F_melting"] = dict(p=ps, retained_fraction=frac, heat_capacity=heat, half_melting_p=half)
        log("F done")

    # ---- G metaheuristics
    if "G" in lenses and sections:
        # beam search (width 3) for a small section set that, kept alone, retains most information
        beam = [(0.0, frozenset())]
        trace = []
        for step in range(min(5, len(sections))):
            cand = {}
            for _, S in beam:
                for h in sections:
                    if h in S:
                        continue
                    T = S | {h}
                    if T not in cand:
                        cand[T] = info(r, model.predict([mask_sections(t, keep=set(T)) for t in texts]))["I"]
            beam = sorted(((v, T) for T, v in cand.items()), key=lambda x: -x[0])[:3]
            trace.append(dict(size=step + 1, best_sections=sorted(beam[0][1]), retained=beam[0][0] / base["I"] if base["I"] > 0 else 0))
        out["G_beam"] = trace
        # simulated annealing: per-case minimal section set that keeps the prediction within 20% of the full
        # note's movement (objective = #sections + 10 * infeasible; Metropolis acceptance, geometric cooling)
        sizes, avail = [], []
        idx = np.argsort(-np.abs(g))[: min(200, n)]
        fold_of = np.zeros(n, int)
        for k, (_, va) in enumerate(model.split):
            fold_of[va] = k
        for i in idx:
            hs = sorted({h for h, _ in segs[i]})
            if len(hs) < 2:
                continue
            v, m = model.vec[fold_of[i]], model.reg[fold_of[i]]; full = g[i]
            def cost(S):
                if not S:
                    return 1e9
                ok = abs(m.predict(v.transform([mask_sections(texts[i], keep=S)]))[0] - full) <= .2 * abs(full)
                return len(S) + (0 if ok else 10)
            S = set(hs); c = cost(S); best = len(S); T = 2.0
            for it in range(80):
                S2 = S ^ {hs[rng.integers(len(hs))]}
                c2 = cost(S2)
                if c2 <= c or rng.random() < math.exp(-(c2 - c) / T):
                    S, c = S2, c2
                    if c < 10:
                        best = min(best, len(S))
                T *= .95
            sizes.append(best); avail.append(len(hs))
        out["G_annealing"] = dict(cases=len(sizes), median_sections_needed=float(np.median(sizes)) if sizes else None,
                                  median_sections_available=float(np.median(avail)) if avail else None,
                                  share_single_section=float(np.mean(np.array(sizes) == 1)) if sizes else None)
        log("G done")

    # ---- H diffusion on the case graph
    if "H" in lenses or "J" in lenses:
        Xs = TfidfVectorizer(min_df=5, sublinear_tf=True, max_features=60000).fit_transform(texts)
        E = TruncatedSVD(100, random_state=seed).fit_transform(Xs); E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    if "H" in lenses:
        k = 15
        nn = NearestNeighbors(n_neighbors=k + 1).fit(E); _, ind = nn.kneighbors(E)
        rows = np.repeat(np.arange(n), k); cols = ind[:, 1:].ravel()
        same = patient[rows] == patient[cols]                       # never diffuse a patient's own outcome
        W = csr_matrix((np.where(same, 0.0, 1.0), (rows, cols)), shape=(n, n)); W = W.maximum(W.T)
        deg = np.asarray(W.sum(1)).ravel(); P = csr_matrix(W.multiply(1 / np.maximum(deg, 1e-12)[:, None]))
        # leakage-free diffusion: cross-fitted by patient folds -- only TRAINING-fold residuals are diffused,
        # normalised by the diffused training mass, and read off at validation nodes
        diff = []
        preds = {t: np.zeros(n) for t in range(1, 9)}
        for tr, va in GroupKFold(5).split(E, r, patient):
            src = np.zeros(n); src[tr] = r[tr]; mass = np.zeros(n); mass[tr] = 1.0
            x, mm = src, mass
            for t in range(1, 9):
                x, mm = P @ x, P @ mm
                preds[t][va] = (x / np.maximum(mm, 1e-12))[va]
        for t in range(1, 9):
            gt = preds[t]; beta = float(np.dot(gt, r) / max(np.dot(gt, gt), 1e-12))
            diff.append(dict(t=t, I=info(r, beta * gt)["I"], shrink=beta))
        out["H_diffusion"] = dict(curve=diff, best_t=max(diff, key=lambda d: d["I"])["t"])
        try:
            import networkx as nx
            G = nx.from_scipy_sparse_array(W)
            comms = nx.community.louvain_communities(G, seed=seed, resolution=1.0)
            lab = np.zeros(n, int)
            for c, members in enumerate(comms):
                lab[list(members)] = c
            sizes = np.bincount(lab); big = [c for c in range(len(comms)) if sizes[c] >= 50]
            # between-community information, cross-fitted by patient folds (community means from other folds)
            gc = np.zeros(n)
            for tr, va in GroupKFold(5).split(E, r, patient):
                means = {c: r[tr][lab[tr] == c].mean() if (lab[tr] == c).sum() else 0 for c in np.unique(lab)}
                gc[va] = [means[c] for c in lab[va]]
            top_terms = {}
            vec = TfidfVectorizer(min_df=5, sublinear_tf=True, vocabulary=sorted(rep)) if rep else None
            if vec is not None:
                Xr = vec.fit_transform(texts); vocab = np.array(vec.get_feature_names_out())
                for c in big:
                    m = lab == c; lift = np.asarray(Xr[m].mean(0)).ravel() - np.asarray(Xr[~m].mean(0)).ravel()
                    top_terms[int(c)] = vocab[np.argsort(-lift)[:8]].tolist()
            out["H_communities"] = dict(n_communities=len(comms), n_large=len(big), community_info=info(r, gc),
                                        large=[dict(id=int(c), size=int(sizes[c]), mean_residual=float(r[lab == c].mean()),
                                                    top_terms=top_terms.get(int(c), [])) for c in big])
        except Exception as e:                                      # networkx missing etc.
            out["H_communities"] = dict(error=str(e))
        log("H done")

    # ---- I term network
    if "I" in lenses and rep:
        vec = TfidfVectorizer(vocabulary=sorted(rep), binary=True, use_idf=False, norm=None, lowercase=True)
        B = vec.fit_transform(texts).tocsc(); vocab = np.array(vec.get_feature_names_out())
        df_ = np.asarray(B.sum(0)).ravel()
        keep = np.argsort(-df_)[:1500]; B = B[:, keep]; vocab = vocab[keep]; df_ = df_[keep]
        rc = r - r.mean()
        assoc = np.asarray(B.T @ rc).ravel() / np.maximum(df_, 1)        # mean residual among docs with the term
        C = (B.T @ B).toarray().astype(float); np.fill_diagonal(C, 0)
        pmi = np.log(np.maximum(C * n, 1e-12) / np.maximum(np.outer(df_, df_), 1e-12)); A = np.where((C >= 20) & (pmi > .5), pmi, 0)
        Dg = A.sum(1); Pn = A / np.maximum(Dg[:, None], 1e-12)
        s = assoc.copy()
        for _ in range(3):
            s = .5 * assoc + .5 * Pn @ s                              # personalised diffusion of residual association
        try:
            import networkx as nx
            Gt = nx.from_numpy_array(A)
            tcom = nx.community.louvain_communities(Gt, seed=seed, weight="weight")
            comm = []
            for c in tcom:
                c = list(c)
                if len(c) < 5:
                    continue
                score = float(np.average(s[c], weights=df_[c]))
                comm.append(dict(size=len(c), diffused_minutes=score,
                                 top_terms=vocab[np.array(c)[np.argsort(-np.abs(s[c]) * np.sqrt(df_[c]))[:10]]].tolist()))
            comm.sort(key=lambda d: -abs(d["diffused_minutes"]))
            out["I_term_network"] = dict(terms=len(vocab), edges=int((A > 0).sum() // 2), communities=comm[:12],
                                         top_positive_terms=vocab[np.argsort(-s)[:15]].tolist(),
                                         top_negative_terms=vocab[np.argsort(s)[:15]].tolist())
        except Exception as e:
            out["I_term_network"] = dict(error=str(e))
        log("I done")

    # ---- J topology: MST / H0 persistence + Dirichlet energy of r on the MST
    if "J" in lenses:
        from scipy.sparse.csgraph import minimum_spanning_tree
        k = 10
        nn = NearestNeighbors(n_neighbors=k + 1).fit(E); dist, ind = nn.kneighbors(E)
        Wd = csr_matrix((dist[:, 1:].ravel() + 1e-9, (np.repeat(np.arange(n), k), ind[:, 1:].ravel())), shape=(n, n))
        T = minimum_spanning_tree(Wd.maximum(Wd.T)).tocoo()
        diff_patient = patient[T.row] != patient[T.col]
        e = ((r[T.row] - r[T.col]) ** 2)[diff_patient].mean()
        nulls = []
        for _ in range(200):
            rp = r[rng.permutation(n)]; nulls.append(((rp[T.row] - rp[T.col]) ** 2)[diff_patient].mean())
        out["J_topology"] = dict(
            mst_edges=int(T.nnz), h0_death_quantiles={q: float(np.quantile(T.data, q)) for q in (.5, .9, .99)},
            dirichlet_energy=float(e), null_mean=float(np.mean(nulls)), smoothness_ratio=float(e / np.mean(nulls)),
            permutation_p=float((1 + sum(x <= e for x in nulls)) / 201))
        log("J done")
    out["reportable_terms_count"] = len(rep); out["sections_analysed"] = sections
    return out
