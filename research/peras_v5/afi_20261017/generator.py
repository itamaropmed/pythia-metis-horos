"""Ten generated table+text datasets with EXACT known outcome laws p(Y | table, text).

Each dataset uses its own pseudo-vocabulary and writing style (so the ten are out-of-distribution w.r.t. each other):
different word lists, note lengths, boilerplate amounts, word-order shuffling, abbreviation rates.
Table: age band (4) x sex (2) x lab band (3) = 24 cells.  Latent binary findings Z_j ~ Bern(q_j(table)) are
rendered in the note: positive mention w.p. pi_j, explicit negation w.p. nu_j, otherwise silent; posterior of each
Z_j given (table, note) is exact, hence p(Y | table, note) = sum_z P(z | table, note) sigmoid(f(table, z)) exactly.
"""
import numpy as np, string

MECH = ["D01_echo_table_only", "D02_novel_findings", "D03_negation", "D04_synergy", "D05_boilerplate",
        "D06_partial_mentions", "D07_numeric_in_text", "D08_outcome_leak", "D09_multiclass", "D10_style_confound"]


def _words(r, k, lo=3, hi=9):
    return ["".join(r.choice(list(string.ascii_lowercase), int(r.integers(lo, hi)))) for _ in range(k)]


def make(name, n, seed):
    r = np.random.default_rng(seed); ds = np.random.default_rng(1000 + MECH.index(name))    # dataset style is fixed
    V = _words(ds, 400); FIND = _words(ds, 6, 6, 11); NEG = _words(ds, 2, 3, 6); POS = _words(ds, 2, 3, 6)
    style = dict(len_noise=int(ds.integers(8, 40)), shuffle=bool(ds.random() < .3), abbrev=float(ds.random() * .5))
    if name == "D05_boilerplate":
        style["len_noise"] = 220
    ageb = r.integers(0, 4, n); sex = r.integers(0, 2, n); lab = r.integers(0, 3, n)
    cell = ageb * 6 + sex * 3 + lab
    k = 4; q = np.clip(.15 + .1 * ageb[:, None] * (np.arange(k) % 2 == 0) + .15 * lab[:, None] * (np.arange(k) >= 2), .05, .9)
    if name == "D10_style_confound":
        q = np.full((n, k), .25)
    pi = np.full(k, .9); nu = np.full(k, .0)
    if name == "D03_negation":
        nu = np.full(k, .9)
    if name == "D06_partial_mentions":
        pi = np.full(k, .5)
    Z = (r.random((n, k)) < q).astype(int)
    tab = np.column_stack([ageb - 1.5, sex - .5, lab - 1.0])
    K = 4 if name == "D09_multiclass" else 2

    def logit(z):            # z: (n, k) candidate latent configuration
        base = .6 * tab[:, 0] + .5 * tab[:, 1] + .7 * tab[:, 2]
        if name == "D01_echo_table_only" or name == "D07_numeric_in_text" or name == "D08_outcome_leak":
            return base
        if name == "D04_synergy":
            return .2 * base + 2.2 * (2 * sex - 1) * (z[:, 0] - .5) + 1.6 * (ageb - 1.5) * (z[:, 1] - .5)
        if name == "D10_style_confound":
            return base + 1.2 * z[:, 0]
        return .5 * base + 1.6 * z[:, 0] + 1.3 * z[:, 1] - 1.1 * z[:, 2] + .9 * z[:, 3] - .8

    # exact posterior over Z given (table, note): independent features
    mention = (r.random((n, k)) < pi) & (Z == 1); deny = (r.random((n, k)) < nu) & (Z == 0)
    post = np.where(mention, 1.0, np.where(deny, 0.0, q * (1 - pi) / (q * (1 - pi) + (1 - q) * (1 - nu))))
    numval = None
    if name == "D07_numeric_in_text":            # a lab value only in the text, written as a number
        numval = np.round(r.gamma(4, .3, n), 1)
    if K == 2:
        p1 = np.zeros(n)
        for bits in range(2 ** k):
            z = np.array([(bits >> j) & 1 for j in range(k)])[None, :].repeat(n, 0)
            pz = np.prod(np.where(z == 1, post, 1 - post), axis=1)
            lg = logit(z) + (1.4 * (numval - 1.2) if numval is not None else 0)
            p1 += pz * 1 / (1 + np.exp(-lg))
        P = np.column_stack([1 - p1, p1])
    else:
        W = np.random.default_rng(5).normal(0, 1.2, (3 + k, K)); P = np.zeros((n, K))
        for bits in range(2 ** k):
            z = np.array([(bits >> j) & 1 for j in range(k)])[None, :].repeat(n, 0)
            pz = np.prod(np.where(z == 1, post, 1 - post), axis=1)
            A = np.column_stack([tab, z]) @ W; e = np.exp(A - A.max(1, keepdims=True)); P += pz[:, None] * e / e.sum(1, keepdims=True)
    y = np.array([r.choice(K, p=p) for p in P]) if K > 2 else (r.random(n) < P[:, 1]).astype(int)
    leak_flag = None
    if name == "D08_outcome_leak":               # text reveals the outcome w.p. .3 (generated FROM y)
        leak_flag = r.random(n) < .3
        P = np.where(leak_flag[:, None], np.eye(2)[y], P)        # exact posterior: revealed rows are degenerate
    # ---- render notes
    notes = []
    style_tok = _words(ds, 24) if name == "D10_style_confound" else None
    for i in range(n):
        noise = [(t[:3] if (r.random() < style["abbrev"] and len(t) > 5) else t)
                 for t in r.choice(V, int(r.integers(style["len_noise"] // 2, style["len_noise"] + 1)))]
        phrases = [[t] for t in noise]                       # only noise words are abbreviated
        if name == "D01_echo_table_only" or name == "D10_style_confound":
            phrases += [[f"age{ageb[i]}"], [["female", "male"][sex[i]]], [f"lab{lab[i]}"]]
        for j in range(k):
            if mention[i, j]:
                phrases.append([POS[j % 2], FIND[j]])
            elif deny[i, j]:
                phrases.append([NEG[j % 2], FIND[j]])
        if numval is not None:
            phrases.append([FIND[5], f"{numval[i]:.1f}"])
        if leak_flag is not None and leak_flag[i]:
            phrases.append([["discharged", "expired"][y[i]]])
        if style_tok is not None:                 # writer style tied to the table cell, no outcome signal
            phrases += [[t] for t in list(style_tok[(cell[i] % 8) * 3:(cell[i] % 8) * 3 + 3]) * 3]
        if style["shuffle"]:                      # shuffle at PHRASE level (negation stays attached)
            phrases = [phrases[t] for t in r.permutation(len(phrases))]
        notes.append(" ".join(t for ph in phrases for t in ph))
    return dict(notes=notes, cell=cell, y=y, P=P, K=K, table=np.column_stack([ageb, sex, lab]), post=post)
