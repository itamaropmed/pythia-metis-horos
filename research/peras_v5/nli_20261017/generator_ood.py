"""Unseen (OOD) table + text families for testing Natural Language Information (NLI).  None of these generators or
vocabularies was used during development.  Each family has a signal-strength knob beta (beta = 0 is an exact null:
Y independent of the text given the table, while the text still echoes the table and carries cell-tied style).

The outcome law p(Y=1 | table, text) is known EXACTLY: the text renders latent features deterministically (with
nuisance variation), so the posterior over latents given the note is degenerate and p is a closed-form function.
True NLI (Brier units) = mean over rows of |p_i - pbar_cell(i)|^2 * 2  (binary: |p - pbar|^2 over both classes)."""
import numpy as np, string

FAMILIES = ["O1_grammar_severity", "O2_long_range_negation", "O3_numbers_in_words", "O4_copy_forward_template",
            "O5_two_languages", "O6_local_rare_token"]
NUMW = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def _w(r, k, lo=3, hi=9):
    return ["".join(r.choice(list(string.ascii_lowercase), int(r.integers(lo, hi)))) for _ in range(k)]


def make_ood(family, n, beta, seed):
    r = np.random.default_rng(seed); ds = np.random.default_rng(50_000 + FAMILIES.index(family))
    V = _w(ds, 600); V2 = _w(ds, 600)                       # base vocabulary (and a second 'language')
    ageb = r.integers(0, 4, n); sex = r.integers(0, 2, n); lab = r.integers(0, 3, n); cell = ageb * 6 + sex * 3 + lab
    base = .5 * (ageb - 1.5) + .4 * (sex - .5) + .6 * (lab - 1)
    STY = _w(ds, 30)                                         # writer-style tokens tied to the table cell (confound)
    sig = np.zeros(n); notes = []
    if family == "O1_grammar_severity":
        sev = r.integers(0, 3, n); SYN = [["mild", "slight", "minimal"], ["moderate", "intermediate", "medium"], ["severe", "marked", "extreme"]]
        FIND = _w(ds, 5); sig = (sev - 1.0)
    elif family == "O2_long_range_negation":
        neg = r.random(n) < .5; FIND = _w(ds, 1)[0]; sig = np.where(neg, -1.0, 1.0)
    elif family == "O3_numbers_in_words":
        val = r.integers(0, 10, n); sig = (val - 4.5) / 3.0
    elif family == "O4_copy_forward_template":
        TPL = [" ".join(ds.choice(V, 60)) for _ in range(12)]; tpl = r.integers(0, 12, n)
        flag = r.random(n) < .4; NEW = _w(ds, 2); sig = np.where(flag, 1.0, -.6)
    elif family == "O5_two_languages":
        lang = r.integers(0, 2, n); conc = r.random(n) < .45; C = (_w(ds, 1)[0], _w(ds, 1)[0]); sig = np.where(conc, 1.0, -.8)
    elif family == "O6_local_rare_token":
        rare = r.random(n) < .15; RT = _w(ds, 1)[0]; sig = np.where(rare & (ageb == 3), 2.5, 0.0) - .375 * (ageb == 3)
    p = 1 / (1 + np.exp(-(base + beta * sig)))
    for i in range(n):
        toks = list(r.choice(V2 if (family == "O5_two_languages" and lang[i]) else V, int(r.integers(15, 45))))
        toks += [f"age{ageb[i]}", ["female", "male"][sex[i]], f"lab{lab[i]}"] + [STY[cell[i] % 30]] * 2      # echo + style
        if family == "O1_grammar_severity":
            toks += ["patient", "reports", str(r.choice(FIND)), "of", str(r.choice(SYN[sev[i]])), "intensity"]
        elif family == "O2_long_range_negation":
            pre = ["no", "history", "of"] if neg[i] else ["known", "history", "of"]
            toks = pre + list(r.choice(V, int(r.integers(8, 20)))) + [FIND] + toks          # cue far from the finding
        elif family == "O3_numbers_in_words":
            toks += ["creatinine", NUMW[val[i]], "point", NUMW[int(r.integers(0, 10))]]
        elif family == "O4_copy_forward_template":
            toks = TPL[tpl[i]].split() + toks + ([NEW[0], NEW[1]] if flag[i] else [])
        elif family == "O5_two_languages":
            if conc[i]:
                toks += [C[lang[i]]]
        elif family == "O6_local_rare_token":
            if rare[i]:
                toks += [RT]
        r.shuffle(toks) if family not in ("O2_long_range_negation", "O3_numbers_in_words") else None
        notes.append(" ".join(toks))
    P = np.column_stack([1 - p, p]); y = (r.random(n) < p).astype(int)
    pbar = np.zeros(n)
    for c in np.unique(cell):
        pbar[cell == c] = p[cell == c].mean()
    true_nli = float((2 * (p - pbar) ** 2).mean())
    return dict(notes=notes, cell=cell, y=y, P=P, true_nli=true_nli, K=2,
                registered_support=(rare.astype(int) if family == "O6_local_rare_token" else None))
