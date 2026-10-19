"""Hard, fully-informative pre-op notes for the 10 designs.
Every note states complexity c (0/1/2), redo status r (0/1) and access-risk flag k (0/1) through
paraphrases, negations, abbreviations and misleading distractors; typos are injected.
Sentences are generated independently, so an LLM's reading of a note is the composition of its
readings of the sentences (the sentence-level cache is validated against whole-note reading)."""
import numpy as np

C = {0: ["straightforward anatomy", "routine case anticipated", "uncomplicated anatomy on prior imaging",
         "standard approach expected", "anatomy looks favorable", "simple case per EP fellow"],
     1: ["moderately dilated left atrium", "moderately tortuous coronary sinus anatomy", "intermediate complexity anticipated",
         "mild anatomic variant noted on CT", "moderate LA enlargement", "somewhat complex substrate, not severe"],
     2: ["highly complex substrate", "severely enlarged LA with extensive scar", "complex congenital anatomy",
         "very challenging anatomy expected", "extensive fibrosis on MRI, difficult case", "high complexity, prolonged mapping likely"]}
R = {1: ["redo procedure", "s/p prior ablation in {y}", "repeat PVI after recurrence", "history of prior EP procedure at outside hospital",
         "second attempt after failed initial ablation", "previous ablation {y}, now recurrent"],
     0: ["no prior procedures", "first-time procedure", "denies previous ablation", "de novo case, no history of prior EP study",
         "no previous interventions", "ablation history: none"]}
K = {1: ["history of difficult vascular access", "known IVC filter, access may be difficult", "prior groin hematoma after access",
         "venous occlusion on prior study", "femoral access expected to be difficult, plan alternative route"],
     0: ["access expected to be unremarkable", "no history of difficult access", "vascular access uncomplicated previously",
         "groin access straightforward per prior notes", "no access issues anticipated"]}
DISTRACT = ["family history of complex arrhythmias", "complex medication list reviewed", "no complications expected per patient",
            "patient anxious, prefers early slot", "redo of labs requested by anesthesia", "previous surgeon on vacation",
            "difficult to reach by phone", "access to transport arranged", "prior colonoscopy uneventful", "allergic to penicillin",
            "BP {a}/{b} at intake", "creatinine {c} mg/dL", "consent signed", "NPO after midnight", "interpreter needed",
            "lives alone", "diabetic, A1c {d}", "former smoker, quit {y}"]
TYPO = {"anatomy": "anatmy", "procedure": "procedue", "previous": "prev", "history": "hx", "complex": "cmplx",
        "access": "acess", "difficult": "dificult", "prior": "pior"}

def canon(s):
    return s.replace("{y}", "2018").replace("{a}", "130").replace("{b}", "80").replace("{c}", "1.1").replace("{d}", "7.2")

def sentence(rng, s):
    """returns (realised sentence, canonical bank string with the same typos)."""
    cs = canon(s)
    s = s.format(y=rng.integers(2005, 2024), a=rng.integers(95, 180), b=rng.integers(50, 100),
                 c=round(rng.uniform(0.6, 2.5), 1), d=round(rng.uniform(5.5, 9.5), 1))
    ty = [k_ for k_ in TYPO if k_ in s and k_ in cs and rng.random() < 0.15][:2]
    for k_ in ty:
        s = s.replace(k_, TYPO[k_], 1); cs = cs.replace(k_, TYPO[k_], 1)
    return s, cs

def note(rng, c, r, k):
    parts = [rng.choice(C[c]), rng.choice(R[r]), rng.choice(K[k])]
    parts += list(rng.choice(DISTRACT, size=rng.integers(1, 4), replace=False))
    rng.shuffle(parts)
    ss = [sentence(rng, p) for p in parts]
    return "Pre-op note: " + "; ".join(a for a, _ in ss) + ".", [b for _, b in ss]

def bank_strings():
    """all canonical sentence templates (placeholders filled generically) + typo variants, for LLM labelling."""
    out = []
    for grp in (C, R, K):
        for v in grp.values(): out += v
    out += DISTRACT
    res = set()
    for s in out:
        base = s.replace("{y}", "2018").replace("{a}", "130").replace("{b}", "80").replace("{c}", "1.1").replace("{d}", "7.2")
        res.add(base)
        for k_, v in TYPO.items():
            if k_ in base: res.add(base.replace(k_, v, 1))
        # double typos
        keys = [k_ for k_ in TYPO if k_ in base]
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                res.add(base.replace(keys[i], TYPO[keys[i]], 1).replace(keys[j], TYPO[keys[j]], 1))
    return sorted(res)
