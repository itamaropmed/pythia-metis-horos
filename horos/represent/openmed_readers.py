"""Open-source medical readers for the Pythia / Metis pipeline (all local, free; no text leaves the machine).

Three reader families, each producing label-free features per note (Pythia "readers", see
docs/theory/INSTRUMENTED_INFORMATION_20261019.md):

1. OpenMedPII        -- OpenMed PII / de-identification models (e.g. OpenMed-PII-SuperClinical-Small-44M-v1).
                        Used (a) as a privacy guard before any remote LLM call and before exporting terms,
                        (b) to build a *de-identified text* reader, so the information lost to de-identification
                        can be certified (P4 regression test).
2. OpenMedNER        -- OpenMed NER models (disease, pharma, anatomy, oncology, pathology ...): entity bags.
3. FastRuleReader    -- free and fast: medspaCy ConText (negation / historical / hypothetical) over target
                        rules; milliseconds per note, the default *cheap reader* for Metis routing.

Model weights are loaded from a local path or a Hugging Face id (offline cache is honoured: set
HF_HUB_OFFLINE=1 on machines without hub access).  Pythia never trains these models; Horos only runs them.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# ------------------------------------------------------------------------------------------------ PII
PHI_LABELS = {"first_name", "last_name", "name", "medical_record_number", "phone_number", "email", "street_address",
              "date", "date_of_birth", "ssn", "account_number", "health_plan_beneficiary_number", "url", "ip_address",
              "certificate_license_number", "vehicle_identifier", "device_identifier", "biometric_identifier", "fax"}


@dataclass
class OpenMedPII:
    model: str = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
    threshold: float = .5
    max_chars: int = 6000
    _loader: object = field(default=None, repr=False)

    def _ld(self):
        if self._loader is None:
            import openmed
            self._loader = openmed.ModelLoader()          # keeps the pipeline in memory across calls
        return self._loader

    def entities(self, text: str):
        import openmed
        if not text or not text.strip():
            return []
        return openmed.extract_pii(text[: self.max_chars], model_name=self.model,
                                   confidence_threshold=self.threshold, loader=self._ld()).entities

    def deidentify(self, text: str, method: str = "mask") -> str:
        import openmed
        if not text or not text.strip():
            return text or ""
        r = openmed.deidentify(text[: self.max_chars], method=method, model_name=self.model, loader=self._ld())
        return getattr(r, "deidentified_text", str(r))

    def guard(self, strings: Iterable[str]) -> dict:
        """Privacy guard for anything about to be exported (reportable terms, headers, report text).
        Returns {string: [phi labels]} for strings containing direct identifiers; empty dict = clean."""
        flagged = {}
        for s in strings:
            labs = sorted({e.label for e in self.entities(s)} & PHI_LABELS)
            if labs:
                flagged[s] = labs
        return flagged


def phi_vocabulary(texts: Iterable[str], pii: "OpenMedPII", labels: set = PHI_LABELS) -> set:
    """Lower-case tokens that the OpenMed PII model tags as direct identifiers anywhere in the corpus.
    Passed as ``exclude`` to the note atlas so no such token is ever reported, whatever its frequency."""
    vocab = set()
    for t in texts:
        for e in pii.entities(t):
            if e.label in labels:
                vocab.update(w.lower() for w in re.findall(r"[A-Za-z]+", e.text))
    return vocab


# ------------------------------------------------------------------------------------------------ NER
OPENMED_NER = {
    "disease": "OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-434M",
    "disease_tiny": "OpenMed/OpenMed-NER-DiseaseDetect-TinyMed-135M",
    "pharma": "OpenMed/OpenMed-NER-PharmaDetect-SuperClinical-434M",
    "anatomy": "OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-109M",
    "oncology": "OpenMed/OpenMed-NER-OncologyDetect-SuperClinical-434M",
    "pathology": "OpenMed/OpenMed-NER-PathologyDetect-ModernClinical-395M",
}


@dataclass
class OpenMedNER:
    """OpenMed token-classification model run through a batched transformers pipeline.
    Long notes are cut into ~1,500-character windows (the models take 512 tokens)."""
    model: str = OPENMED_NER["disease_tiny"]
    threshold: float = .5
    max_chars: int = 12000
    window: int = 1500
    batch_size: int = 32
    _pipe: object = field(default=None, repr=False)

    def _p(self):
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline("token-classification", model=self.model, aggregation_strategy="first",
                                  device=-1)
        return self._pipe

    def entities(self, text: str) -> list[tuple[str, str]]:
        return self.entities_batch([text])[0]

    def entities_batch(self, texts: Sequence[str]) -> list[list[tuple[str, str]]]:
        chunks, owner = [], []
        for i, t in enumerate(texts):
            t = (t or "")[: self.max_chars]
            for k in range(0, len(t), self.window):
                if t[k:k + self.window].strip():
                    chunks.append(t[k:k + self.window]); owner.append(i)
        out = [[] for _ in texts]
        if chunks:
            for i, res in zip(owner, self._p()(chunks, batch_size=self.batch_size)):
                out[i] += [(e["entity_group"], e["word"].lower().strip()) for e in res if e["score"] >= self.threshold]
        return out


# ------------------------------------------------------------------------------------------------ fast rules
DEFAULT_TARGETS = {
    "GENERAL_ANESTHESIA": ["general anesthesia", "general endotracheal", "general anesthetic"],
    "REGIONAL_ANESTHESIA": ["spinal anesthesia", "epidural", "regional anesthesia", "nerve block"],
    "COMPLICATION": ["complication", "complications"],
    "BLEEDING": ["hemorrhage", "bleeding", "blood loss"],
    "EMERGENCY": ["emergent", "emergency", "urgent"],
    "OBESITY": ["obesity", "obese", "morbid obesity"],
    "DIABETES": ["diabetes", "diabetic", "dm2"],
    "HYPERTENSION": ["hypertension", "htn"],
    "CANCER": ["carcinoma", "malignancy", "cancer", "tumor"],
    "ADHESIONS": ["adhesions", "adhesiolysis"],
    "REOPERATION": ["redo", "re-operation", "reoperation", "revision"],
    "CONVERSION": ["converted to open", "conversion to open"],
}


@dataclass
class FastRuleReader:
    """medspaCy ConText reader: one column per (target, status) with status in {asserted, negated, other}."""
    targets: dict = field(default_factory=lambda: dict(DEFAULT_TARGETS))
    max_chars: int = 20000

    def __post_init__(self):
        import sys
        import medspacy
        from medspacy.ner import TargetRule
        try:                                    # PyRuSH logs every token at DEBUG level
            from loguru import logger
            logger.remove(); logger.add(sys.stderr, level="WARNING")
        except Exception:
            pass
        self.nlp = medspacy.load(enable=["medspacy_pyrush", "medspacy_target_matcher", "medspacy_context"])
        self.nlp.get_pipe("medspacy_target_matcher").add(
            [TargetRule(p, lab) for lab, ps in self.targets.items() for p in ps])
        self.columns = [f"{t}:{s}" for t in self.targets for s in ("asserted", "negated", "other")]

    def features(self, texts: Sequence[str]) -> np.ndarray:
        X = np.zeros((len(texts), len(self.columns)), float); idx = {c: i for i, c in enumerate(self.columns)}
        for i, doc in enumerate(self.nlp.pipe([t[: self.max_chars] for t in texts], batch_size=32)):
            for e in doc.ents:
                st = "negated" if e._.is_negated else ("other" if (e._.is_historical or e._.is_hypothetical
                                                                   or e._.is_family) else "asserted")
                X[i, idx[f"{e.label_}:{st}"]] = 1.0
        return X


# ------------------------------------------------------------------------------------------------ features
def entity_bag_features(ents: Sequence[Sequence[tuple[str, str]]], groups: Sequence[object], min_groups: int = 50):
    """Entity bags -> (counts per entity label) + (presence of entity strings seen in >= min_groups distinct
    patients/notes).  The min_groups rule is the same privacy floor used for reportable terms."""
    labels = sorted({l for e in ents for l, _ in e})
    by_term = {}
    for g, e in zip(groups, ents):
        for _, t in e:
            by_term.setdefault(t, set()).add(g)
    vocab = sorted(t for t, s in by_term.items() if len(s) >= min_groups)
    vi = {t: j for j, t in enumerate(vocab)}; li = {l: j for j, l in enumerate(labels)}
    X = np.zeros((len(ents), len(labels) + len(vocab)))
    for i, e in enumerate(ents):
        c = Counter(l for l, _ in e)
        for l, k in c.items():
            X[i, li[l]] = np.log1p(k)
        for _, t in e:
            if t in vi:
                X[i, len(labels) + vi[t]] = 1.0
    return X, labels + [f"term:{t}" for t in vocab]


_PHI_REGEX = re.compile(r"\b(\d{3}[-.\s]\d{3}[-.\s]\d{4}|\d{2}/\d{2}/\d{4}|MRN\s*\d+|\d{3}-\d{2}-\d{4})\b", re.I)


def regex_phi_sweep(strings: Iterable[str]) -> list[str]:
    """Zero-cost backstop that runs even when no model is available."""
    return [s for s in strings if _PHI_REGEX.search(s)]
