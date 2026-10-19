"""Deterministic, label-free text perturbations for representation stress tests.

These transformations are controls, not data augmentation defaults.  They
never access outcomes and return aggregate provenance without exposing text.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import re
from typing import Sequence

import numpy as np


_KINDS = {
    "identity", "character_shuffle", "token_shuffle", "case_dropout",
    "span_mask", "segment_dropout", "segment_duplicate", "segment_shuffle",
}
_SEGMENT_START = re.compile(r"(?=\[NOTE_TYPE:)")
_PIECES = re.compile(r"\S+|\s+")


@dataclass(frozen=True)
class TextPerturbationSpec:
    kind: str
    fraction: float = 1.0
    seed: int = 0
    version: str = "text-stress-v1"

    def validate(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unsupported text perturbation: {self.kind!r}")
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError("perturbation fraction must lie in [0, 1]")


@dataclass(frozen=True)
class PerturbationAudit:
    spec: TextPerturbationSpec
    n_rows: int
    changed_rows: int
    empty_before: int
    empty_after: int
    characters_before: int
    characters_after: int
    source_hash: str
    result_hash: str
    label_accessed: bool = False
    text_persisted: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _hash_texts(row_ids: Sequence[object], texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for row_id, text in zip(row_ids, texts):
        digest.update(str(row_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _rng(spec: TextPerturbationSpec, row_id: object) -> np.random.Generator:
    key = f"{spec.version}\0{spec.kind}\0{spec.seed}\0{row_id}".encode("utf-8")
    entropy = np.frombuffer(hashlib.sha256(key).digest(), dtype=np.uint32)
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _segments(text: str) -> list[str]:
    return [part for part in _SEGMENT_START.split(text) if part]


def _perturb_one(text: str, spec: TextPerturbationSpec, rng: np.random.Generator) -> str:
    if spec.kind == "identity" or not text:
        return text
    if spec.kind == "case_dropout":
        return "" if rng.random() < spec.fraction else text
    if spec.kind == "character_shuffle":
        chars = np.asarray(list(text), dtype="<U1")
        selected = rng.random(chars.size) < spec.fraction
        positions = np.flatnonzero(selected)
        if positions.size > 1:
            chars[positions] = chars[positions][rng.permutation(positions.size)]
        return "".join(chars.tolist())
    if spec.kind == "token_shuffle":
        pieces = _PIECES.findall(text)
        positions = [i for i, value in enumerate(pieces) if not value.isspace()]
        selected = [i for i in positions if rng.random() < spec.fraction]
        if len(selected) > 1:
            values = [pieces[i] for i in selected]
            order = rng.permutation(len(values))
            for i, j in zip(selected, order):
                pieces[i] = values[int(j)]
        return "".join(pieces)
    if spec.kind == "span_mask":
        n_mask = int(np.ceil(len(text) * spec.fraction))
        if n_mask == 0:
            return text
        start = int(rng.integers(0, len(text) - n_mask + 1))
        chars = list(text)
        for i in range(start, start + n_mask):
            if not chars[i].isspace():
                chars[i] = "█"
        return "".join(chars)

    segments = _segments(text)
    if not segments:
        return text
    if spec.kind == "segment_dropout":
        keep = rng.random(len(segments)) >= spec.fraction
        return " ".join(segment for segment, retained in zip(segments, keep) if retained)
    if spec.kind == "segment_duplicate":
        n_duplicate = int(np.ceil(len(segments) * spec.fraction))
        if n_duplicate == 0:
            return text
        selected = rng.choice(len(segments), size=n_duplicate, replace=True)
        return " ".join(segments + [segments[int(i)] for i in selected])
    if spec.kind == "segment_shuffle":
        selected = [i for i in range(len(segments)) if rng.random() < spec.fraction]
        values = [segments[i] for i in selected]
        order = rng.permutation(len(values))
        for i, j in zip(selected, order):
            segments[i] = values[int(j)]
        return " ".join(segments)
    raise AssertionError("validated perturbation kind was not implemented")


def perturb_texts(
    texts: Sequence[object], row_ids: Sequence[object], spec: TextPerturbationSpec
) -> tuple[np.ndarray, PerturbationAudit]:
    """Return perturbed text and a content-free reproducibility audit."""
    spec.validate()
    if len(texts) != len(row_ids):
        raise ValueError("texts and row_ids must align one-for-one")
    source = np.asarray(["" if value is None else str(value) for value in texts], dtype=object)
    result = np.asarray([
        _perturb_one(text, spec, _rng(spec, row_id))
        for text, row_id in zip(source, row_ids)
    ], dtype=object)
    audit = PerturbationAudit(
        spec=spec, n_rows=len(source), changed_rows=int(np.sum(source != result)),
        empty_before=int(np.sum(source == "")), empty_after=int(np.sum(result == "")),
        characters_before=int(sum(map(len, source))), characters_after=int(sum(map(len, result))),
        source_hash=_hash_texts(row_ids, source), result_hash=_hash_texts(row_ids, result),
    )
    return result, audit
