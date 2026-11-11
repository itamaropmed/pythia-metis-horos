import numpy as np
import pytest
import json

from research.instrumented_20261019 import reader_features
from research.instrumented_20261019.reader_features import tfidf


def test_tfidf_basis_is_invariant_to_heldout_text_changes():
    train = [
        "alpha beta alpha", "beta gamma beta", "alpha gamma beta",
        "gamma beta alpha", "delta alpha beta", "alpha delta gamma",
    ]
    heldout = ["zebra zeta hiddenterm zebra"] * 6
    texts = train + heldout
    fit_rows = np.array([True] * len(train) + [False] * len(heldout))
    first = tfidf(texts, k=2, fit_rows=fit_rows)
    changed = texts.copy()
    changed[len(train):] = ["quasar omega unseenword quasar"] * len(heldout)
    second = tfidf(changed, k=2, fit_rows=fit_rows)
    np.testing.assert_allclose(first[fit_rows], second[fit_rows], rtol=0., atol=0.)


def test_tfidf_requires_explicit_development_rows():
    with pytest.raises(ValueError, match="fit_rows is required"):
        tfidf(["alpha beta"] * 6, k=1)


def test_entity_bag_vocabulary_is_invariant_to_heldout_entities(tmp_path, monkeypatch):
    monkeypatch.setattr(reader_features, "CACHE", tmp_path)
    path = tmp_path / "ner_disease.jsonl"
    records = []
    for i in range(10):
        term = "common" if i < 6 else "heldout_only"
        records.append({"i": i, "v": [["DISEASE", term]]})
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    ids = np.arange(10)
    fit_rows = np.array([True] * 6 + [False] * 4)
    first = reader_features.ner_bag(ids, "disease", min_docs=2, fit_rows=fit_rows)
    for row in records[6:]:
        row["v"] = [["DISEASE", "different_heldout_only"]]
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    second = reader_features.ner_bag(ids, "disease", min_docs=2, fit_rows=fit_rows)
    np.testing.assert_array_equal(first[:6], second[:6])
    assert first.shape == second.shape


def test_entity_bag_requires_explicit_development_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(reader_features, "CACHE", tmp_path)
    path = tmp_path / "ner_disease.jsonl"
    path.write_text("".join(json.dumps({"i": i, "v": [["DISEASE", "common"]]}) + "\n"
                               for i in range(6)))
    with pytest.raises(ValueError, match="fit_rows is required"):
        reader_features.ner_bag(np.arange(6), "disease", min_docs=2)
