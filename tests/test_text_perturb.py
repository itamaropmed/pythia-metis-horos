import numpy as np
import pytest

from horos.represent.perturb import TextPerturbationSpec, perturb_texts


TEXTS = np.asarray([
    "[NOTE_TYPE: A] alpha beta gamma",
    "[NOTE_TYPE: B] delta [NOTE_TYPE: C] epsilon zeta",
    "",
], dtype=object)
ROWS = np.asarray(["case-a", "case-b", "case-c"])


@pytest.mark.parametrize(
    "kind",
    [
        "identity", "character_shuffle", "token_shuffle", "case_dropout",
        "span_mask", "segment_dropout", "segment_duplicate", "segment_shuffle",
    ],
)
def test_perturbations_are_deterministic_and_content_safe(kind):
    spec = TextPerturbationSpec(kind=kind, fraction=0.75, seed=19)
    first, audit = perturb_texts(TEXTS, ROWS, spec)
    second, repeat = perturb_texts(TEXTS, ROWS, spec)
    assert np.array_equal(first, second)
    assert audit.result_hash == repeat.result_hash
    assert audit.n_rows == len(TEXTS)
    assert audit.label_accessed is False
    assert audit.text_persisted is False
    assert "alpha" not in str(audit.to_dict())


def test_character_shuffle_preserves_character_multiset_and_length():
    result, audit = perturb_texts(
        TEXTS, ROWS, TextPerturbationSpec("character_shuffle", fraction=1.0, seed=7)
    )
    for before, after in zip(TEXTS, result):
        assert len(before) == len(after)
        assert sorted(before) == sorted(after)
    assert audit.characters_before == audit.characters_after


def test_case_dropout_and_segment_duplicate_have_expected_direction():
    dropped, drop_audit = perturb_texts(
        TEXTS, ROWS, TextPerturbationSpec("case_dropout", fraction=1.0, seed=2)
    )
    duplicated, duplicate_audit = perturb_texts(
        TEXTS, ROWS, TextPerturbationSpec("segment_duplicate", fraction=1.0, seed=2)
    )
    assert np.all(dropped == "")
    assert drop_audit.characters_after == 0
    assert duplicate_audit.characters_after > duplicate_audit.characters_before
    assert len(duplicated[1]) > len(TEXTS[1])


def test_row_keyed_randomness_is_order_invariant():
    spec = TextPerturbationSpec("token_shuffle", fraction=1.0, seed=31)
    normal, _ = perturb_texts(TEXTS, ROWS, spec)
    reverse, _ = perturb_texts(TEXTS[::-1], ROWS[::-1], spec)
    assert np.array_equal(normal, reverse[::-1])


def test_invalid_specs_and_alignment_refuse():
    with pytest.raises(ValueError, match="unsupported"):
        perturb_texts(TEXTS, ROWS, TextPerturbationSpec("unknown"))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        perturb_texts(TEXTS, ROWS, TextPerturbationSpec("case_dropout", fraction=1.1))
    with pytest.raises(ValueError, match="align"):
        perturb_texts(TEXTS, ROWS[:2], TextPerturbationSpec("identity"))
