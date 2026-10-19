"""Information-set ladder comparisons with aligned-row enforcement."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .api import certify
from .contracts import FoldMatrix, GuardSpec, Representation, TaskCard


def evaluate_ladder(task: TaskCard,
                    entries: Sequence[tuple[Representation, FoldMatrix]],
                    guard: GuardSpec | None = None,
                    slices: dict[str, np.ndarray] | None = None,
                    slice_sets: Sequence[dict[str, np.ndarray] | None] | None = None) -> dict:
    if not entries:
        raise ValueError("information-set ladder is empty")
    base_rep, base_folds = entries[0]
    base_folds.validate()
    base_ids = (np.arange(np.asarray(base_folds.y).size) if base_folds.row_ids is None
                else np.asarray(base_folds.row_ids))
    rows = []
    previous = None
    if slice_sets is not None and len(slice_sets) != len(entries):
        raise ValueError("slice_sets must align one-for-one with ladder entries")
    for index, (representation, folds) in enumerate(entries):
        folds.validate()
        ids = np.arange(np.asarray(folds.y).size) if folds.row_ids is None else np.asarray(folds.row_ids)
        if not np.array_equal(ids, base_ids) or not np.array_equal(folds.y, base_folds.y):
            raise ValueError("all ladder entries must use identical ordered rows and targets")
        entry_slices = slice_sets[index] if slice_sets is not None else slices
        certificate = certify(task, folds, representation, guard, entry_slices)
        row = {"representation": representation.name, "certificate": certificate.to_dict()}
        if previous is not None and certificate.status == previous.status == "complete":
            row["change_from_previous"] = {
                "achieved_loss": certificate.achieved_loss - previous.achieved_loss,
                "bound": certificate.bound - previous.bound,
                "estimator": certificate.estimator - previous.estimator,
                "interpretation": (
                    "A worse fitted result does not imply that added information is harmful; "
                    "Bayes risk is monotone under nested information sets."
                ),
            }
        rows.append(row)
        previous = certificate
    return {
        "task": task.__dict__, "baseline": base_rep.name, "n_rows": int(base_ids.size),
        "entries": rows,
    }
