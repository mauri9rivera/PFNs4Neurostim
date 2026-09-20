"""Subject splits per dataset (task #1 Step 2).

The held-out / train split exists so that any choice made by looking at results —
the primary acquisition function (H0-5), knob ranges, equivalence margins — is
made on TRAIN subjects and then *applied* to held-out subjects. Reporting a
headline number on a subject that was used to pick the settings would overstate it.

Exclusions, carried over from the legacy constants:

* NHP subject 2 (Macaque1) is excluded from ``ALL``: its signal is pure noise.
* ``5d_rat`` subject 6 is a placeholder in the loader and is excluded as well.

Values mirror ``utils.data_utils`` exactly; a test asserts they have not drifted.
"""
from __future__ import annotations

__all__ = [
    "HELD_OUT_SUBJECTS",
    "TRAIN_SUBJECTS",
    "ALL_SUBJECTS",
    "subjects_for",
    "DATASETS",
]

#: Subjects reserved for reporting. Nothing may be *chosen* on these.
HELD_OUT_SUBJECTS: dict[str, tuple[int, ...]] = {
    "rat": (0, 5),
    "nhp": (1,),
    "spinal": (0, 2, 5, 9),
    "5d_rat": (1, 4, 5),
}

#: Subjects on which settings may be selected.
TRAIN_SUBJECTS: dict[str, tuple[int, ...]] = {
    "rat": (1, 2, 3, 4),
    "nhp": (0, 3),
    "spinal": (1, 3, 4, 6, 7, 8, 10),
    "5d_rat": (0, 2, 3),
}

#: Every usable subject (NHP subject 2 excluded: pure-noise signal).
ALL_SUBJECTS: dict[str, tuple[int, ...]] = {
    "rat": (0, 1, 2, 3, 4, 5),
    "nhp": (0, 1, 3),
    "spinal": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    "5d_rat": (0, 1, 2, 3, 4, 5),
}

#: Dataset names with a defined split.
DATASETS: tuple[str, ...] = tuple(ALL_SUBJECTS)


def subjects_for(dataset: str, split: str = "held_out") -> tuple[int, ...]:
    """Return the subject indices of one split.

    Args:
        dataset: Dataset name.
        split: ``'held_out'``, ``'train'`` or ``'all'``.

    Returns:
        Subject indices.

    Raises:
        KeyError: If the dataset has no defined split.
        ValueError: If the split name is unknown.
    """
    tables = {"held_out": HELD_OUT_SUBJECTS, "train": TRAIN_SUBJECTS, "all": ALL_SUBJECTS}
    if split not in tables:
        raise ValueError(f"Unknown split {split!r}; expected one of {sorted(tables)}.")
    table = tables[split]
    if dataset not in table:
        raise KeyError(f"No {split!r} split defined for dataset {dataset!r}; known: {sorted(table)}.")
    return table[dataset]
