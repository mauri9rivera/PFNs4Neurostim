"""Subject splits per dataset (task #1 Step 2).

The held-out / train split exists so that any choice made by looking at results —
the primary acquisition function (H0-5), knob ranges, equivalence margins — is
made on TRAIN subjects and then *applied* to held-out subjects. Reporting a
headline number on a subject that was used to pick the settings would overstate it.

Exclusions, carried over from the legacy constants:

* ``5d_rat`` indices are positions in
  :data:`pfns4neurostim.data.loaders.rat_5d.SUBJECTS` (cohort
  ``noOutliers-2026-09-25``): 0 rCer1.5, 1 BCI00, 2 rCer1.12, 3 rCer1.14, 4 rCer1.15.
  ``rData03`` and the ``5D_step4_noartrej`` placeholder are not in that cohort, so the
  dataset went from six indices to five and every index below the removed one shifted
  down by one. Results computed under the previous indexing are not comparable; the
  cohort stamp on every channel keeps them out of the cell cache.

Intentional divergence from the legacy constants (roadmap DS1, closed 2026-10-09): NHP subject 2
(Macaque1, ``data/monkeys/Macaque1_M1_181212.mat``) was excluded from every legacy constant as a
"pure-noise signal", a qualitative call that was never backed by a pre-declared, uniformly-applied
threshold. It is restored to ``ALL_SUBJECTS`` here, so every NHP dataset config now evaluates all
four subjects (2 Cebus, 2 Macaque). ``HELD_OUT_SUBJECTS`` and ``TRAIN_SUBJECTS`` still mirror the
legacy train/held-out partition exactly, since that split predates subject 2's inclusion and no
train/held-out split has been used since 2026-09-20 (`CLAUDE.md` Section 5) — nothing downstream
reads them except ``legacy_code`` and the regression test below. Subject 2 enters with every channel:
the user dropped DS1's planned channel-exclusion rule on 2026-10-15, so no reliability / SNR filter applies.

Values other than NHP's ``ALL_SUBJECTS`` mirror ``legacy_io`` exactly; a test asserts they have not
drifted.
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
    "5d_rat": (0, 3, 4),
}

#: Subjects on which settings may be selected.
TRAIN_SUBJECTS: dict[str, tuple[int, ...]] = {
    "rat": (1, 2, 3, 4),
    "nhp": (0, 3),
    "spinal": (1, 3, 4, 6, 7, 8, 10),
    "5d_rat": (1, 2),
}

#: Every usable subject (NHP subject 2 restored, DS1, 2026-10-09).
ALL_SUBJECTS: dict[str, tuple[int, ...]] = {
    "rat": (0, 1, 2, 3, 4, 5),
    "nhp": (0, 1, 2, 3),
    "spinal": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    "5d_rat": (0, 1, 2, 3, 4),
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
