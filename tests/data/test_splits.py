"""Subject splits must not drift from the legacy constants (task #1 Step 2),

except for NHP's ``ALL_SUBJECTS``, which intentionally diverges as of roadmap DS1 (2026-10-09):
subject 2 (Macaque1) was excluded from every legacy constant as a "pure-noise signal" without a
pre-declared threshold, and is restored here. ``HELD_OUT_SUBJECTS`` / ``TRAIN_SUBJECTS`` still
mirror the legacy train/held-out partition, which predates subject 2's inclusion and is unused by
the current (no-split) policy — so for NHP alone, train | held_out no longer covers all.
"""
from __future__ import annotations

import pytest

from pfns4neurostim.data import splits

#: DS1: NHP's ALL_SUBJECTS intentionally diverges from legacy_io and from TRAIN | HELD_OUT.
_DS1_EXCEPTION = "nhp"


class TestSplits:
    """Values, disjointness and lookup behaviour."""

    def test_values_match_the_legacy_constants(self) -> None:
        """Guards the migration: the package copy must equal the original, except DS1's NHP ALL_SUBJECTS."""
        from pfns4neurostim.data.legacy_io import ALL_SUBJECTS, HELD_OUT_SUBJECTS, TRAIN_SUBJECTS

        for dataset in splits.DATASETS:
            assert tuple(HELD_OUT_SUBJECTS[dataset]) == splits.HELD_OUT_SUBJECTS[dataset]
            assert tuple(TRAIN_SUBJECTS[dataset]) == splits.TRAIN_SUBJECTS[dataset]
            if dataset == _DS1_EXCEPTION:
                assert set(splits.ALL_SUBJECTS[dataset]) == set(ALL_SUBJECTS[dataset]) | {2}
            else:
                assert tuple(ALL_SUBJECTS[dataset]) == splits.ALL_SUBJECTS[dataset]

    @pytest.mark.parametrize("dataset", splits.DATASETS)
    def test_train_and_held_out_are_disjoint(self, dataset: str) -> None:
        """A subject used to choose settings must never also report them."""
        train = set(splits.TRAIN_SUBJECTS[dataset])
        held = set(splits.HELD_OUT_SUBJECTS[dataset])
        assert train & held == set()

    @pytest.mark.parametrize("dataset", [d for d in splits.DATASETS if d != _DS1_EXCEPTION])
    def test_train_and_held_out_cover_all(self, dataset: str) -> None:
        train = set(splits.TRAIN_SUBJECTS[dataset])
        held = set(splits.HELD_OUT_SUBJECTS[dataset])
        assert train | held == set(splits.ALL_SUBJECTS[dataset])

    def test_nhp_all_subjects_covers_train_and_held_out_plus_subject_2(self) -> None:
        """DS1: NHP ALL_SUBJECTS is exactly the legacy train|held_out partition plus subject 2."""
        train = set(splits.TRAIN_SUBJECTS["nhp"])
        held = set(splits.HELD_OUT_SUBJECTS["nhp"])
        assert set(splits.ALL_SUBJECTS["nhp"]) == train | held | {2}

    def test_nhp_subject_2_is_included(self) -> None:
        """DS1 (2026-10-09): subject 2 is restored, pending the pre-declared channel-exclusion rule."""
        assert 2 in splits.ALL_SUBJECTS["nhp"]

    def test_subjects_for_returns_each_split(self) -> None:
        assert splits.subjects_for("nhp", "held_out") == (1,)
        assert splits.subjects_for("nhp", "train") == (0, 3)
        assert splits.subjects_for("nhp", "all") == (0, 1, 2, 3)

    def test_unknown_split_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown split"):
            splits.subjects_for("nhp", "validation")

    def test_unknown_dataset_raises(self) -> None:
        with pytest.raises(KeyError, match="No 'train' split"):
            splits.subjects_for("cortical", "train")
