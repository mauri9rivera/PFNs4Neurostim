"""Subject splits must not drift from the legacy constants (task #1 Step 2)."""
from __future__ import annotations

import pytest

from pfns4neurostim.data import splits


class TestSplits:
    """Values, disjointness and lookup behaviour."""

    def test_values_match_the_legacy_constants(self) -> None:
        """Guards the migration: the package copy must equal the original."""
        from pfns4neurostim.data.legacy_io import ALL_SUBJECTS, HELD_OUT_SUBJECTS, TRAIN_SUBJECTS

        for dataset in splits.DATASETS:
            assert tuple(HELD_OUT_SUBJECTS[dataset]) == splits.HELD_OUT_SUBJECTS[dataset]
            assert tuple(TRAIN_SUBJECTS[dataset]) == splits.TRAIN_SUBJECTS[dataset]
            assert tuple(ALL_SUBJECTS[dataset]) == splits.ALL_SUBJECTS[dataset]

    @pytest.mark.parametrize("dataset", splits.DATASETS)
    def test_train_and_held_out_are_disjoint(self, dataset: str) -> None:
        """A subject used to choose settings must never also report them."""
        train = set(splits.TRAIN_SUBJECTS[dataset])
        held = set(splits.HELD_OUT_SUBJECTS[dataset])
        assert train & held == set()

    @pytest.mark.parametrize("dataset", splits.DATASETS)
    def test_train_and_held_out_cover_all(self, dataset: str) -> None:
        train = set(splits.TRAIN_SUBJECTS[dataset])
        held = set(splits.HELD_OUT_SUBJECTS[dataset])
        assert train | held == set(splits.ALL_SUBJECTS[dataset])

    def test_nhp_subject_2_stays_excluded(self) -> None:
        """NHP subject 2 is a pure-noise recording and must never be included."""
        assert 2 not in splits.ALL_SUBJECTS["nhp"]

    def test_subjects_for_returns_each_split(self) -> None:
        assert splits.subjects_for("nhp", "held_out") == (1,)
        assert splits.subjects_for("nhp", "train") == (0, 3)
        assert splits.subjects_for("nhp", "all") == (0, 1, 3)

    def test_unknown_split_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown split"):
            splits.subjects_for("nhp", "validation")

    def test_unknown_dataset_raises(self) -> None:
        with pytest.raises(KeyError, match="No 'train' split"):
            splits.subjects_for("cortical", "train")
