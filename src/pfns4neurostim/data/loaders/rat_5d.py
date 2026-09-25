"""Raw loader for the 5-D rat motor-cortex stimulation dataset.

The search space is five continuous/ordinal axes — pulse width (us), frequency (Hz),
train duration (ms) and the (x, y) position of the stimulating electrode on the 4x8
implant — so there is no 2-D electrode grid: ``ch2xy`` holds raw physical coordinates
and ``grid_shape`` is ``None``. The candidate pool is the full condition set
(2048 combinations), which makes this the scalability arm of Hyp A.

**Cohort ``noOutliers`` (2026-09-25).** This replaces the flat ``<subject>_5D.mat``
cohort. Two things changed, and both move a judgement out of our code and into the
recording lab's own review:

* ``valid_own`` — a per-repetition validity flag shipped with the data
  (``1`` valid, ``0`` noisy baseline, ``-1`` response outlier; see each subject's
  README). It replaces ``legacy_io._sort_valid_5drat_reps``, which flagged any
  repetition more than 2 SD from its own condition mean. That rule was symmetric, so
  it discarded genuinely large responses — exactly the sites a BO run is looking for —
  and its threshold was ours, not the lab's.
* ``flag_valid_emg`` — a per-EMG-channel boolean from visual inspection. It replaces
  the hand-maintained ``valid_emg_idx`` lists, which were transcribed from the README
  prose and had drifted from it (rCer1.14 kept all five channels "per user override";
  the lab's flag keeps three).

``rData03`` is not part of this cohort, so it is gone and **every subject index below
the removed one shifts**. That, and the changed validity flags, is why every cached
5d_rat cell of the previous cohort is stale: see :data:`COHORT`.

Response metric: **peak EMG** (metric 0 of the four the file carries), matching the
previous cohort and the other datasets.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.io

__all__ = ["COHORT", "SUBJECTS", "Rat5DSubject", "load_subject", "subject_dir"]

#: Cohort stamp of the raw files this loader reads. It travels on every channel
#: (``meta['data_cohort']``) into the cell-cache identity, so results computed on a
#: different cohort can never be returned as a hit for this one. Bump it whenever the
#: raw files, the validity flags or the subject ordering change.
COHORT: str = "noOutliers-2026-09-25"

#: Metric index of peak EMG within ``emg_response``'s last axis.
_PEAK_METRIC: int = 0

#: Columns of ``stim_combinations`` that form the 5-D search space:
#: [PW (us), freq (Hz), train duration (ms), x_ch, y_ch]. Column 3 (count) and
#: column 4 (channel id) are redundant with the (x, y) pair.
_SEARCH_AXES: tuple[int, ...] = (0, 1, 2, 5, 6)

#: Filename of the response file inside each subject directory.
_MAT_NAME: str = "5D_step4_OutliersRemoved.mat"


@dataclass(frozen=True)
class Rat5DSubject:
    """One recorded animal of the 5d_rat cohort.

    Attributes:
        key: Directory name under ``<data_root>/5d_rat/``, and the animal's lab id.
        emg_names: Muscle behind each raw EMG channel, in file order, as the
            subject's README documents them. Its length must match the file's EMG
            axis; a mismatch means the README and the recording have diverged and is
            raised rather than guessed around.
    """

    key: str
    emg_names: tuple[str, ...]


#: The cohort, in index order. **The index is the subject id used everywhere else**
#: (configs, splits, run tags, cached cells), so this tuple is append-only: reordering
#: or removing an entry silently re-labels every existing result.
#:
#: Order follows the previous cohort's, with ``rData03`` (index 0 there) dropped, so
#: every remaining animal keeps its relative position and only shifts down by one.
SUBJECTS: tuple[Rat5DSubject, ...] = (
    Rat5DSubject(
        key="rCer1.5",
        emg_names=(
            "left extensor carpi radialis",
            "left flexor carpi ulnaris",
            "left triceps",
            "left biceps",
        ),
    ),
    Rat5DSubject(
        key="BCI00",
        emg_names=(
            "left extensor carpi radialis",
            "left biceps",
            "left triceps",
            "left flexor carpi ulnaris",
            # The file carries a fifth channel that the README's muscle table does not
            # name. It is kept when flag_valid_emg marks it usable, under this name, so
            # a figure legend can never imply a muscle the recording does not document.
            "unnamed channel 5",
        ),
    ),
    Rat5DSubject(
        key="rCer1.12",
        emg_names=(
            "left extensor carpi radialis",
            "left flexor carpi ulnaris",
            "left biceps",
            "left triceps",
            "deltoid",
        ),
    ),
    Rat5DSubject(
        key="rCer1.14",
        emg_names=(
            "left extensor carpi radialis",
            "left flexor carpi ulnaris",
            "left biceps",
            "left triceps",
            "deltoid",
        ),
    ),
    Rat5DSubject(
        key="rCer1.15",
        emg_names=(
            "left extensor carpi radialis",
            "left flexor carpi ulnaris",
            "left biceps",
            "left triceps",
            "deltoid",
        ),
    ),
)


def subject_dir(subject: int, data_root: str) -> str:
    """Return the directory holding one subject's raw files.

    Args:
        subject: Subject index into :data:`SUBJECTS`.
        data_root: Directory holding the raw ``.mat`` trees.

    Returns:
        Absolute or relative path of ``<data_root>/5d_rat/<key>``.

    Raises:
        IndexError: If the subject index is not in the cohort.
    """
    if not 0 <= subject < len(SUBJECTS):
        raise IndexError(
            f"5d_rat subject {subject} is not in cohort {COHORT!r}, which has "
            f"{len(SUBJECTS)} subjects: "
            + ", ".join(f"{i}={s.key}" for i, s in enumerate(SUBJECTS))
            + ". Indices shifted when rData03 left the cohort."
        )
    return os.path.join(str(data_root), "5d_rat", SUBJECTS[subject].key)


def load_subject(subject: int, data_root: str = "./data") -> dict[str, Any]:
    """Load one 5d_rat subject into the raw dictionary the preprocessing consumes.

    Only the EMG channels the lab marked usable (``flag_valid_emg``) are returned, and
    a repetition counts as valid only where ``valid_own == 1``; the other two codes
    (``0`` noisy baseline, ``-1`` response outlier) are dropped. Invalid repetitions
    are masked out of the per-site mean and SD rather than averaged in.

    Args:
        subject: Subject index into :data:`SUBJECTS`.
        data_root: Directory holding the raw ``.mat`` trees.

    Returns:
        Dict with ``emgs``, ``nChan``, ``DimSearchSpace``, ``ch2xy`` [n_cond, 5],
        ``grid_shape`` (always ``None``), ``sorted_resp`` [n_cond, n_emgs, n_reps],
        ``sorted_isvalid`` (same shape, 0/1), ``sorted_respMean`` and
        ``sorted_respSD`` [n_cond, n_emgs], plus ``data_cohort`` and ``subject_key``.

    Raises:
        FileNotFoundError: If the subject's ``.mat`` file is missing.
        RuntimeError: If the README's muscle list and the file's EMG axis disagree, if
            the file declares no usable EMG channel, or if a response is non-finite.
    """
    spec = SUBJECTS[subject] if 0 <= subject < len(SUBJECTS) else None
    path = os.path.join(subject_dir(subject, data_root), _MAT_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"5d_rat subject {subject} ({spec.key}) has no {_MAT_NAME} at {path}. "
            f"Cohort {COHORT!r} stores one directory per animal; unpack "
            "datasets_noOutliers.zip into <data_root>/5d_rat/."
        )
    mat = scipy.io.loadmat(path)

    resp_all = np.asarray(mat["emg_response"], dtype=np.float64)     # [R, E_raw, C, M]
    valid_all = np.asarray(mat["valid_own"])                         # [E_raw, C, R]
    flags = np.asarray(mat["flag_valid_emg"]).ravel().astype(bool)   # [E_raw]
    params = np.asarray(mat["stim_combinations"], dtype=np.float64)  # [C, 7]

    n_raw_emgs = resp_all.shape[1]
    if len(spec.emg_names) != n_raw_emgs:
        raise RuntimeError(
            f"5d_rat subject {subject} ({spec.key}): the README documents "
            f"{len(spec.emg_names)} EMG channels but {_MAT_NAME} carries {n_raw_emgs}."
        )
    if flags.size != n_raw_emgs:
        raise RuntimeError(
            f"5d_rat subject {subject} ({spec.key}): flag_valid_emg has {flags.size} "
            f"entries for {n_raw_emgs} EMG channels."
        )
    keep = np.flatnonzero(flags)                                     # [E]
    if keep.size == 0:
        raise RuntimeError(
            f"5d_rat subject {subject} ({spec.key}): flag_valid_emg marks no EMG "
            "channel usable, so the subject has no channel to evaluate."
        )

    # The metric axis is taken first: `resp_all[:, keep, :, _PEAK_METRIC]` would combine
    # two advanced indices separated by a slice, which moves the broadcast axis to the
    # front and silently transposes the result.
    peak = resp_all[..., _PEAK_METRIC]                                    # [R, E_raw, C]
    # [R, E, C] -> [C, E, R]: the (site, EMG, repetition) layout every dataset uses.
    sorted_resp = peak[:, keep, :].transpose(2, 1, 0)                     # [C, E, R]
    # valid_own is [E_raw, C, R]; 1 is the only valid code.
    sorted_isvalid = (valid_all[keep] == 1).transpose(1, 0, 2).astype(np.int64)   # [C, E, R]
    if not np.isfinite(sorted_resp[sorted_isvalid == 1]).all():
        raise RuntimeError(
            f"5d_rat subject {subject} ({spec.key}): a repetition flagged valid by "
            "valid_own has a non-finite peak-EMG response."
        )

    masked = np.ma.masked_where(sorted_isvalid == 0, sorted_resp)         # [C, E, R]
    sorted_respMean = np.ma.filled(masked.mean(axis=-1), fill_value=0.0)  # [C, E]
    sorted_respSD = np.ma.filled(masked.std(axis=-1), fill_value=0.0)     # [C, E]

    ch2xy = params[:, list(_SEARCH_AXES)]                                 # [C, 5]
    n_cond = int(sorted_resp.shape[0])

    return {
        "emgs": [spec.emg_names[i] for i in keep],
        "nChan": n_cond,
        "DimSearchSpace": n_cond,
        "ch2xy": ch2xy,
        "grid_shape": None,
        "sorted_resp": sorted_resp,
        "sorted_isvalid": sorted_isvalid,
        "sorted_respMean": sorted_respMean,
        "sorted_respSD": sorted_respSD,
        "data_cohort": COHORT,
        "subject_key": spec.key,
    }
