"""Raw-data loaders: one module per dataset, one entry point for every caller.

:func:`load_subject` is the single seam between the package and the recorded ``.mat``
files. Everything downstream (``data.preprocessing``, ``data.channels``) works on the
dictionary it returns and never opens a file itself, so adding a dataset or replacing
a cohort touches one module here and nothing else.

Datasets still served by the frozen pre-restructure loader
(:mod:`pfns4neurostim.data.legacy_io`) are dispatched to it unchanged; ``5d_rat`` was
carved out on 2026-09-25 when its cohort was replaced (see :mod:`.rat_5d`).

Loaders may report a **data cohort** — a stamp naming the exact raw files a channel
came from. It travels in ``ChannelData.meta['data_cohort']`` into the cell-cache
identity, so replacing a dataset's raw files invalidates that dataset's cached results
and leaves every other dataset's untouched.
"""
from __future__ import annotations

from typing import Any

from . import rat_5d

__all__ = ["load_subject", "data_cohort", "rat_5d"]

#: Datasets with a native loader in this package; anything else goes to ``legacy_io``.
_NATIVE: dict[str, Any] = {"5d_rat": rat_5d.load_subject}


def load_subject(dataset: str, subject: int, data_root: str = "./data") -> dict[str, Any]:
    """Load one subject's raw data dictionary.

    Args:
        dataset: Dataset name (``'nhp'``, ``'rat'``, ``'spinal'``, ``'5d_rat'``).
        subject: Subject index within the dataset.
        data_root: Directory holding the raw ``.mat`` trees.

    Returns:
        The raw dictionary (``sorted_resp``, ``sorted_respMean``, ``sorted_isvalid``,
        ``ch2xy``, ``grid_shape``, ...) that :mod:`pfns4neurostim.data.preprocessing`
        consumes.
    """
    loader = _NATIVE.get(dataset)
    if loader is not None:
        return loader(subject, data_root)

    from ..legacy_io import load_data  # noqa: PLC0415 - frozen loader, imported on use

    return load_data(dataset, subject, data_root=data_root)


def data_cohort(dataset: str) -> str | None:
    """Return the cohort stamp of a dataset's raw files, when its loader declares one.

    Args:
        dataset: Dataset name.

    Returns:
        The stamp, or ``None`` for a dataset whose loader does not version its files.
    """
    return rat_5d.COHORT if dataset == "5d_rat" else None
