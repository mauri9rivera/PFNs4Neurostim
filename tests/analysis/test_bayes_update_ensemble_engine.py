"""The ensemble TabPFN engine of the update analysis (``tabpfn_v2_5_e8``, added 2026-11-13).

Pins the two properties the Narval run relies on: every ``n_estimators = 1`` cell keeps its identity (so no cached
cell is recomputed or shadowed), and the ensemble engine gets an identity of its own and really builds 8 members.
"""
from __future__ import annotations

import copy

import pytest

pytest.importorskip("tabpfn")

from pfns4neurostim.analysis import bayes_update as B  # noqa: E402
from pfns4neurostim.experiments import mechanism as M  # noqa: E402


def _params() -> dict:
    """A resolved-enough ``bayes_update`` block for the engine helpers (the defaults, central tail measure)."""
    p = copy.deepcopy(M.BAYES_UPDATE_DEFAULTS)
    p["tabpfn"]["tail_measure"] = "central"
    return p


def test_deployed_engine_identity_is_unchanged() -> None:
    """The deployed engine's identity carries no ``n_estimators`` key: pre-2026-11-13 cells keep hitting."""
    ident = M._engine_params("tabpfn_v2_5", _params(), draw=3)
    assert "n_estimators" not in ident
    assert ident["inference_seed"] == 3
    assert ident["tail_measure"] == "central"


def test_ensemble_engine_has_its_own_identity() -> None:
    """``tabpfn_v2_5_e8`` differs from the deployed engine only by ``n_estimators = 8``."""
    p = _params()
    e1 = M._engine_params("tabpfn_v2_5", p, draw=0)
    e8 = M._engine_params("tabpfn_v2_5_e8", p, draw=0)
    assert e8.pop("n_estimators") == 8
    assert e8 == e1


def test_make_engine_sets_the_ensemble_size() -> None:
    """The factory passes the ensemble size to the regressor; the deployed engine stays at 1."""
    p = _params()
    assert M._make_engine("tabpfn_v2_5", p, "cpu").reg.n_estimators == 1
    assert M._make_engine("tabpfn_v2_5_e8", p, "cpu").reg.n_estimators == 8


def test_refit_engine_rejects_an_empty_ensemble() -> None:
    """``n_estimators < 1`` fails at construction rather than inside TabPFN."""
    with pytest.raises(ValueError, match="n_estimators"):
        B.PFNRefitEngine(n_estimators=0)
