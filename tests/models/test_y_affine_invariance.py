"""How invariant is a surrogate to an affine rescaling of its training targets?

This is the claim the online y scaler leans on. ``OnlineYScaler`` refits the y transform inside the BO loop
and maps predictions back before any metric (see :mod:`tests.data.test_online_scaling`), so a surrogate that
is affine-invariant in y cannot be moved by ``experiment.online_y_scaler`` at all, and the audit arms then
measure what causal scaling costs the models that are NOT invariant (the GP family, whose priors are stated
in the target's units).

**Measured 2026-09-30, and it corrects the earlier claim that "TabPFN is exactly invariant".** TabPFN-2.5 is
exactly invariant to a pure *rescaling* of y, and its posterior *mean* is invariant to a shift as well (to
~1e-6 of the target range), so every decision -- acquisition ranking, recommendation, regret, R-squared -- is
unmoved. Its predictive *standard deviation* is not: it depends on where the targets sit, not only on their
spread, and the deviation grows with the offset measured in units of the observed spread. On the grid below:

    offset / range      0.1      0.5      1.0      5.3     52.7
    median |dsd| / sd    0 %   0.02 %   0.04 %    1.1 %    159 %
    max    |dsd| / sd    0 %    0.5 %    1.0 %     30 %   1206 %

Both online modes subtract an offset (the running mean for ``zscore``, the running minimum for ``minmax``),
so the operating regime is offset ~ O(range) and the effect on sigma is ~1 % at the tail -- except early in a
run, when a nearly flat initial design makes the observed spread tiny and the offset large in those units.
The consequence for reporting: regret and R-squared stay comparable across ``online_y_scaler`` settings for
TabPFN, and **calibration (coverage, ECE, NLL, CRPS) does not** -- on top of NLL already being incomparable
because the units change.

The TabPFN tests are marked ``slow`` because they load real weights; they run on CPU and need no GPU.
"""
from __future__ import annotations

import numpy as np
import pytest

from pfns4neurostim.data.preprocessing import OnlineYScaler
from pfns4neurostim.models.registry import build_surrogate

#: Offsets, in units of the training targets' range, at which sigma is still expected to hold to 2 %.
#: Both online modes live here: their offset is a running mean or minimum of the observations so far.
OPERATING_OFFSETS: tuple[float, ...] = (0.1, 0.5, 1.0)

#: Offset far outside that regime, kept as an explicit negative case: it must NOT be invariant, so a change
#: that made it look invariant (a wrapper that stopped inverting something) fails loudly.
PATHOLOGICAL_OFFSET: float = 50.0

#: Tolerances, set from the 2026-09-30 measurements with an order of magnitude of headroom: in the operating
#: regime the mean moves by <= 5e-6 of the range and sigma by <= 1 %, and float32 rounding at a 1000x rescale
#: contributes ~5e-7 of the range / ~2e-4 of sigma.
MEAN_ATOL_FRACTION_OF_RANGE: float = 1e-4
#: Even at the pathological offset the mean stays this close (measured 3e-4 of the range), which is why the
#: DECISION-level claim survives a regime where sigma does not.
MEAN_ATOL_FRACTION_PATHOLOGICAL: float = 1e-3
SIGMA_RTOL: float = 0.02
#: Pure-rescaling agreement, loose only by float32 rounding (measured 1.7e-4 at 1000x).
RESCALE_RTOL: float = 1e-3


@pytest.fixture()
def toy_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A smooth 2D bump on a scattered design: ``(X_train, y_train, X_query)``."""
    rng = np.random.default_rng(0)
    X = rng.random((60, 2))                                           # [60, 2]
    y = np.exp(-((X[:, 0] - 0.7) ** 2 + (X[:, 1] - 0.3) ** 2) / 0.1)  # [60]
    return X[:24], y[:24], X


def _predict_in_original_units(
    model: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    scale: float,
    offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit on ``scale * y + offset``, predict, and map the prediction back to the original units.

    Exactly what :func:`evaluation.bo_loop._fit_scaled` and ``_predict_unscaled`` do around a surrogate,
    with the affine map fixed instead of refitted at every step.

    Args:
        model: Registry key of the surrogate.
        X_train: Training coordinates, shape [t, D].
        y_train: Training targets in original units, shape [t].
        X_query: Query coordinates, shape [N, D].
        scale: Positive multiplier applied to the targets.
        offset: Additive offset applied to the targets.

    Returns:
        ``(mean, std)`` in the original target units, each shape [N].
    """
    surrogate = build_surrogate(model, device="cpu")
    surrogate.fit(X_train, scale * y_train + offset)
    mean, std = surrogate.predict_marginals(X_query)     # [N], [N]
    return (mean - offset) / scale, std / scale


@pytest.mark.slow
class TestTabPFNUnderPureRescaling:
    """A pure scale change washes out completely: TabPFN standardizes the spread of y internally."""

    @pytest.mark.parametrize("scale", [0.25, 2.0, 100.0])
    def test_mean_and_sigma_are_unchanged(
        self,
        toy_problem: tuple[np.ndarray, np.ndarray, np.ndarray],
        scale: float,
    ) -> None:
        X_train, y_train, X_query = toy_problem
        base = _predict_in_original_units("tabpfn_v2_5", X_train, y_train, X_query, 1.0, 0.0)
        got = _predict_in_original_units("tabpfn_v2_5", X_train, y_train, X_query, scale, 0.0)
        y_range = float(np.ptp(y_train))
        # Exact (bitwise 0.0) up to a few-fold rescale; float32 rounding enters above ~10x.
        assert np.allclose(got[0], base[0], rtol=RESCALE_RTOL, atol=1e-5 * y_range)
        assert np.allclose(got[1], base[1], rtol=RESCALE_RTOL, atol=1e-9)


@pytest.mark.slow
class TestTabPFNUnderAShift:
    """A shift leaves every decision alone but moves sigma; the size of the move is what is pinned here."""

    def test_the_mean_is_invariant(self, toy_problem: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        X_train, y_train, X_query = toy_problem
        y_range = float(np.ptp(y_train))
        base_mean, _ = _predict_in_original_units("tabpfn_v2_5", X_train, y_train, X_query, 1.0, 0.0)
        for frac in OPERATING_OFFSETS + (PATHOLOGICAL_OFFSET,):
            mean, _ = _predict_in_original_units(
                "tabpfn_v2_5", X_train, y_train, X_query, 1.0, frac * y_range
            )
            delta = float(np.max(np.abs(mean - base_mean))) / y_range
            tol = (MEAN_ATOL_FRACTION_PATHOLOGICAL if frac == PATHOLOGICAL_OFFSET
                   else MEAN_ATOL_FRACTION_OF_RANGE)
            assert delta < tol, (
                f"TabPFN's mean moved by {delta:.2e} of the range at offset {frac}x range; regret and R^2 "
                "are only comparable across online_y_scaler settings because this stays negligible"
            )

    def test_the_recommendation_is_unmoved(
        self,
        toy_problem: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """The recommendation is ``argmax`` of the posterior mean: the decision-level form of the claim."""
        X_train, y_train, X_query = toy_problem
        y_range = float(np.ptp(y_train))
        picks = {
            int(np.argmax(_predict_in_original_units(
                "tabpfn_v2_5", X_train, y_train, X_query, 1.0, frac * y_range)[0]))
            for frac in (0.0,) + OPERATING_OFFSETS
        }
        assert len(picks) == 1, f"TabPFN recommended different sites under a shift of y: {picks}"

    @pytest.mark.parametrize("frac", OPERATING_OFFSETS)
    def test_sigma_holds_to_two_percent_in_the_operating_regime(
        self,
        toy_problem: tuple[np.ndarray, np.ndarray, np.ndarray],
        frac: float,
    ) -> None:
        X_train, y_train, X_query = toy_problem
        y_range = float(np.ptp(y_train))
        _, base_std = _predict_in_original_units("tabpfn_v2_5", X_train, y_train, X_query, 1.0, 0.0)
        _, std = _predict_in_original_units("tabpfn_v2_5", X_train, y_train, X_query, 1.0, frac * y_range)
        rel = np.abs(std - base_std) / np.maximum(base_std, 1e-12)
        assert float(np.max(rel)) < SIGMA_RTOL, (
            f"TabPFN's sigma moved by {float(np.max(rel)):.1%} at offset {frac}x range: calibration is more "
            "scaling-sensitive than the 2026-09-30 measurement recorded"
        )

    def test_sigma_is_not_invariant_at_a_pathological_offset(
        self,
        toy_problem: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """The documented limit of the claim: sigma depends on where the targets sit, not only their spread.

        This is not a defect to fix but a property to remember -- it is why a calibration number may not be
        compared across ``online_y_scaler`` settings, and why an early-run cell whose initial design is
        nearly flat (tiny observed spread, hence a large offset in those units) is the risky case.
        """
        X_train, y_train, X_query = toy_problem
        y_range = float(np.ptp(y_train))
        _, base_std = _predict_in_original_units("tabpfn_v2_5", X_train, y_train, X_query, 1.0, 0.0)
        _, std = _predict_in_original_units(
            "tabpfn_v2_5", X_train, y_train, X_query, 1.0, PATHOLOGICAL_OFFSET * y_range
        )
        rel = np.abs(std - base_std) / np.maximum(base_std, 1e-12)
        assert float(np.max(rel)) > SIGMA_RTOL


@pytest.mark.slow
class TestTheGPIsNotAffineInvariant:
    """The counterpart claim, and the reason the audit arms exist at all.

    A GP with FIXED hyperparameters cannot follow an affine map of y, because its prior variance is stated in
    the target's units. If this ever passes for ``gp_naive``, the online-y arms have nothing to measure.
    """

    def test_gp_fixed_predictions_depend_on_the_target_units(
        self,
        toy_problem: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        X_train, y_train, X_query = toy_problem
        base_mean, _ = _predict_in_original_units("gp_naive", X_train, y_train, X_query, 1.0, 0.0)
        mean, _ = _predict_in_original_units("gp_naive", X_train, y_train, X_query, 20.0, 100.0)
        assert not np.allclose(mean, base_mean, rtol=1e-2, atol=1e-2 * float(np.ptp(y_train)))


class TestTheScalerItselfIsExactlyAffine:
    """No weights needed: the transform the loop applies is affine and invertible in closed form."""

    @pytest.mark.parametrize("mode", ["minmax", "zscore"])
    def test_round_trip_recovers_an_affine_image(self, mode: str) -> None:
        y = np.array([1.0, 2.0, 5.0, 3.0, 2.5])
        scaler = OnlineYScaler(mode).fit(y)
        transformed = scaler.transform(y)
        # An affine map has a constant ratio of successive differences.
        ratios = np.diff(transformed) / np.diff(y)
        assert np.allclose(ratios, ratios[0])
        assert np.allclose(scaler.inverse_mean(transformed), y)

    @pytest.mark.parametrize("mode", ["minmax", "zscore"])
    def test_the_offset_is_what_breaks_a_surrogate_s_invariance(self, mode: str) -> None:
        """Both modes shift as well as scale, which is why the shift tests above are the relevant ones."""
        y = np.array([1.0, 2.0, 5.0, 3.0, 2.5])
        assert OnlineYScaler(mode).fit(y).offset != 0.0
