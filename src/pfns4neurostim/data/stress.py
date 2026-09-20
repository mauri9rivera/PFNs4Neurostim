"""Stress knobs for the Hypothesis B regime sweeps.

A knob is a pure, seeded transformation of a :class:`~.channels.ChannelData` at a
given *level*, plus a report of what the transformation actually achieved:

    stressed = knob.apply(channel, level, rng)   # new ChannelData, no mutation
    row.update(knob.achieved(stressed))          # e.g. {'achieved_snr_db': 12.4}

The sweep runner never branches on which knob it is running: it iterates
``knob.levels``, records ``knob.achieved(...)``, and the figure layer reads its
x-axis from ``visualization.style.KNOB_X_AXIS``. Adding a knob is therefore one
subclass with two methods, and no change anywhere else.

Implemented: **K2 SNR** (residual amplification), **K5 outlier contamination**,
**K6 sparsity** (BO budget and electrode dropout). Declared but not yet
implemented: K1 decoy (needs the Demo 1 generator) and K7 spatial shuffle. Each
placeholder carries its name, levels and label key so configs, schemas and
figures can reference it, and raises a message naming the step that implements it.

Knobs differ in *what* they alter, which the ``alters`` class attribute records:
``trials`` (K2, K5) rewrites the observation bank, ``pool`` (K6 dropout) restricts
which sites may be queried without touching the ground truth, and ``budget``
(K6 budget) changes only how many queries the loop gets.

Roadmap: ``.claude/roadmap.md`` sections S1-S7; plan: task #10.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Sequence

import numpy as np

from dataclasses import dataclass, replace

from .channels import ChannelData
from .snr import achieved_snr_db

__all__ = [
    "StressKnob",
    "KnobNotApplicable",
    "CalibratedLevel",
    "calibrate_levels",
    "floor_snr_db",
    "K2SNRKnob",
    "K1DecoyKnob",
    "K5OutlierKnob",
    "K6BudgetKnob",
    "K6DropoutKnob",
    "K7ShuffleKnob",
    "KNOB_REGISTRY",
    "register_knob",
    "build_knob",
    "available_knobs",
]


class KnobNotApplicable(RuntimeError):
    """Raised when a knob cannot be applied to a particular channel.

    Distinct from a bug: some channels simply lack what a knob needs (a channel
    with no lab-flagged trials cannot be contaminated with real artefacts). The
    sweep runner catches this, logs the skip, and carries on with the rest of the
    grid rather than losing the whole run.
    """


class StressKnob(ABC):
    """One stress dimension of the Hyp B regime sweep.

    Subclasses declare their identity and default ladder as class variables and
    implement :meth:`apply`. :meth:`achieved` defaults to reporting achieved SNR,
    which every knob can report and which is the K2 x-axis.

    Attributes:
        name: Registry key, e.g. ``'k2_snr'``; also the ``knob`` column value.
        default_levels: Pre-registered level ladder, ordered from mildest to
            most severe stress.
        nominal_level: The level that must reproduce the unstressed channel.
        alters: Which part of the problem the knob touches — ``'trials'``,
            ``'pool'``, ``'budget'`` or ``'map'``. The runner uses this to know
            whether a level changes the BO budget rather than the data.
        implemented: False for declared-but-pending knobs.
    """

    name: ClassVar[str] = ""
    default_levels: ClassVar[tuple[float, ...]] = ()
    nominal_level: ClassVar[float] = 1.0
    alters: ClassVar[str] = "trials"
    implemented: ClassVar[bool] = True

    def __init__(self, levels: Sequence[float] | None = None) -> None:
        """Create the knob.

        Args:
            levels: Level ladder overriding ``default_levels`` (from the config).

        Raises:
            ValueError: If an empty ladder is given.
        """
        chosen = tuple(float(v) for v in (levels if levels is not None else self.default_levels))
        if not chosen:
            raise ValueError(f"{type(self).__name__}: empty level ladder.")
        self.levels: tuple[float, ...] = chosen

    @abstractmethod
    def apply(self, channel: ChannelData, level: float, rng: np.random.Generator) -> ChannelData:
        """Return a stressed copy of ``channel`` at ``level``.

        Implementations must not mutate ``channel`` and must be deterministic
        given ``rng``.

        Args:
            channel: Nominal channel.
            level: Knob level.
            rng: Seeded generator; the only source of randomness allowed.

        Returns:
            A new :class:`ChannelData`.
        """

    def achieved(self, channel: ChannelData) -> dict[str, float]:
        """Report what the stressed channel actually realizes.

        Args:
            channel: The stressed channel returned by :meth:`apply`.

        Returns:
            Columns merged into the tidy result row.
        """
        return {"achieved_snr_db": achieved_snr_db(channel)}

    def budget_for(self, level: float, budget: int) -> int:
        """Return the BO budget at this level (only K6-budget changes it).

        Args:
            level: Knob level.
            budget: Configured budget.

        Returns:
            Budget in total queries including ``n_init`` (P0.3).
        """
        return int(budget)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(levels={self.levels})"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
KNOB_REGISTRY: dict[str, type[StressKnob]] = {}


def register_knob(cls: type[StressKnob]) -> type[StressKnob]:
    """Register a knob class under its ``name``.

    Args:
        cls: Knob class with a non-empty ``name``.

    Returns:
        ``cls`` unchanged, so this works as a decorator.

    Raises:
        ValueError: On a missing or duplicate name.
    """
    if not cls.name:
        raise ValueError(f"register_knob: {cls.__name__} has no name.")
    if cls.name in KNOB_REGISTRY:
        raise ValueError(f"register_knob: duplicate knob name {cls.name!r}.")
    KNOB_REGISTRY[cls.name] = cls
    return cls


def build_knob(
    name: str,
    levels: Sequence[float] | None = None,
    **params: Any,
) -> StressKnob:
    """Construct a registered knob by name.

    Args:
        name: Registry key, e.g. ``'k2_snr'``.
        levels: Level ladder overriding the knob's default.
        **params: Knob-specific parameters (e.g. K5's ``source``). An unknown
            parameter raises, rather than being silently ignored.

    Returns:
        The knob instance.

    Raises:
        KeyError: If the name is not registered.
        TypeError: If a parameter is not one the knob declares.
        NotImplementedError: If the knob is a declared placeholder.
    """
    if name not in KNOB_REGISTRY:
        raise KeyError(
            f"Unknown knob {name!r}. Registered: {sorted(KNOB_REGISTRY)}."
        )
    cls = KNOB_REGISTRY[name]
    try:
        knob = cls(levels, **params)
    except TypeError as exc:
        raise TypeError(
            f"Knob {name!r} does not accept {sorted(params)}: {exc}"
        ) from exc
    if not cls.implemented:
        raise NotImplementedError(
            f"Knob {name!r} is declared but not implemented yet "
            f"({cls.__doc__.strip().splitlines()[0] if cls.__doc__ else ''})"
        )
    return knob


def available_knobs(*, implemented_only: bool = False) -> list[str]:
    """List registered knob names.

    Args:
        implemented_only: Exclude declared placeholders.

    Returns:
        Sorted knob names.
    """
    return sorted(
        name for name, cls in KNOB_REGISTRY.items() if cls.implemented or not implemented_only
    )


# ---------------------------------------------------------------------------
# K2 — SNR (implemented)
# ---------------------------------------------------------------------------
@register_knob
class K2SNRKnob(StressKnob):
    """K2: in-vivo residual amplification (Demo 2).

    Rescales each trial's deviation from its site's ground truth:

        y~[s, r] = y_gt[s] + alpha * (y[s, r] - y_gt[s])

    The ground truth is untouched, so regret and R-squared keep their meaning and
    only the observation noise changes. Invalid (NaN) trials stay NaN. alpha < 1
    is the trial-averaging direction (alpha = 1/sqrt(m) mimics averaging m
    trials), alpha = 1 is the nominal anchor, and alpha > 1 degrades SNR by
    exactly -20*log10(alpha) dB.

    Additive Gaussian noise is deliberately *not* used here: real neurostim noise
    is heteroscedastic (SD proportional to the mean), and amplifying the measured
    residuals preserves that structure. The additive variant is a control that
    belongs to the Demo 1 synthetic generator.
    """

    name: ClassVar[str] = "k2_snr"
    default_levels: ClassVar[tuple[float, ...]] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
    nominal_level: ClassVar[float] = 1.0
    alters: ClassVar[str] = "trials"

    def apply(self, channel: ChannelData, level: float, rng: np.random.Generator) -> ChannelData:
        """Amplify within-site residuals by ``level``.

        Args:
            channel: Nominal channel.
            level: Amplification factor alpha > 0.
            rng: Unused (the transform is deterministic) but part of the contract.

        Returns:
            Stressed channel with the same ground truth and NaN mask.

        Raises:
            ValueError: If ``level <= 0``.
            RuntimeError: If amplification produces non-finite values.
        """
        alpha = float(level)
        if alpha <= 0.0:
            raise ValueError(f"K2SNRKnob: alpha must be > 0, got {alpha}.")

        Y = np.asarray(channel.Y_trials, dtype=np.float64)          # [N, R]
        gt = np.asarray(channel.y_gt, dtype=np.float64)[:, None]    # [N, 1]
        stressed = gt + alpha * (Y - gt)                            # [N, R], NaN preserved
        finite_before = np.isfinite(Y)
        if not np.isfinite(stressed[finite_before]).all():
            raise RuntimeError(
                f"K2SNRKnob({channel.label}, alpha={alpha}): amplification produced "
                "non-finite values from finite inputs."
            )
        return channel.with_trials(
            stressed,
            stress={"knob": self.name, "level": alpha},
        )


# ---------------------------------------------------------------------------
# Declared placeholders — see task #10 Steps 8-9
# ---------------------------------------------------------------------------
class _PlaceholderKnob(StressKnob):
    """Base for knobs that are declared in the schema but not yet implemented."""

    implemented: ClassVar[bool] = False

    def apply(self, channel: ChannelData, level: float, rng: np.random.Generator) -> ChannelData:
        """Always raises; the knob exists only so configs and figures can name it."""
        raise NotImplementedError(
            f"Knob {self.name!r} is not implemented yet. See task #10 in "
            ".claude/task_plan.md for the step that implements it."
        )


@register_knob
class K1DecoyKnob(_PlaceholderKnob):
    """K1 decoy peak (Demo 1 only): second hotspot at separation d, ratio a2/a1.

    Implemented at task #10 Step 9, after the synthetic generator (S0), since a
    second hotspot cannot be injected into a real measured map.
    """

    name: ClassVar[str] = "k1_decoy"
    default_levels: ClassVar[tuple[float, ...]] = (0.5, 0.7, 0.85, 0.95, 0.98)
    nominal_level: ClassVar[float] = 0.5
    alters: ClassVar[str] = "map"

    def achieved(self, channel: ChannelData) -> dict[str, float]:
        """Report amplitude ratio and hotspot separation (to be filled in by S1)."""
        raise NotImplementedError("K1 achieved-metrics land with task #10 Step 9.")


@register_knob
class K5OutlierKnob(StressKnob):
    """K5: contaminate a fraction of trials with artefacts.

    A fraction ``epsilon`` of the valid trial slots is overwritten with outliers.
    Two sources:

    * ``invalid`` (default, Demo 2) — real trials the lab flagged
      ``sorted_isvalid == 0``. These are the artefacts the experiment actually
      produces (stimulation bleed-through, movement, saturation), so they beat any
      synthetic heavy tail for realism.
    * ``heavy_tail`` (Demo 1, and the fallback) — draws from a Student-t with
      ``df`` degrees of freedom, scaled to ``scale`` times the site's own trial
      spread. Used where the dataset carries no validity flags.

    **Donor scarcity is real and is reported.** Measured 2026-09-20: lab-flagged
    trials are 0.41% of NHP trial slots, 0.11% of spinal, 6.60% of 5d_rat. Donors
    are therefore sampled **with replacement** — a bootstrap of the channel's own
    artefact distribution — and ``achieved`` reports ``n_donor_trials`` so a cell
    resting on a handful of distinct artefacts is visible in the output rather
    than hidden. A channel with no donors raises :class:`KnobNotApplicable`; use
    ``source='heavy_tail'`` to sweep those channels anyway.
    """

    name: ClassVar[str] = "k5_outliers"
    default_levels: ClassVar[tuple[float, ...]] = (0.0, 0.05, 0.1, 0.2, 0.4)
    nominal_level: ClassVar[float] = 0.0
    alters: ClassVar[str] = "trials"

    def __init__(
        self,
        levels: Sequence[float] | None = None,
        *,
        source: str = "invalid",
        df: float = 2.0,
        scale: float = 5.0,
    ) -> None:
        """Create the knob.

        Args:
            levels: Contamination fractions to sweep.
            source: ``'invalid'`` (real lab-flagged artefacts) or ``'heavy_tail'``.
            df: Degrees of freedom of the Student-t, for ``heavy_tail``.
            scale: Multiple of the site's trial spread, for ``heavy_tail``.

        Raises:
            ValueError: On an unknown source or a level outside [0, 1].
        """
        super().__init__(levels)
        if source not in ("invalid", "heavy_tail"):
            raise ValueError(f"K5OutlierKnob: unknown source {source!r}.")
        for level in self.levels:
            if not 0.0 <= level <= 1.0:
                raise ValueError(f"K5OutlierKnob: level must be in [0, 1], got {level}.")
        self.source = source
        self.df = float(df)
        self.scale = float(scale)

    def apply(self, channel: ChannelData, level: float, rng: np.random.Generator) -> ChannelData:
        """Overwrite a fraction ``level`` of valid trials with artefacts.

        Args:
            channel: Nominal channel.
            level: Contamination fraction in [0, 1].
            rng: Seeded generator.

        Returns:
            Stressed channel; ground truth and NaN mask are unchanged.

        Raises:
            KnobNotApplicable: With ``source='invalid'`` on a channel that has no
                lab-flagged trials.
        """
        Y = np.asarray(channel.Y_trials, dtype=np.float64).copy()   # [N, R]
        valid = np.isfinite(Y)                                      # [N, R]
        n_valid = int(valid.sum())
        n_replace = int(round(float(level) * n_valid))
        n_donors = 0

        if n_replace > 0:
            rows, cols = np.nonzero(valid)
            pick = rng.choice(rows.size, size=min(n_replace, rows.size), replace=False)
            target_rows, target_cols = rows[pick], cols[pick]

            if self.source == "invalid":
                donors = self._donor_pool(channel)
                n_donors = int(donors.size)
                # With replacement: the donor pool is tiny (see the class docstring),
                # so this is a bootstrap of the channel's own artefact distribution.
                values = donors[rng.integers(0, donors.size, size=target_rows.size)]
            else:
                spread = np.nanstd(Y, axis=1)                        # [N]
                spread = np.where(np.isfinite(spread) & (spread > 0), spread, 1.0)
                draws = rng.standard_t(self.df, size=target_rows.size)
                values = Y[target_rows, target_cols] + self.scale * spread[target_rows] * draws
                n_donors = -1   # not applicable for the synthetic source

            Y[target_rows, target_cols] = values

        if not np.isfinite(Y[valid]).all():
            raise RuntimeError(
                f"K5OutlierKnob({channel.label}): contamination produced non-finite values."
            )
        return channel.with_trials(
            Y,
            stress={
                "knob": self.name,
                "level": float(level),
                "source": self.source,
                "n_replaced": int(n_replace),
                "n_donor_trials": int(n_donors),
            },
        )

    def _donor_pool(self, channel: ChannelData) -> np.ndarray:
        """Return the channel's lab-flagged artefact values.

        Args:
            channel: The channel.

        Returns:
            Finite artefact values, shape [n_donors].

        Raises:
            KnobNotApplicable: If the channel has no flagged trials.
        """
        if channel.Y_invalid is None:
            raise KnobNotApplicable(
                f"{channel.label}: dataset carries no validity flags, so K5 has no real "
                "artefacts to draw from. Use source='heavy_tail' to sweep it anyway."
            )
        donors = channel.Y_invalid[np.isfinite(channel.Y_invalid)]
        if donors.size == 0:
            raise KnobNotApplicable(
                f"{channel.label}: no lab-flagged invalid trials, so real-artefact "
                "contamination is impossible here (measured 2026-09-20: most NHP and "
                "spinal channels have none). Use source='heavy_tail' for this channel."
            )
        return np.asarray(donors, dtype=np.float64)

    def achieved(self, channel: ChannelData) -> dict[str, float]:
        """Report achieved SNR plus the contamination actually realized.

        Args:
            channel: The stressed channel.

        Returns:
            ``achieved_snr_db``, ``achieved_contamination`` and ``n_donor_trials``
            (the last is -1 for the synthetic source, where donors do not apply).
        """
        out = {"achieved_snr_db": achieved_snr_db(channel)}
        stress = channel.stress
        n_valid = int(np.isfinite(channel.Y_trials).sum())
        if stress and n_valid:
            out["achieved_contamination"] = float(stress.get("n_replaced", 0)) / n_valid
            out["n_donor_trials"] = float(stress.get("n_donor_trials", 0))
        return out


@register_knob
class K6BudgetKnob(StressKnob):
    """K6 sparsity via BO budget: explicit iteration counts (never a % of grid).

    The level *is* the budget, in total queries including ``n_init`` (P0.3). The
    data is untouched: this knob asks how performance degrades when the experiment
    simply gets fewer stimulations, which is the constraint that actually binds in
    an operating room.

    Expressing the level as an iteration count rather than a fraction of the grid
    is deliberate (P0.3): a "20% budget" means something different on a 96-site
    array than on a 2048-condition 5D grid, and reviewers cannot compare them.
    """

    name: ClassVar[str] = "k6_budget"
    default_levels: ClassVar[tuple[float, ...]] = (10.0, 20.0, 30.0, 50.0)
    nominal_level: ClassVar[float] = 50.0
    alters: ClassVar[str] = "budget"

    def apply(self, channel: ChannelData, level: float, rng: np.random.Generator) -> ChannelData:
        """Return the channel unchanged, recording the budget as provenance.

        Args:
            channel: Nominal channel.
            level: The BO budget for this level.
            rng: Unused.

        Returns:
            The same data with stress provenance attached.

        Raises:
            ValueError: If the level is not a positive whole number of queries.
        """
        if level < 1 or float(level) != int(level):
            raise ValueError(
                f"K6BudgetKnob: level is a query count and must be a positive integer, got {level}."
            )
        return channel.with_trials(
            channel.Y_trials.copy(),
            stress={"knob": self.name, "level": float(level)},
        )

    def budget_for(self, level: float, budget: int) -> int:
        """Return the level itself as the budget (total queries incl. ``n_init``).

        Args:
            level: Knob level.
            budget: The config budget, ignored by this knob.

        Returns:
            The budget for this level.
        """
        return int(level)

    def achieved(self, channel: ChannelData) -> dict[str, float]:
        """Report achieved SNR and the realized budget.

        Args:
            channel: The stressed channel.

        Returns:
            ``achieved_snr_db`` and ``achieved_budget``.
        """
        out = {"achieved_snr_db": achieved_snr_db(channel)}
        if channel.stress:
            out["achieved_budget"] = float(channel.stress.get("level", float("nan")))
        return out


@register_knob
class K6DropoutKnob(StressKnob):
    """K6 sparsity via random electrode dropout from the candidate pool.

    A fraction of sites becomes unqueryable — broken electrodes, or an array that
    simply has fewer contacts. The sites are **masked, not deleted**: the ground
    truth still spans the full map, so regret is measured against the true optimum
    even when that optimum is among the dropped electrodes. Deleting the rows
    instead would quietly redefine the target and make dropout look harmless
    (regret against the best *remaining* site) precisely when it hurts most.
    """

    name: ClassVar[str] = "k6_dropout"
    default_levels: ClassVar[tuple[float, ...]] = (0.0, 0.1, 0.25, 0.5)
    nominal_level: ClassVar[float] = 0.0
    alters: ClassVar[str] = "pool"

    def apply(self, channel: ChannelData, level: float, rng: np.random.Generator) -> ChannelData:
        """Mask a random fraction ``level`` of sites as unqueryable.

        Args:
            channel: Nominal channel.
            level: Dropout fraction in [0, 1).
            rng: Seeded generator.

        Returns:
            Stressed channel with a ``queryable`` mask.

        Raises:
            ValueError: If the level is outside [0, 1).
            KnobNotApplicable: If fewer than two sites would remain.
        """
        if not 0.0 <= level < 1.0:
            raise ValueError(f"K6DropoutKnob: level must be in [0, 1), got {level}.")
        n = channel.n_sites
        n_drop = int(round(float(level) * n))
        if n - n_drop < 2:
            raise KnobNotApplicable(
                f"{channel.label}: dropout {level} would leave {n - n_drop} of {n} sites; "
                "at least two are needed for a search problem."
            )
        queryable = np.ones(n, dtype=bool)                          # [N]
        if n_drop:
            queryable[rng.choice(n, size=n_drop, replace=False)] = False
        return replace(
            channel,
            queryable=queryable,
            stress={"knob": self.name, "level": float(level), "n_dropped": int(n_drop)},
        )

    def achieved(self, channel: ChannelData) -> dict[str, float]:
        """Report achieved SNR, realized dropout and how many sites remain.

        Args:
            channel: The stressed channel.

        Returns:
            ``achieved_snr_db``, ``achieved_dropout`` and ``n_queryable``.
        """
        out = {"achieved_snr_db": achieved_snr_db(channel)}
        out["achieved_dropout"] = 1.0 - channel.n_queryable / channel.n_sites
        out["n_queryable"] = float(channel.n_queryable)
        return out


@register_knob
class K7ShuffleKnob(_PlaceholderKnob):
    """K7 spatial-shuffle structure knob f: destroy coordinate-response binding.

    Implemented at task #10 Step 8 by lifting
    ``utils.data_utils.shuffle_response_pairing`` (already used by the legacy
    ``mechanistic_ablation.py``) onto the knob contract.
    """

    name: ClassVar[str] = "k7_shuffle"
    default_levels: ClassVar[tuple[float, ...]] = (0.0, 0.25, 0.5, 0.75, 1.0)
    nominal_level: ClassVar[float] = 0.0
    alters: ClassVar[str] = "map"


# ---------------------------------------------------------------------------
# SNR-targeted calibration (per-dataset ladders)
# ---------------------------------------------------------------------------
def floor_snr_db(channel: ChannelData) -> float:
    """Return a channel's unstressed ("floor") achieved SNR in dB.

    Args:
        channel: Nominal channel.

    Returns:
        Achieved SNR of the channel before any knob is applied.
    """
    return achieved_snr_db(channel)


@dataclass(frozen=True)
class CalibratedLevel:
    """One solved knob level and how close it got to its SNR target.

    Attributes:
        target_db: Requested SNR change in dB.
        level: The knob level chosen.
        achieved_db: The SNR change that level actually produces on this channel.
        seed: The RNG seed the calibration used. Applying the knob with this seed
            reproduces ``achieved_db`` exactly; using any other stream re-rolls the
            randomness and the realized degradation drifts from the stated ladder.
        resolved: Whether ``achieved_db`` is within tolerance of ``target_db``.
            False means the knob **cannot** hit that target on this channel. For
            K5 with real artefacts the response is a step function: one replaced
            trial is the smallest possible move and can already cost double-digit
            dB. Such a level must be reported, not silently plotted as if it were
            the requested severity.
    """

    target_db: float
    level: float
    achieved_db: float
    seed: int
    resolved: bool


def calibrate_levels(
    knob: StressKnob,
    channel: ChannelData,
    targets_db: Sequence[float],
    rng: np.random.Generator,
    *,
    bracket: tuple[float, float] | None = None,
    tol_db: float = 0.5,
    max_iter: int = 40,
) -> list[CalibratedLevel]:
    """Solve for the knob levels that degrade this channel's SNR by given amounts.

    A fixed level ladder means different things on different datasets: the same
    contamination fraction that barely dents a clean channel can annihilate a
    noisy one (measured 2026-09-20: 20% real-artefact contamination moved one NHP
    channel from -3.5 dB to -31.1 dB). Specifying the ladder as **dB of degradation
    relative to each channel's own floor** makes a sweep comparable across
    datasets, and matches the axis the K2 figures already use.

    Each target is solved by bisection on the level using the knob's own achieved
    SNR, so it works for any knob whose severity is monotone in its level without
    the calibration knowing which knob it is. Where the response is a **step
    function**, the search returns the closest level it found and flags
    ``resolved=False`` rather than pretending the target was met.

    Args:
        knob: The knob to calibrate.
        channel: The channel to calibrate against (levels are per channel).
        targets_db: Target SNR changes in dB, e.g. ``[0, -1, -2, -4]``. Zero maps
            to the knob's nominal level exactly, with no search.
        rng: Seeded generator; one child seed is drawn for the whole calibration.
        bracket: ``(low, high)`` level bracket; defaults to the knob's own ladder.
        tol_db: How close counts as resolved.
        max_iter: Bisection iterations per target.

    Returns:
        One :class:`CalibratedLevel` per target, in the order given.

    Raises:
        ValueError: If a target is positive (knobs degrade, they do not improve).
        KnobNotApplicable: Propagated from ``knob.apply``.
    """
    targets = [float(t) for t in targets_db]
    if any(t > 0.0 for t in targets):
        raise ValueError(
            f"calibrate_levels: targets are SNR *degradations* and must be <= 0 dB, got {targets}."
        )
    floor = floor_snr_db(channel)
    lo, hi = bracket if bracket is not None else (
        min(knob.nominal_level, min(knob.levels)),
        max(knob.levels),
    )

    # One child seed for the whole calibration: every trial level is applied with
    # the same randomness, so the search sees a smooth curve rather than
    # Monte-Carlo noise, and repeated calibrations reproduce exactly.
    child_seed = int(rng.integers(0, 2**31 - 1))

    def delta_at(level: float) -> float:
        """SNR change in dB produced by this level on this channel.

        Args:
            level: Knob level to evaluate.

        Returns:
            Achieved SNR minus the channel floor, in dB.
        """
        stressed = knob.apply(channel, level, np.random.default_rng(child_seed))
        return achieved_snr_db(stressed) - floor

    out: list[CalibratedLevel] = []
    for target in targets:
        if target == 0.0:
            nominal = float(knob.nominal_level)
            out.append(CalibratedLevel(target, nominal, delta_at(nominal), child_seed, True))
            continue

        low, high = float(lo), float(hi)
        best_level, best_delta = high, delta_at(high)
        if best_delta > target:
            # The knob cannot reach this target anywhere within its bracket.
            out.append(CalibratedLevel(target, high, best_delta, child_seed, False))
            continue

        for _ in range(max_iter):
            mid = 0.5 * (low + high)
            delta = delta_at(mid)
            if abs(delta - target) < abs(best_delta - target):
                best_level, best_delta = mid, delta
            if abs(delta - target) <= tol_db:
                break
            if delta > target:      # not degraded enough yet -> stronger level
                low = mid
            else:
                high = mid
        out.append(
            CalibratedLevel(
                target,
                float(best_level),
                float(best_delta),
                child_seed,
                abs(best_delta - target) <= tol_db,
            )
        )
    return out
