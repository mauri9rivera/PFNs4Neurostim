"""Stress knobs for the Hypothesis B regime sweeps.

A knob is a pure, seeded transformation of a :class:`~.channels.ChannelData` at a
given *level*, plus a report of what the transformation actually achieved:

    stressed = knob.apply(channel, level, rng)   # new ChannelData, no mutation
    row.update(knob.achieved(stressed))          # e.g. {'achieved_snr_db': 12.4}

The sweep runner never branches on which knob it is running: it iterates
``knob.levels``, records ``knob.achieved(...)``, and the figure layer reads its
x-axis from ``visualization.style.KNOB_X_AXIS``. Adding a knob is therefore one
subclass with two methods, and no change anywhere else.

Implemented (stress design of 2026-09-23, restructured 2026-10-19): **K2** in two
settings (channel-relative residual amplification against each channel's floor, and
one global absolute noise level for every channel), **K5** epsilon-contamination of
the trial slots with a heavy tail, and **K6** electrode failure during the run. The
K1 decoy peak, the K6 budget knob and the K7 spatial shuffle were removed on
2026-10-19; the BO budget is a plain config key, and its effect on robustness is
read off the per-step traces of the other sweeps. No placeholder knobs remain;
``_PlaceholderKnob`` is kept for the next declared-but-pending one, which carries
its name, levels and label key so configs, schemas and figures can reference it
before it computes anything.

Knobs differ in *what* they alter, which the ``alters`` class attribute records:
``trials`` (K2, K5) rewrites the observation bank and ``pool`` (K6 failure) changes
what some sites return during the run.

Roadmap: ``.claude/roadmap.md`` sections S1-S7; plan: task #10.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Sequence

import numpy as np

from dataclasses import dataclass, replace

from .channels import ChannelData
from .snr import achieved_snr_db, noise_power

__all__ = [
    "StressKnob",
    "KnobNotApplicable",
    "CalibratedLevel",
    "calibrate_levels",
    "floor_snr_db",
    "K2ChannelNoiseKnob",
    "K2GlobalNoiseKnob",
    "K5OutlierKnob",
    "K6FailureKnob",
    "KNOB_REGISTRY",
    "register_knob",
    "build_knob",
    "available_knobs",
]


class KnobNotApplicable(RuntimeError):
    """Raised when a knob cannot be applied to a particular channel.

    Distinct from a bug: some channels simply lack what a knob needs (a small
    array cannot lose 90% of its electrodes and remain a search problem). The
    sweep runner catches this, logs the skip, and carries on with the rest of the
    grid rather than losing the whole run.
    """


class StressKnob(ABC):
    """One stress dimension of the Hyp B regime sweep.

    Subclasses declare their identity and default ladder as class variables and
    implement :meth:`apply`. :meth:`achieved` defaults to reporting achieved SNR,
    which every knob can report and which is the K2 x-axis.

    Attributes:
        name: Registry key, e.g. ``'k2_channel'``; also the ``knob`` column value.
        default_levels: Pre-registered level ladder, ordered from mildest to
            most severe stress.
        nominal_level: The level that must reproduce the unstressed channel.
            A knob whose nominal level is not a fixed number leaves this NaN
            and overrides :meth:`nominal_in`.
        symbol: Mathtext symbol of the knob's level, the canonical notation of
            the manuscript (settled 2026-10-12). Every knob has its own letter:
            ``gamma`` and ``eta`` for the two K2 settings rather than one
            ``alpha``, and ``varepsilon`` / ``varphi`` for the two fractions, so
            no symbol means two different things across figures.
        alters: Which part of the problem the knob touches — ``'trials'`` or
            ``'pool'``.
        implemented: False for declared-but-pending knobs.
        seed_key: Namespace of the knob's RNG streams in ``seeding.seed_for``.
            Defaults to ``name``; a renamed knob keeps its old key so its seeds,
            and therefore its cached cells, survive the rename.
    """

    name: ClassVar[str] = ""
    default_levels: ClassVar[tuple[float, ...]] = ()
    nominal_level: ClassVar[float] = 1.0
    symbol: ClassVar[str] = ""
    alters: ClassVar[str] = "trials"
    implemented: ClassVar[bool] = True
    seed_key: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Default ``seed_key`` to the subclass's ``name``."""
        super().__init_subclass__(**kwargs)
        if not cls.__dict__.get("seed_key"):
            cls.seed_key = cls.name

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
        given ``rng`` and the channel.

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

    @classmethod
    def nominal_in(cls, levels: Sequence[float]) -> float | None:
        """Return the unstressed level of a realized ladder, or ``None`` if it has none.

        The figure layer marks this level as the **default regime** of the
        dataset, so it has to be resolvable from a run's tidy frame alone (the
        ``--replot`` path never rebuilds the knob). Most knobs have a fixed
        nominal level; a knob without one overrides this to resolve it from the
        realized ladder.

        Args:
            levels: The levels actually present in the run, in any order.

        Returns:
            The nominal level, or ``None`` when the ladder does not contain it
            (a sweep that never visits the unstressed regime).
        """
        if not np.isfinite(cls.nominal_level):
            return None
        for level in levels:
            if np.isclose(float(level), cls.nominal_level):
                return float(level)
        return None

    def solve_level(self, channel: ChannelData, target_db: float) -> float | None:
        """Return the level that shifts ``channel``'s SNR by exactly ``target_db``, if one exists.

        Knobs whose SNR shift inverts in closed form override this; the default
        ``None`` makes :func:`calibrate_levels` refuse a dB ladder for the knob.

        Args:
            channel: Channel the level is solved for.
            target_db: SNR change relative to the channel's floor, in dB.

        Returns:
            The level, or ``None`` when the knob has no exact inversion.
        """
        return None

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
        name: Registry key, e.g. ``'k2_channel'``.
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
# K2 — SNR, two settings (restructured 2026-09-23)
# ---------------------------------------------------------------------------
@register_knob
class K2ChannelNoiseKnob(StressKnob):
    """K2-channel: noise relative to each channel's own floor SNR (draws).

    Rescales each trial's deviation from its site's ground truth:

        y~[s, r] = y_gt[s] + alpha * (y[s, r] - y_gt[s])

    The ground truth is untouched, so regret and R-squared keep their meaning and
    only the observation noise changes. Invalid (NaN) trials stay NaN. alpha < 1
    is the trial-averaging direction, alpha = 1 the nominal anchor, and the
    achieved SNR moves by exactly ``-20*log10(alpha)`` dB relative to the
    channel's floor, whatever that floor is. Every channel therefore receives the
    *same relative* damage; contrast :class:`K2GlobalNoiseKnob`.

    Because the shift is analytic, :meth:`solve_level` inverts a dB target
    exactly, so a ladder can be stated in dB (``knob.targets_db``) and may go
    anywhere, including below 0 dB of absolute SNR.
    """

    name: ClassVar[str] = "k2_channel"
    seed_key: ClassVar[str] = "k2_snr"   # pre-2026-09-23 name: keeps seeds and cached cells
    default_levels: ClassVar[tuple[float, ...]] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
    nominal_level: ClassVar[float] = 1.0
    symbol: ClassVar[str] = r"$\gamma$"
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
            raise ValueError(f"K2ChannelNoiseKnob: alpha must be > 0, got {alpha}.")
        Y = np.asarray(channel.Y_trials, dtype=np.float64)          # [N, R]
        gt = np.asarray(channel.y_gt, dtype=np.float64)[:, None]    # [N, 1]
        stressed = gt + alpha * (Y - gt)                            # [N, R], NaN preserved
        _check_finite(stressed, Y, f"{type(self).__name__}({channel.label}, alpha={alpha})")
        return channel.with_trials(stressed, stress={"knob": self.name, "level": alpha})

    def solve_level(self, channel: ChannelData, target_db: float) -> float:
        """Return the alpha that shifts this channel's SNR by ``target_db`` exactly.

        Args:
            channel: The channel (unused: the shift does not depend on it).
            target_db: SNR change in dB; negative degrades, positive improves.

        Returns:
            ``alpha = 10 ** (-target_db / 20)``.
        """
        return float(10.0 ** (-float(target_db) / 20.0))


@register_knob
class K2GlobalNoiseKnob(StressKnob):
    """K2-global: one absolute noise level added to every trial of every channel.

    Each valid trial receives independent Gaussian noise of standard deviation
    ``alpha`` in **z-scored response units**:

        y~[s, r] = y[s, r] + alpha * e[s, r],   e ~ N(0, 1)

    Every channel is standardized on its own trials, so the same ``alpha`` is the
    same absolute noise on every channel. It is *not* proportional to a channel's
    floor SNR: a clean channel loses more dB than a noisy one at the same level,
    which is precisely what distinguishes this setting from
    :class:`K2ChannelNoiseKnob`. Achieved SNR is reported per channel and is free
    to fall below 0 dB.
    """

    name: ClassVar[str] = "k2_global"
    default_levels: ClassVar[tuple[float, ...]] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    nominal_level: ClassVar[float] = 0.0
    symbol: ClassVar[str] = r"$\eta$"
    alters: ClassVar[str] = "trials"

    def __init__(self, levels: Sequence[float] | None = None) -> None:
        """Create the knob.

        Args:
            levels: Added-noise standard deviations (z-scored units) to sweep.

        Raises:
            ValueError: On a negative level.
        """
        super().__init__(levels)
        if any(level < 0.0 for level in self.levels):
            raise ValueError(f"K2GlobalNoiseKnob: levels must be >= 0, got {self.levels}.")

    def apply(self, channel: ChannelData, level: float, rng: np.random.Generator) -> ChannelData:
        """Add Gaussian noise of standard deviation ``level`` to every valid trial.

        Args:
            channel: Nominal channel.
            level: Noise standard deviation in z-scored units, >= 0.
            rng: Seeded generator.

        Returns:
            Stressed channel with the same ground truth and NaN mask.
        """
        alpha = float(level)
        Y = np.asarray(channel.Y_trials, dtype=np.float64)          # [N, R]
        stressed = Y + alpha * rng.standard_normal(Y.shape)         # [N, R], NaN preserved
        _check_finite(stressed, Y, f"{type(self).__name__}({channel.label}, alpha={alpha})")
        return channel.with_trials(stressed, stress={"knob": self.name, "level": alpha})



def _check_finite(stressed: np.ndarray, original: np.ndarray, where: str) -> None:
    """Fail fast if a knob turned finite trials into non-finite ones.

    Args:
        stressed: Stressed trial bank, shape [N, R].
        original: Trial bank before the knob, shape [N, R].
        where: Knob and channel, for the message.

    Raises:
        RuntimeError: On any non-finite value where the original was finite.
    """
    if not np.isfinite(stressed[np.isfinite(original)]).all():
        raise RuntimeError(f"{where}: stress produced non-finite values from finite inputs.")


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
class K5OutlierKnob(StressKnob):
    """K5: Huber epsilon-contamination of the trial bank with heavy-tailed artefacts.

    A fraction ``epsilon`` of **all valid trial slots** of the channel (sites x
    repetitions, e.g. 2048 x 8 on ``5d_rat``) is replaced by an artefact

        y~[s, r] = y_gt[s] + scale * sigma_noise * t,   t ~ Student-t(df)

    where ``sigma_noise`` is the channel's within-site trial standard deviation.
    Two choices make the same epsilon mean the same thing on every dataset
    (decision 2026-09-23): epsilon counts slots, not trials per site, so datasets
    with 20 and 8 repetitions are contaminated in the same proportion; and the
    source is the same heavy tail everywhere. Real lab-flagged artefacts are no
    longer used: they are 0.41% of NHP slots against 6.60% of ``5d_rat``, so a
    donor-based knob measured the datasets' artefact supply, not the models.
    """

    name: ClassVar[str] = "k5_outliers"
    default_levels: ClassVar[tuple[float, ...]] = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)
    nominal_level: ClassVar[float] = 0.0
    symbol: ClassVar[str] = r"$\varepsilon$"
    alters: ClassVar[str] = "trials"

    def __init__(
        self,
        levels: Sequence[float] | None = None,
        *,
        df: float = 3.0,
        scale: float = 5.0,
    ) -> None:
        """Create the knob.

        Args:
            levels: Contamination fractions epsilon in [0, 1] to sweep.
            df: Degrees of freedom of the Student-t contaminant.
            scale: Contaminant scale in multiples of the channel's trial SD.

        Raises:
            ValueError: On a level outside [0, 1] or a non-positive df/scale.
        """
        super().__init__(levels)
        for level in self.levels:
            if not 0.0 <= level <= 1.0:
                raise ValueError(f"K5OutlierKnob: level must be in [0, 1], got {level}.")
        if df <= 0.0 or scale <= 0.0:
            raise ValueError(f"K5OutlierKnob: df and scale must be > 0, got df={df}, scale={scale}.")
        self.df = float(df)
        self.scale = float(scale)

    def apply(self, channel: ChannelData, level: float, rng: np.random.Generator) -> ChannelData:
        """Replace a fraction ``level`` of the valid trial slots with artefacts.

        Args:
            channel: Nominal channel.
            level: Contamination fraction epsilon in [0, 1].
            rng: Seeded generator.

        Returns:
            Stressed channel; ground truth and NaN mask are unchanged.
        """
        Y = np.asarray(channel.Y_trials, dtype=np.float64).copy()   # [N, R]
        rows, cols = np.nonzero(np.isfinite(Y))                     # [n_valid] each
        n_replace = int(round(float(level) * rows.size))
        if n_replace > 0:
            pick = rng.choice(rows.size, size=n_replace, replace=False)
            sigma = float(np.sqrt(noise_power(channel)))
            draws = rng.standard_t(self.df, size=n_replace)         # [n_replace]
            Y[rows[pick], cols[pick]] = channel.y_gt[rows[pick]] + self.scale * sigma * draws
        _check_finite(Y, channel.Y_trials, f"{type(self).__name__}({channel.label})")
        return channel.with_trials(
            Y,
            stress={"knob": self.name, "level": float(level), "n_replaced": n_replace},
        )

    def achieved(self, channel: ChannelData) -> dict[str, float]:
        """Report achieved SNR and the contamination actually realized.

        Args:
            channel: The stressed channel.

        Returns:
            ``achieved_snr_db`` and ``achieved_contamination`` (replaced / valid slots).
        """
        n_valid = int(np.isfinite(channel.Y_trials).sum())
        return {
            "achieved_snr_db": achieved_snr_db(channel),
            "achieved_contamination": float(channel.stress.get("n_replaced", 0)) / max(n_valid, 1),
        }


@register_knob
class K6FailureKnob(StressKnob):
    """K6 sparsity via electrode failure during the run.

    One mask covers a fraction ``epsilon`` of the electrodes. Each masked
    electrode fails at its own random moment of the run, drawn uniformly over the
    run's progress (fraction of the budget spent), and from then on returns
    ``dead_value`` (0.0 in z-scored units, i.e. the channel's mean response: an
    uninformative reading) for every remaining query. A failed electrode stays
    queryable: the optimizer is not told, and has to notice from the data.

    Scoring (decision 2026-09-23): regret, exploration, identification, R-squared
    and calibration are computed on the **surviving** electrodes, since those are
    the only ones a clinician could still use at the end of the session. The
    failure times are stored as run fractions, so the knob is independent of the
    budget.
    """

    name: ClassVar[str] = "k6_failure"
    default_levels: ClassVar[tuple[float, ...]] = (0.0, 0.1, 0.25, 0.5)
    nominal_level: ClassVar[float] = 0.0
    symbol: ClassVar[str] = r"$\varphi$"
    alters: ClassVar[str] = "pool"

    def __init__(self, levels: Sequence[float] | None = None, *, dead_value: float = 0.0) -> None:
        """Create the knob.

        Args:
            levels: Failed-electrode fractions epsilon in [0, 1) to sweep.
            dead_value: Response of a failed electrode, in z-scored units.

        Raises:
            ValueError: On a level outside [0, 1).
        """
        super().__init__(levels)
        for level in self.levels:
            if not 0.0 <= level < 1.0:
                raise ValueError(f"K6FailureKnob: level must be in [0, 1), got {level}.")
        self.dead_value = float(dead_value)

    def apply(self, channel: ChannelData, level: float, rng: np.random.Generator) -> ChannelData:
        """Draw which electrodes fail and when.

        Args:
            channel: Nominal channel.
            level: Failed-electrode fraction epsilon in [0, 1).
            rng: Seeded generator.

        Returns:
            Stressed channel with ``failure_time`` set.

        Raises:
            KnobNotApplicable: If fewer than two electrodes would survive.
        """
        n = channel.n_sites
        n_fail = int(round(float(level) * n))
        if n - n_fail < 2:
            raise KnobNotApplicable(
                f"{channel.label}: failure fraction {level} would leave {n - n_fail} of {n} "
                "electrodes; at least two are needed for a search problem."
            )
        failure_time = np.full(n, np.inf)                           # [N], inf = never fails
        failing = rng.choice(n, size=n_fail, replace=False)
        failure_time[failing] = rng.uniform(0.0, 1.0, size=n_fail)
        return replace(
            channel,
            failure_time=failure_time,
            failure_value=self.dead_value,
            stress={"knob": self.name, "level": float(level), "n_failed": n_fail},
        )

    def achieved(self, channel: ChannelData) -> dict[str, float]:
        """Report achieved SNR, the realized failure fraction and the survivors.

        Args:
            channel: The stressed channel.

        Returns:
            ``achieved_snr_db``, ``achieved_failure`` and ``n_survivors``.
        """
        n_survivors = int(channel.survivors.sum())
        return {
            "achieved_snr_db": achieved_snr_db(channel),
            "achieved_failure": 1.0 - n_survivors / channel.n_sites,
            "n_survivors": float(n_survivors),
        }


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
    """One knob level solved from an SNR target on one channel.

    Attributes:
        target_db: Requested SNR change relative to the channel floor, in dB.
        level: The knob level that realizes it.
        achieved_db: The SNR change that level actually produces on this channel.
        seed: RNG seed the level is applied with, so the run realizes exactly
            ``achieved_db``.
    """

    target_db: float
    level: float
    achieved_db: float
    seed: int


def calibrate_levels(
    knob: StressKnob,
    channel: ChannelData,
    targets_db: Sequence[float],
    rng: np.random.Generator,
) -> list[CalibratedLevel]:
    """Solve the knob levels that shift this channel's SNR by the given amounts.

    A ladder stated in **dB relative to each channel's own floor** gives every
    channel the same relative damage (the K2-channel setting). Only knobs whose
    SNR shift inverts exactly (``solve_level``) accept such a ladder: an
    approximate search on a stochastic knob would state a severity the run does
    not realize. Targets may be positive (improvement) and the resulting absolute
    SNR may fall below 0 dB.

    Args:
        knob: The knob; its ``solve_level`` must return a level (not ``None``).
        channel: The channel to calibrate against (levels are per channel).
        targets_db: SNR changes in dB, e.g. ``[6, 0, -6, -12, -18]``.
        rng: Seeded generator; one child seed is drawn for the whole ladder.

    Returns:
        One :class:`CalibratedLevel` per target, in the order given.

    Raises:
        ValueError: If the knob cannot invert a dB target exactly.
        KnobNotApplicable: Propagated from ``knob.apply``.
    """
    floor = floor_snr_db(channel)
    child_seed = int(rng.integers(0, 2**31 - 1))
    out: list[CalibratedLevel] = []
    for target in (float(t) for t in targets_db):
        level = knob.solve_level(channel, target)
        if level is None:
            raise ValueError(
                f"Knob {knob.name!r} has no exact dB inversion; state its ladder in knob.levels."
            )
        level = float(level)
        stressed = knob.apply(channel, level, np.random.default_rng(child_seed))
        out.append(CalibratedLevel(target, level, achieved_snr_db(stressed) - floor, child_seed))
    return out
