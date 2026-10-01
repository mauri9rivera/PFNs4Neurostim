"""The deployment portfolio: every Mila job still to run, in priority order, with wall-time estimates.

The agent never submits jobs. This script PRINTS the plan, or with ``--emit-bash`` writes a one-command
submission script that the USER runs on the login node:

    python scripts/portfolio.py                                                    # the plan (all groups)
    python scripts/portfolio.py --emit-bash --group stress > scripts/submit_portfolio.sh     # Hyp B (restructured knobs, Demo 1)
    python scripts/portfolio.py --emit-bash --group bench > scripts/submit_bench.sh          # Hyp 0/A leftovers
    python scripts/portfolio.py --emit-bash --group hypc > scripts/submit_hypc.sh            # Hyp C mechanism analyses
    python scripts/portfolio.py --emit-bash --group externals > scripts/submit_externals.sh  # TabFM, PFNs4BO
    python scripts/portfolio.py --emit-bash --group spinal > scripts/submit_spinal.sh        # every spinal deliverable
    python scripts/portfolio.py --emit-bash --group nhp > scripts/submit_nhp.sh              # NHP re-run under the canonical y scaling
    python scripts/portfolio.py --emit-bash --group audit > scripts/submit_audit.sh          # y-scaling sensitivity arms

Groups (updated 2026-09-25; units whose results already exist were removed — see task_plan.md "Your Mila portfolio"):
    stress     the 5d_rat stress sweeps on the noOutliers cohort (K2 channel/global, K5, K6 failure, Demo 1 K2).
    bench      Hyp A and the Hyp 0 acquisition tables on 5d_rat.
    hypc       the three NHP mechanism analyses plus the 5d_rat placement arm, re-run with the context-size
               sweeps of task #18 (2026-09-30). C1-C3 last ran locally on 2026-09-24/25 at a SINGLE context
               size each. Each is ONE process (``scripts/run_single.sh``) with no cell cache, so a unit that
               hits the 12 h limit loses its work: check the estimates before submitting, and split a unit
               by config rather than hoping.
    externals  the PFN benchmark: base models on 5d_rat (bench env), TabFM (bench env, 24 GB, <= 2 lanes, NHP rerun plus a
               5d_rat calibration job), PFNs4BO 5d_rat (main env) and TabPFN v1 (v1 env, tabpfn<2).
    spinal     every spinal deliverable (Hyp A, Hyp 0, PFN benchmark, all stress knobs); needs data/spinal
               staged on the cluster first. Hyp C (mechanism) is not included: NHP-scoped, and one
               non-resumable process per analysis would exceed the 12 h limit on 90 channels.
    nhp        every NHP deliverable, recomputed under the canonical `online_y_scaler: minmax` (2026-09-30).
               The mode is part of cell identity, so the 2026-09-24 NHP cells are unaddressable; the re-run also
               produces the gp_* fit diagnostics that P0.10 / guardrail G1 need.
    audit      the two y-scaling sensitivity arms (zscore) and the TabPFN invariance check against the archived
               offline cells. CPU-cheap. TabPFN's decisions are y-affine-invariant; its predictive sigma is NOT
               (measured 2026-09-30, tests/models/test_y_affine_invariance.py), so calibration is never compared
               across modes.

Sharded experiments (``bo_benchmark``, ``stress_sweep``) write every job's cells to the shared cache and one assemble job
builds tidy.csv, tables and figures. ``mechanism`` runs as ONE process (``scripts/run_single.sh``) and
writes its own outputs. Estimates use per-repetition costs measured on 2026-09-20/23.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

#: Seconds per repetition at the deliverable budget (96 NHP / 100 5d_rat), per model. Absent = unmeasured.
REP_SECONDS: dict[str, dict[str, float]] = {
    "nhp": {"tabpfn_v2_5": 15.7, "gp_mll": 19.5, "gp_naive": 0.8, "random": 0.2, "tabicl": 30.0, "tabfm": 99.0,
            "pfns4bo": 5.3},
    "5d_rat": {"tabpfn_v2_5": 17.6, "gp_mll": 19.5, "gp_naive": 1.0, "random": 0.4, "tabicl": 33.0, "pfns4bo": 6.7},
    # Spinal (budget 64 on the 8x8 grid) is NOT measured on Mila: the NHP costs above scaled by a local
    # one-rep spinal/NHP timing (2026-09-25: TabPFN-2.5 x0.52, GP-MLL x0.60; the GP factor for the rest).
    "spinal": {"tabpfn_v2_5": 8.2, "gp_mll": 11.7, "gp_naive": 0.5, "random": 0.1, "tabicl": 16.5, "pfns4bo": 3.2},
}
GUESSED: frozenset[str] = frozenset()   # every listed cost is measured (TabICL from the D3 runs: 0.43-0.52 s/step)
#: Channels (subject x EMG) per dataset. 5d_rat dropped from 18 to 17 with the
#: noOutliers cohort (2026-09-25): rData03 left and flag_valid_emg replaced the
#: hand-maintained EMG lists.
CHANNELS = {"nhp": 18, "5d_rat": 17, "spinal": 100}   # spinal: 11 subjects, 100 channels
REPS = 20      # raised with the configs on 2026-09-30; n_reps is not part of cell identity, so reps 0-9
               # of any already-computed cell are served from cache and only 10-19 are new work
GPU_LANE_GAIN = 1.7      # aggregate speed-up of several lanes on one GPU (measured with 3 lanes)
GPU_LANES = 4
CPU_LANES = 8
#: Default --mem of the job scripts (run_bo_benchmark.sh / run_stress_sweep.sh); a unit states --mem only
#: when its models need more than this.
DEFAULT_JOB_MEM_GB = 10
BENCH_ENV = "pfns4neurostim-bench"
MAIN_ENV = "pfns4neurostim"
V1_ENV = "pfns4neurostim-v1"    # tabpfn<2; cannot share an interpreter with the pinned 6.3.2
#: Runners that run as one process and write their own deliverables (no lanes, no cell assembly).
SINGLE_PROCESS: frozenset[str] = frozenset({"mechanism"})


@dataclass(frozen=True)
class Unit:
    """One deliverable x dataset job group.

    Attributes:
        name: Label.
        group: ``stress``, ``bench``, ``hypc`` or ``externals``.
        experiment: ``bo_benchmark``, ``stress_sweep`` or ``mechanism``.
        config: Experiment YAML.
        dataset: ``nhp`` or ``5d_rat`` (keys of :data:`REP_SECONDS`).
        gpu_models: Models needing a GPU (may be empty).
        cpu_models: Models run on the CPU partition (may be empty).
        multiplier: Cells per repetition per model (knob levels, or a budget-scaling factor).
        env: Conda env for the cluster jobs.
        lanes: Lanes of the GPU job.
        mem: ``--mem`` of the GPU job (empty keeps the script's default).
        overrides: Extra ``key=value`` overrides passed to every job of the unit (e.g. a calibration subset).
        note: Free-text remark.
        hours: Wall-hour estimate of a single-process unit (sharded units are estimated from :data:`REP_SECONDS`).
        channels: Channels the unit's config selects, when fewer than the dataset's (:data:`CHANNELS`).
        cpu_only: Single-process unit that needs no GPU, so it goes to ``run_single_cpu.sh`` on ``main-cpu``
            instead of holding one of the two GPUs the per-user cap allows (the placement analysis declares
            ``device: cpu``).
    """

    name: str
    group: str
    experiment: str
    config: str
    dataset: str
    gpu_models: tuple[str, ...]
    cpu_models: tuple[str, ...]
    multiplier: float = 1.0
    env: str = MAIN_ENV
    lanes: int = GPU_LANES
    mem: str = ""
    overrides: tuple[str, ...] = ()
    note: str = ""
    hours: float | None = None
    channels: int | None = None
    cpu_only: bool = False


def _u(name: str, exp: str, cfg: str, ds: str, gpu: tuple[str, ...], cpu: tuple[str, ...], mult: float = 1.0, **kw: object) -> Unit:
    """Shorthand constructor for a unit (group defaults to ``stress``)."""
    return Unit(name, str(kw.pop("group", "stress")), exp, f"configs/experiment/{cfg}.yaml", ds, gpu, cpu, mult, **kw)  # type: ignore[arg-type]


_TP, _GP = ("tabpfn_v2_5",), ("gp_mll", "gp_naive")
_BASE_CPU = ("gp_mll", "gp_naive", "random")

UNITS: tuple[Unit, ...] = (
    # Rebuilt 2026-09-25. Finished units were removed (NHP stress S1a-S6, B3, E5 on 09-24; Hyp C C1-C3 run locally).
    # EVERY 5d_rat unit is a full recompute: the noOutliers cohort (2026-09-25) is part of each 5d_rat cell's
    # identity, so no 5d_rat cell of the old cohort can be reused. Clean the cluster first (runbook 3b).
    # ---- stress: 5d_rat on the new cohort, main env ----
    _u("S1b. K2-channel, 5d_rat", "stress_sweep", "stress_k2_channel_5d_rat", "5d_rat", _TP, _GP, 9),
    _u("S2b. K2-global, 5d_rat", "stress_sweep", "stress_k2_global_5d_rat", "5d_rat", _TP, _GP, 8),
    _u("S3b. K5 slot-fraction heavy tail, 5d_rat", "stress_sweep", "stress_k5_5d_rat", "5d_rat", _TP, _GP, 6),
    _u("S4b. K6 electrode failure, 5d_rat", "stress_sweep", "stress_k6_failure_5d_rat", "5d_rat", _TP, _GP, 5),
    _u("S5b. Demo 1 K2-channel, 5d_rat twins", "stress_sweep", "stress_k2_channel_demo1_5d_rat", "5d_rat", _TP, _GP, 9,
       note="twins inherit the source cohort; the s4-e2 collapse (2026-09-24) must be re-checked on the new cohort"),
    # ---- bench: Hyp 0 / A on 5d_rat, main env ----
    _u("B0. Hyp A (TabPFN vs GP), 5d_rat", "bo_benchmark", "hyp_a_5d_rat", "5d_rat", _TP, _BASE_CPU, 1, group="bench"),
    _u("B1. Acquisition core (ts/ei/ucb), 5d_rat", "bo_benchmark", "hyp0_acq_core_5d_rat", "5d_rat", _TP, _BASE_CPU, 3,
       group="bench", note="ts_marginal cells are shared with B0 (same identity): whichever runs second hits them"),
    _u("B2. UCB kappa grid, 5d_rat", "bo_benchmark", "hyp0_ucb_kappa_5d_rat", "5d_rat", _TP, (), 5, group="bench"),
    # ---- externals: the PFN benchmark (one run per dataset, cells from three environments) ----
    _u("E1. PFN bench base (TabPFN-2.5, TabICL / GP-MLL), 5d_rat", "bo_benchmark", "hyp0_pfn_bench_5d_rat", "5d_rat",
       ("tabpfn_v2_5", "tabicl"), ("gp_mll",), group="externals", env=BENCH_ENV,
       note="bench env (TabICL needs py3.11); TabPFN-2.5 / GP-MLL cells are shared with B0 and hit if B0 ran first"),
    _u("E3. TabFM, NHP (fixed wrapper)", "bo_benchmark", "hyp0_pfn_bench_nhp", "nhp", ("tabfm",), (), group="externals",
       env=BENCH_ENV,
       note="rerun: raw-scale mean + spread/out-of-fold sigma, batched members (2026-09-25; ~1.3 s/step locally); "
            "24 GB, <=2 lanes; sigma is constructed (G3)"),
    # A subset run (calibration, smoke) ALWAYS gets its own tag: a unit shares its run directory with every
    # other unit of the same config and tag, and its assemble job would overwrite the full run's tidy.csv with
    # the subset (happened to hyp0-pfn-bench-5d_rat on 2026-09-24).
    _u("E4. TabFM 5d_rat CALIBRATION (1 channel, 2 reps)", "bo_benchmark", "hyp0_pfn_bench_5d_rat", "5d_rat", ("tabfm",), (),
       group="externals", env=BENCH_ENV, lanes=1,
       overrides=("dataset.subjects=[1]", "dataset.emgs=[0]", "n_reps=2", "tag=5d_rat-tabfm-calibration"),
       note="re-measure with the fixed wrapper (old: ~18 min/rep, 12.8 GB RSS) before planning the full 5d_rat TabFM run"),
    _u("E6. PFNs4BO (native policy), 5d_rat", "bo_benchmark", "hyp0_pfn_bench_5d_rat", "5d_rat", ("pfns4bo",), (),
       group="externals", lanes=2,
       note="main env; cells land in the hyp0-pfn-bench run; ~10 min for 180 reps on 2 lanes (2026-09-24)"),
    # TabPFN v1 joins the same PFN benchmark run from its own environment (2026-09-25).
    _u("E7. TabPFN v1 (classification-head adaptation), NHP", "bo_benchmark", "hyp0_pfn_bench_nhp", "nhp",
       ("tabpfn_v1",), (), group="externals", env=V1_ENV, lanes=2, hours=None,
       note="cost UNMEASURED; v1 API unexecuted until the v1 env exists: build it and run ONE cell before submitting"),
    _u("E8. TabPFN v1 (classification-head adaptation), 5d_rat", "bo_benchmark", "hyp0_pfn_bench_5d_rat", "5d_rat",
       ("tabpfn_v1",), (), group="externals", env=V1_ENV, lanes=2, hours=None,
       note="cost UNMEASURED; after E7"),
    # ---- spinal: every deliverable (2026-09-25). Stage data/spinal on the cluster first (runbook). Budget 64 =
    # the 8x8 grid. Subject 5 has one trial per site on every EMG (no noise floor), so the SNR-based stress
    # sweeps run on the other 10 subjects (90 channels); the benchmarks keep all 100.
    # Units longer than 12 h requeue themselves at the time limit and resume from the cell cache.
    _u("P1. Hyp A (TabPFN vs GP), spinal", "bo_benchmark", "hyp_a_spinal", "spinal", _TP, _BASE_CPU, 1, group="spinal"),
    _u("P2. Acquisition core (ts/ei/ucb), spinal", "bo_benchmark", "hyp0_acq_core_spinal", "spinal", _TP, _BASE_CPU, 3,
       group="spinal", note="ts_marginal cells are shared with P1 (same identity)"),
    _u("P3. UCB kappa grid, spinal", "bo_benchmark", "hyp0_ucb_kappa_spinal", "spinal", _TP, (), 5, group="spinal"),
    _u("P5. K2-channel, spinal", "stress_sweep", "stress_k2_channel_spinal", "spinal", _TP, _GP, 9, group="spinal",
       channels=90),
    _u("P6. K2-global, spinal", "stress_sweep", "stress_k2_global_spinal", "spinal", _TP, _GP, 8, group="spinal",
       channels=90),
    _u("P7. K5 slot-fraction heavy tail, spinal", "stress_sweep", "stress_k5_spinal", "spinal", _TP, _GP, 6,
       group="spinal", channels=90),
    _u("P8. K6 electrode failure, spinal", "stress_sweep", "stress_k6_failure_spinal", "spinal", _TP, _GP, 5,
       group="spinal", channels=90),
    _u("P9. K6 budget, spinal", "stress_sweep", "stress_k6_budget_spinal", "spinal", _TP, _GP, 2.5, group="spinal",
       channels=90, note="levels 10,20,30,50,64 = ~2.5x one full budget"),
    _u("P10. Demo 1 K2-channel, spinal twins", "stress_sweep", "stress_k2_channel_demo1_spinal", "spinal", _TP, _GP, 9,
       group="spinal", channels=90, note="all 90 twins fit without collapse (checked 2026-09-25)"),
    _u("P11. Demo 1 K1 decoy, spinal twins", "stress_sweep", "stress_k1_decoy_spinal", "spinal", _TP, _GP, 5,
       group="spinal", channels=90, note="separation 3 pitches on an 8x8 grid: channels where it does not fit are skipped"),
    _u("P13. PFN bench base (TabPFN-2.5, TabICL / GP-MLL), spinal", "bo_benchmark", "hyp0_pfn_bench_spinal", "spinal",
       ("tabpfn_v2_5", "tabicl"), ("gp_mll",), group="spinal", env=BENCH_ENV,
       note="bench env; TabPFN-2.5 / GP-MLL cells are shared with P1"),
    _u("P14. PFNs4BO (native policy), spinal", "bo_benchmark", "hyp0_pfn_bench_spinal", "spinal", ("pfns4bo",), (),
       group="spinal", lanes=2, note="main env"),
    _u("P15. TabFM, spinal (fixed wrapper)", "bo_benchmark", "hyp0_pfn_bench_spinal", "spinal", ("tabfm",), (),
       group="spinal", env=BENCH_ENV,
       note="cost UNMEASURED on spinal: run after E3 and read its per-rep time; sigma is constructed (G3)"),
    _u("P16. TabPFN v1 (classification-head adaptation), spinal", "bo_benchmark", "hyp0_pfn_bench_spinal", "spinal",
       ("tabpfn_v1",), (), group="spinal", env=V1_ENV, lanes=2, hours=None,
       note="cost UNMEASURED; only after E7 has run one v1 cell successfully"),
    # ---- hypc: the mechanism analyses, one process each, context-size sweeps of task #18 (2026-09-30) ----
    # No cell cache: `mechanism` always recomputes, so these need no tag gymnastics -- but they are also NOT
    # resumable. Give a unit its own tag only to keep an older run directory; otherwise it is overwritten.
    _u("C1. M10 update rule (t = 10/25/50), NHP", "mechanism", "mechanism_update_rule_nhp", "nhp", _TP, (),
       group="hypc", hours=4.0,
       note="~7-8 min per channel (the GP-refit arm dominates) + gates; the layer arm now runs at THREE "
            "context sizes instead of one, so budget ~1 h above the 2026-09-24 run; set "
            "update_rule.link.tidy_csv afterwards for M8"),
    _u("C2. CKA (a) + placement ladder (t = 10/25/50/80), NHP", "mechanism", "mechanism_cka_nhp", "nhp", _TP, (),
       group="hypc", hours=5.5,
       note="(a) ~4.4 h, permutation-null bound (~97% of it) + (b) ~20 min after the 2026-09-30 restructure "
            "(banks, context sites, reference-map embeddings and the floor/ceiling are now shared across the "
            "channels of one grid: ~3.5 h before) + controls. Still CANNOT resume, so if it ever times out, "
            "drop cka.targets to [K_GT, K_GP] -- that halves (a)"),
    _u("C3. Placement MMD / W2 (t = 10...96), NHP", "mechanism", "mechanism_placement_nhp", "nhp", (), ("placement",),
       group="hypc", cpu_only=True, hours=3.0,
       note="CPU unit (device: cpu in the config). 6 context sizes instead of 4 and the ladder now reaches "
            "the full 96-site map; needs libs/tabpfn-v1-prior (bash scripts/mila_setup.sh submodules)"),
    _u("C4. Placement MMD / W2 (t = 10...200), 5d_rat", "mechanism", "mechanism_placement_5d_rat", "5d_rat", (),
       ("placement",), group="hypc", cpu_only=True, hours=None,
       note="cost UNMEASURED: the 5D LinearNDInterpolator on 2048 conditions has never been timed (#6 Step 8). "
            "Run C3 first, then ONE 5d_rat channel by hand before submitting the unit"),
    # ---- nhp: EVERY NHP deliverable, recomputed under the canonical online y scaling (2026-09-30) ----
    # `online_y_scaler: minmax` entered every experiment config on 2026-09-30, and it is part of each cell's
    # identity, so none of the 2026-09-24 NHP cells is addressable any more: these units are re-runs, not new
    # science. They are not waste -- fresh cells also carry the new gp_* fit diagnostics (P0.10 / G1), which is
    # why the separate "audit" recompute units D1/D2 are gone. The old `none` cells stay on disk untouched and
    # are the offline comparison arm; regret and R^2 may be compared across the two, calibration may NOT.
    _u("N1. K2-channel, NHP", "stress_sweep", "stress_k2_channel_nhp", "nhp", _TP, _GP, 9, group="nhp"),
    _u("N2. K2-global, NHP", "stress_sweep", "stress_k2_global_nhp", "nhp", _TP, _GP, 8, group="nhp"),
    _u("N3. K5 slot-fraction heavy tail, NHP", "stress_sweep", "stress_k5_nhp", "nhp", _TP, _GP, 6, group="nhp"),
    _u("N4. K6 electrode failure, NHP", "stress_sweep", "stress_k6_failure_nhp", "nhp", _TP, _GP, 5, group="nhp"),
    _u("N5. K6 budget, NHP", "stress_sweep", "stress_k6_budget_nhp", "nhp", _TP, _GP, 2.5, group="nhp",
       note="levels 10,20,30,50,96 = ~2.5x one full budget"),
    _u("N6. Demo 1 K2-channel, NHP twins", "stress_sweep", "stress_k2_channel_demo1_nhp", "nhp", _TP, _GP, 9,
       group="nhp", note="then --replot --bridge on the N1 run directory for S10"),
    _u("N7. Demo 1 K1 decoy, NHP twins", "stress_sweep", "stress_k1_decoy_nhp", "nhp", _TP, _GP, 5, group="nhp"),
    _u("N13. K7 spatial shuffle, NHP", "stress_sweep", "stress_k7_nhp", "nhp", _TP, _BASE_CPU, 6, group="nhp",
       note="NEW knob (2026-09-30), not a re-run: the only axis that removes spatial structure itself. Random "
            "search is an arm because it is the floor every model must meet at f = 1"),
    _u("N8. Hyp A (TabPFN vs GP), NHP", "bo_benchmark", "hyp_a_nhp", "nhp", _TP, _BASE_CPU, 1, group="nhp"),
    _u("N9. Acquisition core (ts/ei/ucb), NHP", "bo_benchmark", "hyp0_acq_core_nhp", "nhp", _TP, _BASE_CPU, 3,
       group="nhp", note="ts_marginal cells are shared with N8 (same identity)"),
    _u("N10. UCB kappa grid, NHP", "bo_benchmark", "hyp0_ucb_kappa_nhp", "nhp", _TP, (), 5, group="nhp"),
    _u("N11. PFN bench base (TabPFN-2.5, TabICL / GP-MLL), NHP", "bo_benchmark", "hyp0_pfn_bench_nhp", "nhp",
       ("tabpfn_v2_5", "tabicl"), ("gp_mll",), group="nhp", env=BENCH_ENV,
       note="bench env; the 2026-09-24 NHP PFN-bench numbers were offline-scaled and are superseded"),
    _u("N12. PFNs4BO (native policy), NHP", "bo_benchmark", "hyp0_pfn_bench_nhp", "nhp", ("pfns4bo",), (),
       group="nhp", lanes=2, note="main env; re-run of the 2026-09-24 cells under the canonical scaling"),
    # ---- audit: what the canonical scaling changed, and the invariance claim it rests on (2026-09-30) ----
    # Trimmed after the 2026-09-30 flip: the minmax arm IS the N-group now, and the GP-diagnostics recompute
    # units are subsumed by it. What remains is the zscore sensitivity arm and the invariance check.
    _u("Y1. Online y = zscore sensitivity arm, Hyp A NHP (GP arms)", "bo_benchmark", "hyp_a_nhp", "nhp", (),
       ("gp_mll", "gp_naive"), 1, group="audit",
       overrides=("online_y_scaler=zscore", "tag=nhp-onliney-z"),
       note="the third mode, for the appendix panel: minmax (canonical) vs zscore vs the archived offline runs"),
    _u("Y3. Online y = zscore sensitivity arm, K2-channel NHP (GP arms)", "stress_sweep",
       "stress_k2_channel_nhp", "nhp", (), ("gp_mll", "gp_naive"), 9, group="audit",
       overrides=("online_y_scaler=zscore", "tag=nhp-onliney-k2z"),
       note="does the choice of causal scaler move the GP breakdown point?"),
    _u("Y4. TabPFN invariance check vs the archived offline cells", "bo_benchmark", "hyp_a_nhp", "nhp", _TP, (), 1,
       group="audit", lanes=1, channels=2,
       overrides=("online_y_scaler=none", "dataset.subjects=[0]", "dataset.emgs=[0,1]", "n_reps=3",
                  "tag=nhp-offline-tpcheck"),
       note="now runs the OFFLINE mode (the canonical configs are minmax) and compares against N8: regret/R2 "
            "must match, sigma-based metrics need not -- TabPFN's decisions are y-affine-invariant, its "
            "predictive sigma is not (tests/models/test_y_affine_invariance.py). The estimate ignores "
            "n_reps=3, so the real cost is ~1 min"),
)


def _resources(models: tuple[str, ...]) -> tuple[str, str, int]:
    """Environment, ``--mem`` and lane cap a set of models needs, read from their specs (#8 rule 5).

    Model -> environment and model -> resources are data in ``models/pfn/external.py``; this function is the
    only place the portfolio consults them, so a unit never names an environment or a memory figure by hand.
    Built-in models (the GP family, TabPFN-2.5, random search) carry no external spec: they run in the main
    environment inside the default job memory.

    Args:
        models: Model keys of one job.

    Returns:
        ``(conda env, mem flag or '' for the script default, max lanes)``. The memory is the largest
        per-lane requirement in the set times the lane cap, rounded up, and only stated when it exceeds the
        job scripts' default.

    Raises:
        ValueError: If the models of one job do not agree on an environment (they must be split).
    """
    from pfns4neurostim.models.pfn.external import EXTERNAL_SPECS   # noqa: PLC0415 - optional import

    specs = [EXTERNAL_SPECS[m] for m in models if m in EXTERNAL_SPECS]
    if not specs:
        return MAIN_ENV, "", GPU_LANES
    envs = {sp.conda_env for sp in specs}
    if len(envs) > 1:
        raise ValueError(f"models {models} span environments {sorted(envs)}; split them into separate units.")
    lanes = min([GPU_LANES] + [sp.max_lanes for sp in specs])
    need_gb = max(sp.mem_per_lane_gb for sp in specs) * lanes
    mem = f"{int(need_gb + 0.999)}G" if need_gb > DEFAULT_JOB_MEM_GB else ""
    return envs.pop(), mem, lanes


def _check_resources() -> list[str]:
    """Compare every unit's declared env / mem / lanes with what its models' specs imply (#8 rule 5).

    A report rather than an exception: a unit may legitimately ask for fewer lanes than the cap (a
    calibration job), so only a *mismatch that would break the job* -- the wrong environment, or less memory
    than the models need -- is worth naming.

    Returns:
        One line per disagreement; empty when every unit agrees with its specs.
    """
    out: list[str] = []
    for unit in UNITS:
        models = unit.gpu_models + unit.cpu_models
        try:
            env, mem, lanes = _resources(models)
        except ValueError as exc:
            out.append(f"{unit.name}: {exc}")
            continue
        if env != unit.env:
            out.append(f"{unit.name}: env {unit.env!r} declared, specs say {env!r}")
        if unit.mem and mem and unit.mem != mem:
            out.append(f"{unit.name}: --mem={unit.mem!r} declared, specs imply {mem!r}")
    return out


def _script(experiment: str, cpu_only: bool = False) -> str:
    if experiment in SINGLE_PROCESS:
        return "scripts/run_single_cpu.sh" if cpu_only else "scripts/run_single.sh"
    return "scripts/run_bo_benchmark.sh" if experiment == "bo_benchmark" else "scripts/run_stress_sweep.sh"


def _hours(unit: Unit, models: tuple[str, ...], speedup: float) -> float | None:
    """Estimated wall hours, or ``None`` if any model's per-rep cost is unmeasured."""
    if unit.experiment in SINGLE_PROCESS:
        # A single-process unit's whole cost sits in one column: the CPU one when it needs no GPU.
        wanted = unit.cpu_models if unit.cpu_only else unit.gpu_models
        return unit.hours if models is wanted else 0.0
    if not models:
        return 0.0
    costs = [REP_SECONDS[unit.dataset].get(m) for m in models]
    if any(c is None for c in costs):
        return None
    channels = unit.channels or CHANNELS[unit.dataset]
    return sum(costs) * unit.multiplier * channels * REPS / 3600.0 / speedup


def _fmt(hours: float | None) -> str:
    return "?" if hours is None else f"{hours:.1f}"


def _selected(group: str) -> list[Unit]:
    return [u for u in UNITS if group == "all" or u.group == group]


def print_plan(group: str) -> None:
    """Print the human-readable plan."""
    for unit in _selected(group):
        _env, derived_mem, derived_lanes = _resources(unit.gpu_models + unit.cpu_models)
        lanes = min(unit.lanes, derived_lanes)
        mem = unit.mem or derived_mem
        gpu_h = _hours(unit, unit.gpu_models, GPU_LANE_GAIN if lanes > 1 else 1.0)
        cpu_h = _hours(unit, unit.cpu_models, CPU_LANES)
        resources = f", {lanes} lanes" + (f", --mem={mem}" if mem else "")
        print(f"\n### {unit.name}   [{unit.group}, env {unit.env}{resources}]   "
              f"est. wall: GPU ~{_fmt(gpu_h)} h, CPU ~{_fmt(cpu_h)} h")
        if unit.note:
            print(f"# {unit.note}")
    print("\n# Mila caps: 2 GPUs on `main` (extra GPU jobs queue), 8 CPUs on `main-cpu`. Watch: bash scripts/mila.sh queue")
    print("# Submit: bash scripts/submit_nhp.sh | submit_audit.sh | submit_portfolio.sh (stress) | "
          "submit_bench.sh | submit_hypc.sh | submit_externals.sh | submit_spinal.sh")
    print("# Or all of it, prerequisite-gated, in one command: bash scripts/submit_next.sh now")


def emit_bash(group: str) -> None:
    """Write the one-command submission script for ``group`` to stdout."""
    print("#!/bin/bash")
    print(f"# GENERATED by `python scripts/portfolio.py --emit-bash --group {group}` - edit scripts/portfolio.py, not this file.")
    print("#")
    print("# Submits every job of the group in priority order, with one assemble job per unit that runs after its GPU and CPU")
    print("# jobs (--dependency=afterany, so a partial unit still assembles). All jobs of an experiment share ONE run directory:")
    print("# they compute cells (--compute-only) and leave a provenance record in <run_dir>/shards/; the assemble job writes")
    print("# tidy.csv, tables and figures. The agent never submits: you run this ONCE on the login node. Jobs beyond the")
    print("# per-user caps (2 GPUs on `main`, 8 CPUs on `main-cpu`) wait in the queue. Watch with `squeue --me`.")
    print("set -euo pipefail")
    print('cd "${SLURM_SUBMIT_DIR:-$PWD}"')
    print("")
    print("# submit_unit <name> <experiment> <script> <config> <env> <gpu-models|-> <cpu-models|-> <lanes> <mem|-> [overrides...]")
    print("submit_unit() {")
    print('  local name="$1" exp="$2" script="$3" cfg="$4" env="$5" gpu="$6" cpu="$7" lanes="$8" mem="$9" deps="" id')
    print("  shift 9")
    print('  local extra=("$@") memflag=()')
    print('  if [ "$mem" != "-" ]; then memflag=(--mem="$mem"); fi')
    print('  if [ "$gpu" != "-" ]; then')
    print('    id=$(CONDA_ENV="$env" LANES="$lanes" sbatch --parsable ${memflag[@]+"${memflag[@]}"} "$script" "$cfg" "models=[$gpu]" ${extra[@]+"${extra[@]}"})')
    print('    deps="$deps:${id%%;*}"')
    print("  fi")
    print('  if [ "$cpu" != "-" ]; then')
    print('    id=$(CONDA_ENV="$env" sbatch --parsable scripts/run_cpu.sh "$exp" "$cfg" "models=[$cpu]" ${extra[@]+"${extra[@]}"})')
    print('    deps="$deps:${id%%;*}"')
    print("  fi")
    print('  id=$(CONDA_ENV="$env" sbatch --parsable --dependency="afterany$deps" scripts/run_assemble.sh "$exp" "$cfg" ${extra[@]+"${extra[@]}"})')
    print('  echo "submitted: $name  (assemble job ${id%%;*} runs after$deps)"')
    print("}")
    print("")
    print("# submit_single <name> <experiment> <config> <env> <script> [overrides...]   (one process, writes its own outputs)")
    print("submit_single() {")
    print('  local name="$1" exp="$2" cfg="$3" env="$4" script="$5" id')
    print("  shift 5")
    print('  id=$(CONDA_ENV="$env" sbatch --parsable "$script" "$exp" "$cfg" "$@")')
    print('  echo "submitted: $name  (job ${id%%;*})"')
    print("}")
    print("")
    for unit in _selected(group):
        if unit.experiment in SINGLE_PROCESS:
            extra = " ".join(f'"{o}"' for o in unit.overrides)
            print(f'submit_single "{unit.name}" {unit.experiment} {unit.config} {unit.env} '
                  f'{_script(unit.experiment, unit.cpu_only)} {extra}'.rstrip())
            continue
        gm = ",".join(unit.gpu_models) or "-"
        cm = ",".join(unit.cpu_models) or "-"
        # --mem and the lane cap are DERIVED from the models' specs (#8 rule 5): a unit states them only to
        # ask for *less* than the spec allows (a calibration job on one lane), never to repeat a number the
        # spec already holds. Before 2026-09-30 nine units ran PFN backends at the job scripts' default 10 GB
        # although their specs imply 12-16 GB -- the failure mode that OOM-ed TabFM at 10G x 4 lanes.
        _env, derived_mem, derived_lanes = _resources(unit.gpu_models + unit.cpu_models)
        mem = unit.mem or derived_mem or "-"
        lanes = min(unit.lanes, derived_lanes)
        extra = " ".join(f'"{o}"' for o in unit.overrides)
        print(f'submit_unit "{unit.name}" {unit.experiment} {_script(unit.experiment)} {unit.config} {unit.env} "{gm}" "{cm}" '
              f'{lanes} "{mem}" {extra}'.rstrip())
    print("")
    print('echo "Done. Check with: squeue --me"')


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", choices=["stress", "bench", "hypc", "externals", "spinal", "nhp", "audit", "all"],
                        default="all")
    parser.add_argument("--emit-bash", action="store_true", help="Write a submission script to stdout.")
    args = parser.parse_args()
    if args.emit_bash:
        if args.group == "all":
            parser.error("--emit-bash needs one --group (stress, bench, hypc, externals, spinal, nhp or audit).")
        emit_bash(args.group)
    else:
        print_plan(args.group)


if __name__ == "__main__":
    main()
