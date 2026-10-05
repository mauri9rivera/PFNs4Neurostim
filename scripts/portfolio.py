"""The deployment portfolio: every Mila job still to run, in priority order, with wall-time estimates.

The agent never submits jobs. This script PRINTS the plan, or with ``--emit-bash`` writes a one-command
submission script that the USER runs on the login node:

    python scripts/portfolio.py                                                    # the plan (all groups)
    python scripts/portfolio.py --emit-bash --group stress > scripts/submit_portfolio.sh     # Hyp B (restructured knobs, synthetic)
    python scripts/portfolio.py --emit-bash --group bench > scripts/submit_bench.sh          # Hyp 0/A leftovers
    python scripts/portfolio.py --emit-bash --group hypc > scripts/submit_hypc.sh            # Hyp C mechanism analyses
    python scripts/portfolio.py --emit-bash --group externals > scripts/submit_externals.sh  # TabFM, PFNs4BO
    python scripts/portfolio.py --emit-bash --group spinal > scripts/submit_spinal.sh        # every spinal deliverable
    python scripts/portfolio.py --emit-bash --group nhp > scripts/submit_nhp.sh              # NHP re-run under the canonical y scaling
    python scripts/portfolio.py --emit-bash --group audit > scripts/submit_audit.sh          # y-scaling sensitivity arms
    python scripts/portfolio.py --emit-split mila > scripts/submit_mila_gpu.sh     # GPU halves for Mila (you run it)
    python scripts/portfolio.py --emit-split narval > scripts/submit_narval.sh     # CPU halves + overflow GPU (narval.sh do)

``--emit-split`` follows ``SPLIT_PLAN`` (2026-10-05): each unit's GPU half and CPU half go to different clusters, cells meet
in the local cache and are assembled locally with ``--only-cached``. Both scripts take a wave argument (``probe``, ``bulk``,
``synthetic``), so every job class is reviewed on its probe before its siblings are submitted.

Groups (updated 2026-09-25; units whose results already exist were removed — see task_plan.md "Your Mila portfolio"):
    stress     the 5d_rat stress sweeps on the noOutliers cohort (K2 channel/global, K5, K6 failure, synthetic K2).
    bench      Hyp A and the Hyp 0 acquisition tables on 5d_rat.
    hypc       the three NHP mechanism analyses plus the 5d_rat placement arm, re-run with the context-size
               sweeps of task #18 (2026-09-30). C1-C3 last ran locally on 2026-09-24/25 at a SINGLE context
               size each. Each is ONE process (``scripts/run_single.sh``) with no cell cache, so a unit that
               hits the 12 h limit loses its work: check the estimates before submitting, and split a unit
               by config rather than hoping.
    externals  the PFN benchmark: base models on 5d_rat (bench env), TabFM (bench env, 24 GB, <= 2 lanes, NHP rerun plus a
               5d_rat calibration job) and PFNs4BO 5d_rat (main env).
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
#: `gp_mll` re-measured 2026-09-30 after P0.10 redefined it (converged multi-start L-BFGS instead of 100
#: Adam steps): the GP arms run on the CPU lanes, so this moves the CPU column of every unit.
REP_SECONDS: dict[str, dict[str, float]] = {
    "nhp": {"tabpfn_v2_5": 15.7, "gp_mll": 51.5, "gp_naive": 0.8, "random": 0.2, "tabicl": 30.0, "tabfm": 99.0,
            "pfns4bo": 5.3,
            # One cell each, one lane, Narval a100_1g.5gb (2026-10-04, tags a2-check-*): a single measurement on
            # different hardware than the Mila figures above, so treat it as +-30 %. 5d_rat / spinal: unmeasured.
            "tabpfn_v3_5": 31.4, "causilo": 26.2},
    "5d_rat": {"tabpfn_v2_5": 17.6, "gp_mll": 62.9, "gp_naive": 1.0, "random": 0.4, "tabicl": 33.0, "pfns4bo": 6.7},
    # Spinal (budget 64 on the 8x8 grid) is NOT measured on Mila: the NHP costs above scaled by a local
    # one-rep spinal/NHP timing (2026-09-25: TabPFN-2.5 x0.52, GP-MLL x0.60; the GP factor for the rest).
    "spinal": {"tabpfn_v2_5": 8.2, "gp_mll": 34.6, "gp_naive": 0.5, "random": 0.1, "tabicl": 16.5, "pfns4bo": 3.2},
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
DEFAULT_JOB_MEM_GB = 7
BENCH_ENV = "pfns4neurostim-bench"
LATEST_ENV = "pfns4neurostim-latest"
MAIN_ENV = "pfns4neurostim"
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
        parts: Disjoint compute-only overrides (e.g. ``dataset.subjects=[0]``) that split the GPU half into one job per
            part. The assemble job never receives a part, so it always assembles the full grid (a subset override that
            reached the assemble job is what overwrote the 5d_rat PFN-bench tidy.csv on 2026-09-25).
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
    parts: tuple[str, ...] = ()


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
    _u("S1b. K2-channel, 5d_rat", "stress_sweep", "stress_k2_channel_draws_5d_rat", "5d_rat", _TP, _GP, 9),
    _u("S2b. K2-global, 5d_rat", "stress_sweep", "stress_k2_global_5d_rat", "5d_rat", _TP, _GP, 8),
    _u("S3b. K5 slot-fraction heavy tail, 5d_rat", "stress_sweep", "stress_k5_5d_rat", "5d_rat", _TP, _GP, 6),
    _u("S4b. K6 electrode failure, 5d_rat", "stress_sweep", "stress_k6_failure_5d_rat", "5d_rat", _TP, _GP, 5),
    _u("S5b. synthetic K2-channel, 5d_rat twins", "stress_sweep", "stress_k2_channel_synthetic_5d_rat", "5d_rat", _TP, _GP, 9,
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
    _u("E9. TabPFN-3.5 + Causilo, NHP", "bo_benchmark", "hyp0_pfn_bench_nhp", "nhp", ("tabpfn_v3_5", "causilo"), (),
       group="externals", env=LATEST_ENV,
       note="latest env; one cell each validated 2026-10-04 on Narval (a100_1g.5gb: 31 s / 26 s per cell, "
            "3.8 / 1.8 GB RSS); 5d_rat and spinal units wait for their own measurement"),
    # ---- spinal: every deliverable (2026-09-25). Stage data/spinal on the cluster first (runbook). Budget 64 =
    # the 8x8 grid. Subject 5 has one trial per site on every EMG (no noise floor), so the SNR-based stress
    # sweeps run on the other 10 subjects (90 channels); the benchmarks keep all 100.
    # Units longer than 12 h requeue themselves at the time limit and resume from the cell cache.
    _u("P1. Hyp A (TabPFN vs GP), spinal", "bo_benchmark", "hyp_a_spinal", "spinal", _TP, _BASE_CPU, 1, group="spinal"),
    _u("P2. Acquisition core (ts/ei/ucb), spinal", "bo_benchmark", "hyp0_acq_core_spinal", "spinal", _TP, _BASE_CPU, 3,
       group="spinal", note="ts_marginal cells are shared with P1 (same identity)"),
    _u("P3. UCB kappa grid, spinal", "bo_benchmark", "hyp0_ucb_kappa_spinal", "spinal", _TP, (), 5, group="spinal"),
    _u("P5. K2-channel, spinal", "stress_sweep", "stress_k2_channel_draws_spinal", "spinal", _TP, _GP, 9, group="spinal",
       channels=90),
    _u("P6. K2-global, spinal", "stress_sweep", "stress_k2_global_spinal", "spinal", _TP, _GP, 8, group="spinal",
       channels=90),
    _u("P7. K5 slot-fraction heavy tail, spinal", "stress_sweep", "stress_k5_spinal", "spinal", _TP, _GP, 6,
       group="spinal", channels=90),
    _u("P8. K6 electrode failure, spinal", "stress_sweep", "stress_k6_failure_spinal", "spinal", _TP, _GP, 5,
       group="spinal", channels=90),
    _u("P9. K6 budget, spinal", "stress_sweep", "stress_k6_budget_spinal", "spinal", _TP, _GP, 2.5, group="spinal",
       channels=90, note="levels 10,20,30,50,64 = ~2.5x one full budget"),
    _u("P10. synthetic K2-channel, spinal twins", "stress_sweep", "stress_k2_channel_synthetic_spinal", "spinal", _TP, _GP, 9,
       group="spinal", channels=90, note="all 90 twins fit without collapse (checked 2026-09-25)"),
    _u("P11. synthetic K1 decoy, spinal twins", "stress_sweep", "stress_k1_decoy_spinal", "spinal", _TP, _GP, 5,
       group="spinal", channels=90, note="separation 3 pitches on an 8x8 grid: channels where it does not fit are skipped"),
    _u("P13. PFN bench base (TabPFN-2.5, TabICL / GP-MLL), spinal", "bo_benchmark", "hyp0_pfn_bench_spinal", "spinal",
       ("tabpfn_v2_5", "tabicl"), ("gp_mll",), group="spinal", env=BENCH_ENV,
       note="bench env; TabPFN-2.5 / GP-MLL cells are shared with P1"),
    _u("P14. PFNs4BO (native policy), spinal", "bo_benchmark", "hyp0_pfn_bench_spinal", "spinal", ("pfns4bo",), (),
       group="spinal", lanes=2, note="main env"),
    _u("P15. TabFM, spinal (fixed wrapper)", "bo_benchmark", "hyp0_pfn_bench_spinal", "spinal", ("tabfm",), (),
       group="spinal", env=BENCH_ENV,
       note="cost UNMEASURED on spinal: run after E3 and read its per-rep time; sigma is constructed (G3)"),
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
       group="hypc", cpu_only=True, hours=6.8,
       note="MEASURED 6.8 h locally 2026-10-01 (was estimated 3.0): formulation C dominates at ~430 s per "
            "channel-level. Its known-shift gate FAILED for MMD (Spearman 0.900 vs > 0.9) and passed for "
            "W2, so the MMD panels are not interpretable from that run. "
            "CPU unit (device: cpu in the config). 6 context sizes instead of 4 and the ladder now reaches "
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
    _u("N1. K2-channel, NHP", "stress_sweep", "stress_k2_channel_draws_nhp", "nhp", _TP, _GP, 9, group="nhp"),
    _u("N2. K2-global, NHP", "stress_sweep", "stress_k2_global_nhp", "nhp", _TP, _GP, 8, group="nhp"),
    _u("N3. K5 slot-fraction heavy tail, NHP", "stress_sweep", "stress_k5_nhp", "nhp", _TP, _GP, 6, group="nhp"),
    _u("N4. K6 electrode failure, NHP", "stress_sweep", "stress_k6_failure_nhp", "nhp", _TP, _GP, 5, group="nhp"),
    _u("N5. K6 budget, NHP", "stress_sweep", "stress_k6_budget_nhp", "nhp", _TP, _GP, 2.5, group="nhp",
       note="levels 10,20,30,50,96 = ~2.5x one full budget"),
    _u("N6. synthetic K2-channel, NHP twins", "stress_sweep", "stress_k2_channel_synthetic_nhp", "nhp", _TP, _GP, 9,
       group="nhp", note="then --replot --bridge on the N1 run directory for S10"),
    _u("N7. synthetic K1 decoy, NHP twins", "stress_sweep", "stress_k1_decoy_nhp", "nhp", _TP, _GP, 5, group="nhp"),
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
       "stress_k2_channel_draws_nhp", "nhp", (), ("gp_mll", "gp_naive"), 9, group="audit",
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

# ---------------------------------------------------------------------------------------------------------------------
# Split deployment (2026-10-05): GPU halves on Mila, CPU halves on Narval, GPU overflow on Narval.
# ---------------------------------------------------------------------------------------------------------------------
#: Mila `main` allows 8 CPUs per user, so a second GPU job would only queue: one GPU job runs at a time. Measured GPU
#: utilisation at 4 lanes is ~30 %, so 6 lanes share ONE card (half the GPU-hours of two 4-lane jobs for most of the
#: throughput). 6, not 8: sharding is per channel and 18 / 6 = 3 channels per lane, exactly the busiest lane of 8 lanes.
MILA_GPU_LANES = 6
MILA_GPU_MEM = "12G"          # 6 lanes x 1.58 GB (peak per lane, N9 probe 11078548: 9.46 of 10G = 95 %) x 1.25
#: Narval CPU halves: one lane per channel (no per-user cap), ~1.2 GB per GP lane measured on Narval (2026-10-04).
NARVAL_CPU_MEM_PER_LANE_GB = 1.5
WALL_MARGIN = 1.25            # --time = estimate x margin, rounded up to a quarter hour (sharded units requeue anyway)
WALL_ROUND_MIN = 15
WAVES = ("probe", "bulk", "synthetic")


@dataclass(frozen=True)
class Placement:
    """One half of one unit, placed on one cluster for the split deployment.

    Attributes:
        unit_id: Unit id in :data:`UNITS` (the name's prefix, e.g. ``N2``).
        half: ``gpu`` or ``cpu``.
        cluster: ``mila`` (printed for the user) or ``narval`` (``narval.sh do sbatch`` lines for the agent).
        models: Only the models still missing for this unit (2026-10-05 cache scan); cached cells are skipped anyway.
        wave: ``probe`` (first job of a class), ``bulk``, ``synthetic`` (after the cross-cluster twin check), or ``done``
            (already ran; kept as the measured record, never emitted).
        hours: Wall-hour estimate at measured rates; ``--time`` adds :data:`WALL_MARGIN`.
        lanes: Lanes (= CPUs) of the job.
        mem: ``--mem`` of the job.
        gpu_type: Narval GPU slice (``a100``, ``a100_2g.10gb`` ...); empty for CPU or Mila jobs.
        parts: Disjoint overrides, one job each (see :attr:`Unit.parts`).
        overrides: Extra overrides for every job of this half (e.g. a calibration subset and its own tag).
        config: Experiment YAML when it differs from the unit's (the 5d_rat twin of an NHP unit).
    """

    unit_id: str
    half: str
    cluster: str
    models: tuple[str, ...]
    wave: str
    hours: float
    lanes: int
    mem: str
    gpu_type: str = ""
    parts: tuple[str, ...] = ()
    overrides: tuple[str, ...] = ()
    config: str = ""


def _cpu_mem(lanes: int) -> str:
    """``--mem`` for a Narval CPU half: lanes x measured per-lane footprint, rounded up."""
    return f"{int(lanes * NARVAL_CPU_MEM_PER_LANE_GB + 0.999)}G"


# Rates: NHP TabPFN-2.5 ~1180 cells/GPU-h at 6 lanes (N9 probe 11078548: 720 cells in 0.61 h, GPU util 63 %, against
# 492-759 at 4 lanes); GP-MLL ~60 s per rep on a Narval core (N13 probe 4692312: 18 lanes x 360 cells in 2.0 h, grade A),
# one channel per CPU lane. Synthetic units wait for the twin check. Wave `done` = already ran, kept as the record.
_M, _NC, _RC = MILA_GPU_LANES, CHANNELS["nhp"], CHANNELS["5d_rat"]
_NM, _RM = _cpu_mem(CHANNELS["nhp"]), _cpu_mem(CHANNELS["5d_rat"])
SPLIT_PLAN: tuple[Placement, ...] = (
    # ---- probes: one per job class ----
    Placement("N9", "gpu", "mila", ("tabpfn_v2_5",), "done", 0.61, _M, MILA_GPU_MEM),
    Placement("N13", "cpu", "narval", ("gp_mll", "gp_naive", "random"), "done", 2.0, _NC, _NM),
    # TabFM stays on Mila (2026-10-05): its bench env cannot be built on Narval (numpy==2.2.6 is not in the Alliance
    # wheelhouse). 1 lane / 24G fits beside the N9 probe within Mila's caps (7 of 8 CPUs, 34 of 48 GB, 2 of 2 GPUs).
    # The full TabFM 5d_rat run is placed once this calibration has measured it.
    Placement("E4", "gpu", "mila", ("tabfm",), "probe", 0.8, 1, "24G"),
    # ---- bulk: Mila GPU, TabPFN / TabICL halves (one runs at a time) ----
    Placement("N2", "gpu", "mila", ("tabpfn_v2_5",), "bulk", 2.5, _M, MILA_GPU_MEM),
    Placement("N3", "gpu", "mila", ("tabpfn_v2_5",), "bulk", 1.9, _M, MILA_GPU_MEM),
    Placement("N4", "gpu", "mila", ("tabpfn_v2_5",), "bulk", 1.6, _M, MILA_GPU_MEM),
    Placement("N5", "gpu", "mila", ("tabpfn_v2_5",), "bulk", 0.8, _M, MILA_GPU_MEM),
    Placement("N11", "gpu", "mila", ("tabicl",), "bulk", 1.6, _M, "12G"),
    Placement("E1", "gpu", "mila", ("tabicl",), "bulk", 1.6, _M, "12G"),
    # ---- bulk: Narval CPU, GP halves (one lane per channel) ----
    Placement("N2", "cpu", "narval", ("gp_mll", "gp_naive"), "bulk", 2.3, _NC, _NM),
    Placement("N3", "cpu", "narval", ("gp_mll", "gp_naive"), "bulk", 1.8, _NC, _NM),
    Placement("N4", "cpu", "narval", ("gp_mll", "gp_naive"), "bulk", 1.5, _NC, _NM),
    Placement("N5", "cpu", "narval", ("gp_mll", "gp_naive"), "bulk", 0.7, _NC, _NM),
    Placement("N9", "cpu", "narval", ("gp_mll",), "bulk", 0.9, _NC, _NM),   # + ts_marginal: those cells sit on Mila only
    Placement("S2b", "cpu", "narval", ("gp_naive",), "bulk", 0.1, _RC, _RM),
    Placement("S3b", "cpu", "narval", ("gp_mll", "gp_naive"), "bulk", 2.2, _RC, _RM),
    Placement("S4b", "cpu", "narval", ("gp_mll", "gp_naive"), "bulk", 1.8, _RC, _RM),
    # ---- bulk: Narval GPU overflow (latest env built and measured there; its one-cell checks live there) ----
    Placement("E9", "gpu", "narval", ("tabpfn_v3_5", "causilo"), "bulk", 2.0, 4, "20G", gpu_type="a100_2g.10gb"),
    Placement("E9", "gpu", "narval", ("tabpfn_v3_5", "causilo"), "bulk", 2.0, 4, "20G", gpu_type="a100_2g.10gb",
              config="configs/experiment/hyp0_pfn_bench_5d_rat.yaml"),
    # ---- synthetic: only after the cross-cluster twin check (A3 Step 15) ----
    Placement("N7", "gpu", "mila", ("tabpfn_v2_5",), "synthetic", 1.6, _M, MILA_GPU_MEM),
    Placement("N6", "gpu", "mila", ("tabpfn_v2_5",), "synthetic", 2.8, _M, MILA_GPU_MEM),
    Placement("N7", "cpu", "narval", ("gp_mll", "gp_naive"), "synthetic", 1.5, _NC, _NM),
    Placement("N6", "cpu", "narval", ("gp_mll",), "synthetic", 2.6, _NC, _NM),
    Placement("S5b", "cpu", "narval", ("gp_mll", "gp_naive"), "synthetic", 3.2, _RC, _RM),
    Placement("S5b", "gpu", "narval", ("tabpfn_v2_5",), "synthetic", 6.5, 4, "8G", gpu_type="a100_2g.10gb",
              parts=tuple(f"dataset.subjects=[{s}]" for s in range(5))),
)


def _unit(unit_id: str) -> Unit:
    """Return the unit whose name starts with ``<unit_id>.``.

    Args:
        unit_id: Unit id, e.g. ``N2``.

    Returns:
        The matching :class:`Unit`.

    Raises:
        KeyError: If no unit has that id.
    """
    for unit in UNITS:
        if unit.name.split(".", 1)[0] == unit_id:
            return unit
    raise KeyError(f"no unit {unit_id!r} in UNITS")


def _walltime(hours: float) -> str:
    """Return ``HH:MM:SS`` for ``hours`` x :data:`WALL_MARGIN`, rounded up to :data:`WALL_ROUND_MIN` minutes."""
    minutes = int(-(-hours * WALL_MARGIN * 60 // WALL_ROUND_MIN) * WALL_ROUND_MIN)
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


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
    print("# One stage at a time, each individually gated: bash scripts/submit_next.sh status | <stage>")
    print("# There is deliberately NO submit-everything mode; `spinal` additionally needs --force.")


def emit_bash(group: str) -> None:
    """Write the one-command submission script for ``group`` to stdout."""
    print("#!/bin/bash")
    print(f"# GENERATED by `python scripts/portfolio.py --emit-bash --group {group}` - edit scripts/portfolio.py, not this file.")
    print("#")
    print("# Submits every job of the group in priority order, with one assemble job per unit that runs after its GPU and CPU")
    print("# jobs (--dependency=afterany, so a partial unit still assembles). All jobs of an experiment share ONE run directory:")
    print("# they compute cells (--compute-only) and leave a provenance record in <run_dir>/shards/; the assemble job writes")
    print("# tidy.csv, tables and figures. The agent never submits: you run this ONCE on the login node. Jobs beyond the")
    print("# per-user caps (on Mila: 2 GPUs on `main`, 8 CPUs on `main-cpu`) wait in the queue. Watch with `squeue --me`.")
    print("# Runs unchanged on Mila and Narval: the cluster-specific sbatch flags come from scripts/cluster.sh.")
    print("set -euo pipefail")
    print('cd "${SLURM_SUBMIT_DIR:-$PWD}"')
    print('read -r -a GPU_FLAGS <<< "$(bash scripts/cluster.sh flags gpu)"')
    print('read -r -a CPU_FLAGS <<< "$(bash scripts/cluster.sh flags cpu)"')
    print("")
    print("# submit_unit <name> <experiment> <script> <config> <env> <gpu-models|-> <cpu-models|-> <lanes> <mem|-> <parts|-> [overrides...]")
    print("#   parts: '|'-separated compute-only overrides, one GPU job each; the assemble job never sees a part (full grid).")
    print("submit_unit() {")
    print('  local name="$1" exp="$2" script="$3" cfg="$4" env="$5" gpu="$6" cpu="$7" lanes="$8" mem="$9" parts="${10}" deps="" id part')
    print("  shift 10")
    print('  local extra=("$@") memflag=() plist=()')
    print('  if [ "$mem" != "-" ]; then memflag=(--mem="$mem"); fi')
    print('  if [ "$parts" = "-" ]; then plist=(""); else IFS="|" read -r -a plist <<< "$parts"; fi')
    print('  if [ "$gpu" != "-" ]; then')
    print('    for part in "${plist[@]}"; do')
    print('      id=$(CONDA_ENV="$env" LANES="$lanes" sbatch --parsable ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} ${memflag[@]+"${memflag[@]}"} "$script" "$cfg" "models=[$gpu]" ${part:+"$part"} ${extra[@]+"${extra[@]}"})')
    print('      deps="$deps:${id%%;*}"')
    print("    done")
    print("  fi")
    print('  if [ "$cpu" != "-" ]; then')
    print('    id=$(CONDA_ENV="$env" sbatch --parsable ${CPU_FLAGS[@]+"${CPU_FLAGS[@]}"} scripts/run_cpu.sh "$exp" "$cfg" "models=[$cpu]" ${extra[@]+"${extra[@]}"})')
    print('    deps="$deps:${id%%;*}"')
    print("  fi")
    print('  id=$(CONDA_ENV="$env" sbatch --parsable ${CPU_FLAGS[@]+"${CPU_FLAGS[@]}"} --dependency="afterany$deps" scripts/run_assemble.sh "$exp" "$cfg" ${extra[@]+"${extra[@]}"})')
    print('  echo "submitted: $name  (assemble job ${id%%;*} runs after$deps)"')
    print("}")
    print("")
    print("# submit_single <name> <experiment> <config> <env> <script> [overrides...]   (one process, writes its own outputs)")
    print("submit_single() {")
    print('  local name="$1" exp="$2" cfg="$3" env="$4" script="$5" id flags=(${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"})')
    print("  shift 5")
    print('  case "$script" in *_cpu.sh) flags=(${CPU_FLAGS[@]+"${CPU_FLAGS[@]}"}) ;; esac')
    print('  id=$(CONDA_ENV="$env" sbatch --parsable ${flags[@]+"${flags[@]}"} "$script" "$exp" "$cfg" "$@")')
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
        parts = "|".join(unit.parts) or "-"
        print(f'submit_unit "{unit.name}" {unit.experiment} {_script(unit.experiment)} {unit.config} {unit.env} "{gm}" "{cm}" '
              f'{lanes} "{mem}" "{parts}" {extra}'.rstrip())
    print("")
    print('echo "Done. Check with: squeue --me"')


def _split_lines(placement: Placement) -> list[str]:
    """Submission lines of one placement (one per part).

    Args:
        placement: The half to submit.

    Returns:
        Shell lines: ``sbatch`` for Mila (the user runs them), ``narval.sh do sbatch`` for Narval (the agent runs them).
    """
    unit = _unit(placement.unit_id)
    cfg = placement.config or unit.config
    env, _mem, _lanes = _resources(placement.models)
    name = f"{placement.unit_id}-{placement.half}"
    if placement.config:
        name += "-" + placement.config.rsplit("_", 1)[-1].removesuffix(".yaml")
    script = "scripts/run_cpu.sh" if placement.half == "cpu" else _script(unit.experiment)
    runner = (unit.experiment,) if placement.half == "cpu" else ()
    models = f'"models=[{",".join(placement.models)}]"'
    extra = " ".join(f'"{o}"' for o in unit.overrides + placement.overrides)
    time = _walltime(placement.hours)
    lines = []
    for i, part in enumerate(placement.parts or ("",)):
        job = f"{name}-p{i}" if placement.parts else name
        args = " ".join([*runner, cfg, models] + ([f'"{part}"'] if part else []) + ([extra] if extra else []))
        if placement.cluster == "mila":
            flags = "${CPU_FLAGS[@]}" if placement.half == "cpu" else "${GPU_FLAGS[@]}"
            lines.append(f'CONDA_ENV={env} LANES={placement.lanes} sbatch --parsable {flags} --cpus-per-task={placement.lanes} '
                         f'--mem={placement.mem} --time={time} --job-name={job} {script} {args}')
        else:
            gpu = f" --gpu-type {placement.gpu_type}" if placement.gpu_type else ""
            lanes = "" if placement.half == "cpu" else f" --lanes {placement.lanes}"
            lines.append(f"bash scripts/narval.sh do sbatch {placement.half}{lanes} --conda-env {env}{gpu} --cpus {placement.lanes} "
                         f"--mem {placement.mem} --time {time} --job-name {job} -- {script} {args}")
    return lines


def emit_split(cluster: str) -> None:
    """Write the wave-gated submission script of one cluster's halves of :data:`SPLIT_PLAN` to stdout.

    Args:
        cluster: ``mila`` or ``narval``.
    """
    mine = [p for p in SPLIT_PLAN if p.cluster == cluster]
    print("#!/bin/bash")
    print(f"# GENERATED by `python scripts/portfolio.py --emit-split {cluster}` - edit SPLIT_PLAN in scripts/portfolio.py, not this file.")
    print("#")
    print("# Split deployment (2026-10-05): GPU halves on Mila, CPU halves and GPU overflow on Narval. These jobs only COMPUTE")
    print("# cells; there is no assemble job here -- the halves meet in the LOCAL cache and are assembled locally with")
    print("# --only-cached. Run one wave at a time and review its probe first:   bash <this script> probe|bulk|synthetic")
    if cluster == "mila":
        print("# YOU run this on the Mila login node (the agent never submits on Mila). `main` allows 8 CPUs per user, so with")
        print(f"# {MILA_GPU_LANES}-lane jobs exactly one GPU job runs at a time and the rest wait (QOSMaxCpuPerUserLimit) -- intended.")
    else:
        print("# The agent runs these lines one by one after your approval (each is a `narval.sh do sbatch` call, audited in")
        print("# output/logs/narval_audit.log). NARVAL_DRY_RUN=1 prints the remote commands without submitting.")
    print("set -euo pipefail")
    if cluster == "mila":
        print('cd "${SLURM_SUBMIT_DIR:-$PWD}"')
        print('read -r -a GPU_FLAGS <<< "$(bash scripts/cluster.sh flags gpu)"')
        print('read -r -a CPU_FLAGS <<< "$(bash scripts/cluster.sh flags cpu)"')
    print('wave="${1:?usage: bash $0 probe|bulk|synthetic}"')
    print('case "$wave" in')
    for wave in WAVES:
        print(f"  {wave})")
        rows = [p for p in mine if p.wave == wave]
        if not rows:
            print('    echo "nothing in this wave for this cluster" ;;')
            continue
        for p in rows:
            for line in _split_lines(p):
                print(f"    {line}")
        print("    ;;")
    print('  *) echo "unknown wave: $wave (probe|bulk|synthetic)" >&2; exit 2 ;;')
    print("esac")
    gpu_h = sum(p.hours * max(len(p.parts), 1) for p in mine if p.half == "gpu")
    cpu_h = sum(p.hours * p.lanes for p in mine if p.half == "cpu")
    print(f'echo "estimates for {cluster}: GPU ~{gpu_h:.0f} job-h, CPU ~{cpu_h:.0f} core-h (all waves)"')


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", choices=["stress", "bench", "hypc", "externals", "spinal", "nhp", "audit", "all"],
                        default="all")
    parser.add_argument("--emit-bash", action="store_true", help="Write a submission script to stdout.")
    parser.add_argument("--emit-split", choices=["mila", "narval"], help="Write one cluster's halves of SPLIT_PLAN to stdout.")
    args = parser.parse_args()
    if args.emit_split:
        emit_split(args.emit_split)
    elif args.emit_bash:
        if args.group == "all":
            parser.error("--emit-bash needs one --group (stress, bench, hypc, externals, spinal, nhp or audit).")
        emit_bash(args.group)
    else:
        print_plan(args.group)


if __name__ == "__main__":
    main()
