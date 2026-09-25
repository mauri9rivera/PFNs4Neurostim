"""The deployment portfolio: every Mila job still to run, in priority order, with wall-time estimates.

The agent never submits jobs. This script PRINTS the plan, or with ``--emit-bash`` writes a one-command
submission script that the USER runs on the login node:

    python scripts/portfolio.py                                                    # the plan (all groups)
    python scripts/portfolio.py --emit-bash --group stress > scripts/submit_portfolio.sh     # Hyp B (restructured knobs, Demo 1)
    python scripts/portfolio.py --emit-bash --group bench > scripts/submit_bench.sh          # Hyp 0/A leftovers + GT sensitivity
    python scripts/portfolio.py --emit-bash --group hypc > scripts/submit_hypc.sh            # Hyp C mechanism analyses
    python scripts/portfolio.py --emit-bash --group externals > scripts/submit_externals.sh  # TabFM, PFNs4BO

Groups (updated 2026-09-25; units whose results already exist were removed — see task_plan.md "Your Mila portfolio"):
    stress     the 5d_rat stress sweeps on the noOutliers cohort (K2 channel/global, K5, K6 failure, Demo 1 K2).
    bench      Hyp A and the Hyp 0 acquisition tables on 5d_rat.
    hypc       none left (C1-C3 ran locally 2026-09-24/25).
    externals  the PFN benchmark: base models on 5d_rat (bench env), TabFM (bench env, 24 GB, <= 2 lanes, NHP rerun plus a
               5d_rat calibration job), PFNs4BO 5d_rat (main env) and TabPFN v1 (v1 env, tabpfn<2).

Sharded experiments (``bo_benchmark``, ``stress_sweep``) write every job's cells to the shared cache and one assemble job
builds tidy.csv, tables and figures. ``mechanism`` and ``gt_sensitivity`` run as ONE process (``scripts/run_single.sh``) and
write their own outputs. Estimates use per-repetition costs measured on 2026-09-20/23.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

#: Seconds per repetition at the deliverable budget (96 NHP / 100 5d_rat), per model. Absent = unmeasured.
REP_SECONDS: dict[str, dict[str, float]] = {
    "nhp": {"tabpfn_v2_5": 15.7, "gp_mll": 19.5, "gp_naive": 0.8, "random": 0.2, "tabicl": 30.0, "tabfm": 99.0,
            "pfns4bo": 5.3},
    "5d_rat": {"tabpfn_v2_5": 17.6, "gp_mll": 19.5, "gp_naive": 1.0, "random": 0.4, "tabicl": 33.0, "pfns4bo": 6.7},
}
GUESSED: frozenset[str] = frozenset()   # every listed cost is measured (TabICL from the D3 runs: 0.43-0.52 s/step)
#: Channels (subject x EMG) per dataset. 5d_rat dropped from 18 to 17 with the
#: noOutliers cohort (2026-09-25): rData03 left and flag_valid_emg replaced the
#: hand-maintained EMG lists.
CHANNELS = {"nhp": 18, "5d_rat": 17}
REPS = 10
GPU_LANE_GAIN = 1.7      # aggregate speed-up of several lanes on one GPU (measured with 3 lanes)
GPU_LANES = 4
CPU_LANES = 8
BENCH_ENV = "pfns4neurostim-bench"
MAIN_ENV = "pfns4neurostim"
V1_ENV = "pfns4neurostim-v1"    # tabpfn<2; cannot share an interpreter with the pinned 6.3.2
#: Runners that run as one process and write their own deliverables (no lanes, no cell assembly).
SINGLE_PROCESS: frozenset[str] = frozenset({"mechanism", "gt_sensitivity"})


@dataclass(frozen=True)
class Unit:
    """One deliverable x dataset job group.

    Attributes:
        name: Label.
        group: ``stress``, ``bench``, ``hypc`` or ``externals``.
        experiment: ``bo_benchmark``, ``stress_sweep``, ``mechanism`` or ``gt_sensitivity``.
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
       env=BENCH_ENV, lanes=2, mem="24G",
       note="rerun: raw-scale mean + spread/out-of-fold sigma, batched members (2026-09-25; ~1.3 s/step locally); "
            "24 GB, <=2 lanes; sigma is constructed (G3)"),
    # A subset run (calibration, smoke) ALWAYS gets its own tag: a unit shares its run directory with every
    # other unit of the same config and tag, and its assemble job would overwrite the full run's tidy.csv with
    # the subset (happened to hyp0-pfn-bench-5d_rat on 2026-09-24).
    _u("E4. TabFM 5d_rat CALIBRATION (1 channel, 2 reps)", "bo_benchmark", "hyp0_pfn_bench_5d_rat", "5d_rat", ("tabfm",), (),
       group="externals", env=BENCH_ENV, lanes=1, mem="24G",
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
)


def _script(experiment: str) -> str:
    if experiment in SINGLE_PROCESS:
        return "scripts/run_single.sh"
    return "scripts/run_bo_benchmark.sh" if experiment == "bo_benchmark" else "scripts/run_stress_sweep.sh"


def _hours(unit: Unit, models: tuple[str, ...], speedup: float) -> float | None:
    """Estimated wall hours, or ``None`` if any model's per-rep cost is unmeasured."""
    if unit.experiment in SINGLE_PROCESS:
        return unit.hours if models is unit.gpu_models else 0.0
    if not models:
        return 0.0
    costs = [REP_SECONDS[unit.dataset].get(m) for m in models]
    if any(c is None for c in costs):
        return None
    return sum(costs) * unit.multiplier * CHANNELS[unit.dataset] * REPS / 3600.0 / speedup


def _fmt(hours: float | None) -> str:
    return "?" if hours is None else f"{hours:.1f}"


def _selected(group: str) -> list[Unit]:
    return [u for u in UNITS if group == "all" or u.group == group]


def print_plan(group: str) -> None:
    """Print the human-readable plan."""
    for unit in _selected(group):
        gpu_h = _hours(unit, unit.gpu_models, GPU_LANE_GAIN if unit.lanes > 1 else 1.0)
        cpu_h = _hours(unit, unit.cpu_models, CPU_LANES)
        print(f"\n### {unit.name}   [{unit.group}, env {unit.env}]   est. wall: GPU ~{_fmt(gpu_h)} h, CPU ~{_fmt(cpu_h)} h")
        if unit.note:
            print(f"# {unit.note}")
    print("\n# Mila caps: 2 GPUs on `main` (extra GPU jobs queue), 8 CPUs on `main-cpu`. Watch: bash scripts/mila.sh queue")
    print("# Submit: bash scripts/submit_portfolio.sh (stress) | submit_bench.sh | submit_hypc.sh | submit_externals.sh")


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
    print("# submit_single <name> <experiment> <config> <env> [overrides...]   (one process, writes its own outputs)")
    print("submit_single() {")
    print('  local name="$1" exp="$2" cfg="$3" env="$4" id')
    print("  shift 4")
    print('  id=$(CONDA_ENV="$env" sbatch --parsable scripts/run_single.sh "$exp" "$cfg" "$@")')
    print('  echo "submitted: $name  (job ${id%%;*})"')
    print("}")
    print("")
    for unit in _selected(group):
        if unit.experiment in SINGLE_PROCESS:
            extra = " ".join(f'"{o}"' for o in unit.overrides)
            print(f'submit_single "{unit.name}" {unit.experiment} {unit.config} {unit.env} {extra}'.rstrip())
            continue
        gm = ",".join(unit.gpu_models) or "-"
        cm = ",".join(unit.cpu_models) or "-"
        mem = unit.mem or "-"
        extra = " ".join(f'"{o}"' for o in unit.overrides)
        print(f'submit_unit "{unit.name}" {unit.experiment} {_script(unit.experiment)} {unit.config} {unit.env} "{gm}" "{cm}" '
              f'{unit.lanes} "{mem}" {extra}'.rstrip())
    print("")
    print('echo "Done. Check with: squeue --me"')


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", choices=["stress", "bench", "hypc", "externals", "all"], default="all")
    parser.add_argument("--emit-bash", action="store_true", help="Write a submission script to stdout.")
    args = parser.parse_args()
    if args.emit_bash:
        if args.group == "all":
            parser.error("--emit-bash needs one --group (stress, bench, hypc or externals).")
        emit_bash(args.group)
    else:
        print_plan(args.group)


if __name__ == "__main__":
    main()
