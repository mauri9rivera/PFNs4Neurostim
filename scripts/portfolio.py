"""The deployment portfolio: every job in priority order, with wall-time estimates.

The agent never submits jobs. This script PRINTS the plan, or with ``--emit-bash`` writes the
one-command submission script (``scripts/submit_portfolio.sh``) that the USER runs on the login node.

Estimates use per-repetition costs measured on 2026-09-20 (see the sprint file): TabPFN 15.7 s (NHP) /
17.6 s (5d_rat) at budget 96/100, GP-MLL 19.5 s on one CPU thread, GP-fixed ~1 s, Random ~0.3 s; a GPU job
with several lanes gains ~1.7x aggregate throughput. TabICL and TabFM costs are UNMEASURED.

    python scripts/portfolio.py                       # everything
    python scripts/portfolio.py --machine mila        # only the cluster jobs
    python scripts/portfolio.py --emit-bash > scripts/submit_portfolio.sh
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

#: Seconds per repetition at the deliverable budget (96 NHP / 100 5d_rat), per model. Absent = unmeasured.
REP_SECONDS: dict[str, dict[str, float]] = {
    "nhp": {"tabpfn_v2_5": 15.7, "gp_mll": 19.5, "gp_naive": 0.8, "random": 0.2, "tabicl": 16.0},
    "5d_rat": {"tabpfn_v2_5": 17.6, "gp_mll": 19.5, "gp_naive": 1.0, "random": 0.4, "tabicl": 18.0},
}
GUESSED: frozenset[str] = frozenset({"tabicl"})   # assumed equal to TabPFN until calibrated
CHANNELS = 18
REPS = 10
GPU_LANE_GAIN = 1.7      # aggregate speed-up of several lanes on one GPU (measured with 3 lanes)
GPU_LANES = 4
CPU_LANES = 8
BENCH_ENV = "pfns4neurostim-bench"


@dataclass(frozen=True)
class Unit:
    """One deliverable x dataset job group.

    Attributes:
        name: Label.
        experiment: ``bo_benchmark`` or ``stress_sweep``.
        config: Experiment YAML.
        dataset: ``nhp`` or ``5d_rat`` (keys of :data:`REP_SECONDS`).
        gpu_models: Models needing a GPU (may be empty).
        cpu_models: Models run on the CPU partition (may be empty).
        multiplier: Cells per repetition per model (knob levels, or a budget-scaling factor).
        machine: ``mila`` or ``local``.
        env: Conda env for the cluster jobs.
        note: Free-text remark.
    """

    name: str
    experiment: str
    config: str
    dataset: str
    gpu_models: tuple[str, ...]
    cpu_models: tuple[str, ...]
    multiplier: float
    machine: str
    env: str = "pfns4neurostim"
    note: str = ""


UNITS: tuple[Unit, ...] = (
    Unit("1. D1 5d_rat", "bo_benchmark", "configs/experiment/hyp_a_5d_rat.yaml", "5d_rat",
         ("tabpfn_v2_5",), ("gp_mll", "gp_naive", "random"), 1, "mila"),
    Unit("2a. K2 noise amplification NHP", "stress_sweep", "configs/experiment/stress_k2_nhp.yaml", "nhp",
         ("tabpfn_v2_5",), ("gp_mll", "gp_naive"), 8, "mila"),
    Unit("2b. K2 noise amplification 5d_rat", "stress_sweep", "configs/experiment/stress_k2_5d_rat.yaml", "5d_rat",
         ("tabpfn_v2_5",), ("gp_mll", "gp_naive"), 8, "mila"),
    Unit("3a. D3 PFN benchmark NHP (TabPFN-2.5, TabICL, GP-MLL)", "bo_benchmark",
         "configs/experiment/hyp0_pfn_bench_nhp.yaml", "nhp", ("tabpfn_v2_5", "tabicl"), ("gp_mll",), 1, "mila", BENCH_ENV,
         "bench env (py3.11); TabICL cost assumed = TabPFN until calibrated"),
    Unit("3b. D3 PFN benchmark 5d_rat (TabPFN-2.5, TabICL, GP-MLL)", "bo_benchmark",
         "configs/experiment/hyp0_pfn_bench_5d_rat.yaml", "5d_rat", ("tabpfn_v2_5", "tabicl"), ("gp_mll",), 1, "mila", BENCH_ENV,
         "bench env (py3.11); TabICL cost assumed = TabPFN until calibrated"),
    Unit("4. K5 outliers 5d_rat", "stress_sweep", "configs/experiment/stress_k5_5d_rat.yaml", "5d_rat",
         ("tabpfn_v2_5",), ("gp_mll", "gp_naive"), 5, "mila", note="real lab artefacts; 5 calibrated levels"),
    Unit("5a. D3 TabFM NHP (LAST PFN job)", "bo_benchmark", "configs/experiment/hyp0_pfn_bench_nhp.yaml", "nhp",
         ("tabfm",), (), 1, "mila", BENCH_ENV,
         "COST UNMEASURED: ~340 s per 96-site predict on CPU, so GPU-only; calibrate first (see the printed calibration line)"),
    Unit("5b. D3 TabFM 5d_rat (LAST PFN job)", "bo_benchmark", "configs/experiment/hyp0_pfn_bench_5d_rat.yaml", "5d_rat",
         ("tabfm",), (), 1, "mila", BENCH_ENV, "COST UNMEASURED: GPU-only; calibrate first"),
    Unit("L1. D1 NHP (RUNNING locally)", "bo_benchmark", "configs/experiment/hyp_a_nhp.yaml", "nhp",
         ("tabpfn_v2_5",), ("gp_mll", "gp_naive", "random"), 1, "local"),
    Unit("L2. K6-budget NHP", "stress_sweep", "configs/experiment/stress_k6_budget_nhp.yaml", "nhp",
         ("tabpfn_v2_5",), ("gp_mll", "gp_naive"), 2.15, "local", note="levels 10,20,30,50,96 = 2.15x one full budget"),
)


def _script(experiment: str) -> str:
    return "scripts/run_bo_benchmark.sh" if experiment == "bo_benchmark" else "scripts/run_stress_sweep.sh"


def _hours(unit: Unit, models: tuple[str, ...], speedup: float) -> float | None:
    """Estimated wall hours, or ``None`` if any model's per-rep cost is unmeasured."""
    if not models:
        return 0.0
    costs = [REP_SECONDS[unit.dataset].get(m) for m in models]
    if any(c is None for c in costs):
        return None
    return sum(costs) * unit.multiplier * CHANNELS * REPS / 3600.0 / speedup


def _fmt(hours: float | None) -> str:
    return "?" if hours is None else f"{hours:.1f}"


def print_plan(machine: str) -> None:
    """Print the human-readable plan."""
    for unit in UNITS:
        if machine != "all" and unit.machine != machine:
            continue
        gpu_h = _hours(unit, unit.gpu_models, GPU_LANE_GAIN)
        cpu_h = _hours(unit, unit.cpu_models, CPU_LANES)
        guessed = sorted(set(unit.gpu_models) & GUESSED)
        print(f"\n### {unit.name}   [{unit.machine}]   est. wall: GPU ~{_fmt(gpu_h)} h, CPU ~{_fmt(cpu_h)} h"
              + (f"  (assumed cost: {', '.join(guessed)})" if guessed else ""))
        if unit.note:
            print(f"# {unit.note}")
        gm, cm = ",".join(unit.gpu_models), ",".join(unit.cpu_models)
        env = "" if unit.env == "pfns4neurostim" else f"CONDA_ENV={unit.env} "
        if unit.machine == "mila":
            if gm:
                print(f"{env}LANES={GPU_LANES} sbatch {_script(unit.experiment)} {unit.config} \"models=[{gm}]\" tag=gpu")
            if cm:
                print(f"{env}sbatch scripts/run_cpu.sh {unit.experiment} {unit.config} \"models=[{cm}]\" tag=cpu")
            if "tabfm" in unit.gpu_models:
                print(f"# calibrate first: {env}sbatch scripts/run_bo_benchmark.sh {unit.config} \"models=[tabfm]\" "
                      "dataset.subjects=[1] dataset.emgs=[0] n_reps=2 tag=calib")
        else:
            print(f"python -m pfns4neurostim {unit.experiment} --config {unit.config} --set \"models=[{gm}]\" tag=local-gpu")
            print(f"python -m pfns4neurostim {unit.experiment} --config {unit.config} --set \"models=[{cm}]\" tag=local-cpu")
            print(f"python -m pfns4neurostim {unit.experiment} --config {unit.config} --only-cached")
    print("\n# Mila caps: 2 GPUs on `main` (extra GPU jobs queue), 8 CPUs on `main-cpu`. Watch: bash scripts/mila.sh queue")
    print("# One command submits everything with dependencies: bash scripts/submit_portfolio.sh")


def emit_bash() -> None:
    """Write the one-command submission script to stdout (Mila units only, priority order)."""
    print("#!/bin/bash")
    print("# GENERATED by `python scripts/portfolio.py --emit-bash` - edit scripts/portfolio.py, not this file.")
    print("#")
    print("# Submits every cluster job in priority order, with one auto-assemble job per unit that runs after its GPU")
    print("# and CPU jobs finish (--dependency=afterany, so a partial unit still assembles). Run it ONCE on the login")
    print("# node; jobs then run unattended (requeue on preemption, cached cells resume), so you can go to sleep.")
    print("# The agent never submits: you run this. Jobs beyond the per-user caps (2 GPUs on `main`, 8 CPUs on")
    print("# `main-cpu`) simply wait in the queue. Watch with `squeue --me`.")
    print("set -euo pipefail")
    print('cd "${SLURM_SUBMIT_DIR:-$PWD}"')
    print("")
    print("# submit_unit <name> <experiment> <script> <config> <env> <gpu-models|-> <cpu-models|->")
    print("submit_unit() {")
    print('  local name="$1" exp="$2" script="$3" cfg="$4" env="$5" gpu="$6" cpu="$7" deps="" id')
    print('  if [ "$gpu" != "-" ]; then')
    print(f'    id=$(CONDA_ENV="$env" LANES={GPU_LANES} sbatch --parsable "$script" "$cfg" "models=[$gpu]" tag=gpu)')
    print('    deps="$deps:${id%%;*}"')
    print("  fi")
    print('  if [ "$cpu" != "-" ]; then')
    print('    id=$(CONDA_ENV="$env" sbatch --parsable scripts/run_cpu.sh "$exp" "$cfg" "models=[$cpu]" tag=cpu)')
    print('    deps="$deps:${id%%;*}"')
    print("  fi")
    print('  id=$(CONDA_ENV="$env" sbatch --parsable --dependency="afterany$deps" scripts/run_assemble.sh "$exp" "$cfg")')
    print('  echo "submitted: $name  (assemble job ${id%%;*} runs after$deps)"')
    print("}")
    print("")
    for unit in UNITS:
        if unit.machine != "mila":
            continue
        gm = ",".join(unit.gpu_models) or "-"
        cm = ",".join(unit.cpu_models) or "-"
        print(f'submit_unit "{unit.name}" {unit.experiment} {_script(unit.experiment)} {unit.config} {unit.env} "{gm}" "{cm}"')
    print("")
    print('echo "Done. Check with: squeue --me"')


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--machine", choices=["mila", "local", "all"], default="all")
    parser.add_argument("--emit-bash", action="store_true", help="Write scripts/submit_portfolio.sh to stdout.")
    args = parser.parse_args()
    if args.emit_bash:
        emit_bash()
    else:
        print_plan(args.machine)


if __name__ == "__main__":
    main()
