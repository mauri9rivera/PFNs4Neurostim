#!/bin/bash
# Priority-ordered Mila submission for a tight deadline (2026-09-21). The USER runs this once on the login node; the agent never submits.
#
#   bash scripts/submit_priority.sh
#
# Every unit runs in TWO waves that add up to the full 10 repetitions with no repeated work:
#   wave A: reps 0-4 (n_reps=5)   -> a complete, consistent 5-rep result exists early (insurance against the clock)
#   wave B: reps 0-9 (n_reps=10)  -> reps 0-4 are cache hits (the seed does not depend on n_reps), only reps 5-9 compute
# Wave B of a unit starts only after wave A of the SAME unit has been assembled (--dependency=afterany), so the two never compute
# the same cell twice, and a slot freed by one unit is immediately usable by another. All wave-A units are queued before any wave-B
# unit, in priority order. The per-user cap is 2 GPUs on `main`; the GP jobs run on `main-cpu` (its own 8 CPUs), one at a time.
# Each unit ends with an assemble job (tidy.csv + tables + figures, no compute). Override with WAVE_A_REPS / FINAL_REPS.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
WAVE_A_REPS="${WAVE_A_REPS:-5}"
FINAL_REPS="${FINAL_REPS:-10}"

# Priority order. Format: name|experiment|gpu-script|config|conda env|gpu models|cpu models
UNITS=(
  "1. D1 5d_rat|bo_benchmark|scripts/run_bo_benchmark.sh|configs/experiment/hyp_a_5d_rat.yaml|pfns4neurostim|tabpfn_v2_5|gp_mll,gp_naive,random"
  "2. D3 PFN benchmark NHP|bo_benchmark|scripts/run_bo_benchmark.sh|configs/experiment/hyp0_pfn_bench_nhp.yaml|pfns4neurostim-bench|tabpfn_v2_5,tabicl|gp_mll"
  "3. K2 NHP|stress_sweep|scripts/run_stress_sweep.sh|configs/experiment/stress_k2_nhp.yaml|pfns4neurostim|tabpfn_v2_5|gp_mll,gp_naive"
  "4. D3 PFN benchmark 5d_rat|bo_benchmark|scripts/run_bo_benchmark.sh|configs/experiment/hyp0_pfn_bench_5d_rat.yaml|pfns4neurostim-bench|tabpfn_v2_5,tabicl|gp_mll"
  "5. K2 5d_rat|stress_sweep|scripts/run_stress_sweep.sh|configs/experiment/stress_k2_5d_rat.yaml|pfns4neurostim|tabpfn_v2_5|gp_mll,gp_naive"
  "6. K5 5d_rat (optional, real lab artefacts)|stress_sweep|scripts/run_stress_sweep.sh|configs/experiment/stress_k5_5d_rat.yaml|pfns4neurostim|tabpfn_v2_5|gp_mll,gp_naive"
)

LAST_ASSEMBLE=""

# submit_wave <name> <experiment> <gpu-script> <config> <env> <gpu-models> <cpu-models> <n_reps> <after-job-id|"">
# Sets LAST_ASSEMBLE to the assemble job id.
submit_wave() {
  local name="$1" exp="$2" script="$3" cfg="$4" env="$5" gpu="$6" cpu="$7" reps="$8" after="${9:-}" deps="" id
  local dep=()
  if [ -n "${after}" ]; then dep=(--dependency="afterany:${after}"); fi
  id=$(CONDA_ENV="${env}" LANES=4 sbatch --parsable ${dep[@]+"${dep[@]}"} "${script}" "${cfg}" "models=[${gpu}]" tag=gpu "n_reps=${reps}")
  deps="${deps}:${id%%;*}"
  id=$(CONDA_ENV="${env}" sbatch --parsable ${dep[@]+"${dep[@]}"} scripts/run_cpu.sh "${exp}" "${cfg}" "models=[${cpu}]" tag=cpu "n_reps=${reps}")
  deps="${deps}:${id%%;*}"
  id=$(CONDA_ENV="${env}" sbatch --parsable --dependency="afterany${deps}" scripts/run_assemble.sh "${exp}" "${cfg}" "n_reps=${reps}")
  LAST_ASSEMBLE="${id%%;*}"
  echo "submitted: ${name} [n_reps=${reps}]  assemble job ${LAST_ASSEMBLE} runs after${deps}"
}

declare -A WAVE_A_DONE=()
for unit in "${UNITS[@]}"; do
  IFS='|' read -r name exp script cfg env gpu cpu <<< "${unit}"
  submit_wave "${name}" "${exp}" "${script}" "${cfg}" "${env}" "${gpu}" "${cpu}" "${WAVE_A_REPS}"
  WAVE_A_DONE["${name}"]="${LAST_ASSEMBLE}"
done
for unit in "${UNITS[@]}"; do
  IFS='|' read -r name exp script cfg env gpu cpu <<< "${unit}"
  submit_wave "${name}" "${exp}" "${script}" "${cfg}" "${env}" "${gpu}" "${cpu}" "${FINAL_REPS}" "${WAVE_A_DONE[${name}]}"
done

echo "Done. Check with: squeue --me"
