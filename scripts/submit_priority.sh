#!/bin/bash
# Priority-ordered Mila submission for a tight deadline (2026-09-21). The USER runs this on the login node; the agent never submits.
#
#   N_REPS=5 bash scripts/submit_priority.sh          # wave 1: 5 reps per cell, everything that must be in the deck
#   bash scripts/submit_priority.sh                   # wave 2: the config default (10 reps); cells of reps 0-4 are cache hits
#
# Cells are keyed by (..., rep, seed) and the seed does not depend on n_reps, so wave 2 only computes reps 5-9. Order = queue
# priority (the per-user cap is 2 GPUs on `main`; main-cpu has its own 8 CPUs, so GP jobs run in parallel and serially with each
# other). Every unit is followed by an auto-assemble job (--dependency=afterany) that builds tidy.csv + figures from the cache.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
N_REPS="${N_REPS:-}"
REPS_SET=()
if [ -n "${N_REPS}" ]; then REPS_SET=("n_reps=${N_REPS}"); fi

# submit_unit <name> <experiment> <gpu-script> <config> <env> <gpu-models|-> <cpu-models|->
submit_unit() {
  local name="$1" exp="$2" script="$3" cfg="$4" env="$5" gpu="$6" cpu="$7" deps="" id
  if [ "$gpu" != "-" ]; then
    id=$(CONDA_ENV="$env" LANES=4 sbatch --parsable "$script" "$cfg" "models=[$gpu]" tag=gpu ${REPS_SET[@]+"${REPS_SET[@]}"})
    deps="$deps:${id%%;*}"
  fi
  if [ "$cpu" != "-" ]; then
    id=$(CONDA_ENV="$env" sbatch --parsable scripts/run_cpu.sh "$exp" "$cfg" "models=[$cpu]" tag=cpu ${REPS_SET[@]+"${REPS_SET[@]}"})
    deps="$deps:${id%%;*}"
  fi
  id=$(CONDA_ENV="$env" sbatch --parsable --dependency="afterany$deps" scripts/run_assemble.sh "$exp" "$cfg" ${REPS_SET[@]+"${REPS_SET[@]}"})
  echo "submitted: $name  (assemble job ${id%%;*} runs after$deps)"
}

# 1. Deliverable 1, 5d_rat (NHP D1 runs locally).
submit_unit "1. D1 5d_rat" bo_benchmark scripts/run_bo_benchmark.sh configs/experiment/hyp_a_5d_rat.yaml pfns4neurostim "tabpfn_v2_5" "gp_mll,gp_naive,random"
# 2. Deliverable 3, NHP (TabPFN-2.5, TabICL, GP-MLL; TabFM is not scheduled).
submit_unit "2. D3 PFN benchmark NHP" bo_benchmark scripts/run_bo_benchmark.sh configs/experiment/hyp0_pfn_bench_nhp.yaml pfns4neurostim-bench "tabpfn_v2_5,tabicl" "gp_mll"
# 3. Deliverable 2, K2 noise amplification, NHP.
submit_unit "3. K2 NHP" stress_sweep scripts/run_stress_sweep.sh configs/experiment/stress_k2_nhp.yaml pfns4neurostim "tabpfn_v2_5" "gp_mll,gp_naive"
# 4. Deliverable 3, 5d_rat.
submit_unit "4. D3 PFN benchmark 5d_rat" bo_benchmark scripts/run_bo_benchmark.sh configs/experiment/hyp0_pfn_bench_5d_rat.yaml pfns4neurostim-bench "tabpfn_v2_5,tabicl" "gp_mll"
# 5. Deliverable 2, K2 noise amplification, 5d_rat (longest unit).
submit_unit "5. K2 5d_rat" stress_sweep scripts/run_stress_sweep.sh configs/experiment/stress_k2_5d_rat.yaml pfns4neurostim "tabpfn_v2_5" "gp_mll,gp_naive"
# 6. Optional: K5 outliers on 5d_rat (real lab artefacts). Last, so it never delays the units above.
submit_unit "6. K5 5d_rat" stress_sweep scripts/run_stress_sweep.sh configs/experiment/stress_k5_5d_rat.yaml pfns4neurostim "tabpfn_v2_5" "gp_mll,gp_naive"

echo "Done. Check with: squeue --me"
