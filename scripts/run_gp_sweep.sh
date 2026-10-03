#!/bin/bash
# SLURM dispatcher for the GP-fixed hyperparameter sweep (scripts/gp_fixed_sweep.py): CPU only, ONE job,
# an internal worker pool (one process per core), no cell cache and no resume. The table is written to
# output/gp_fixed_sweep/<tag>/ and nothing else, so it cannot touch any existing GP-fixed result.
#
# Usage (through `narval.sh do sbatch cpu --cpus 8 ... -- scripts/run_gp_sweep.sh configs/experiment/hyp_a_nhp.yaml [args]`):
#   everything after the config goes to gp_fixed_sweep.py, e.g. --n-reps 10 --tag nhp
#SBATCH --job-name=gp-sweep
#SBATCH --output=logs/gp_sweep_%j.out
#SBATCH --error=logs/gp_sweep_%j.err
# Partition/account are cluster-specific and come from the command line: sbatch $(bash scripts/cluster.sh flags cpu) ...
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=9G
#SBATCH --time=01:00:00
set -euo pipefail

CONFIG="${1:?usage: sbatch scripts/run_gp_sweep.sh <config.yaml> [gp_fixed_sweep.py args]}"
shift || true
ARGS=("$@")

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_DIR}"
# shellcheck disable=SC1091
source "${REPO_DIR}/scripts/_job_common.sh"

echo "[run_gp_sweep] node=$(hostname) job=${SLURM_JOB_ID:-none} config=${CONFIG}"

job_activate
job_stage_data "${CONFIG}"
srun python scripts/gp_fixed_sweep.py --config "${CONFIG}" --workers "${SLURM_CPUS_PER_TASK:-8}" --data-root "${STAGED_ROOT}" ${ARGS[@]+"${ARGS[@]}"}

echo "[run_gp_sweep] done"
