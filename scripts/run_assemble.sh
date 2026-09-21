#!/bin/bash
# Assemble one experiment's tidy.csv, tables and figures from the shared cell cache, with NO compute.
#
# Submitted automatically by scripts/submit_portfolio.sh (`--dependency=afterany` on the unit's GPU and
# CPU jobs). Cells that are missing (e.g. a failed lane) are simply absent from the union.
#
#   sbatch scripts/run_assemble.sh bo_benchmark configs/experiment/hyp_a_5d_rat.yaml
#SBATCH --job-name=assemble
#SBATCH --output=logs/assemble_%j.out
#SBATCH --error=logs/assemble_%j.err
#SBATCH --partition=main-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail

EXPERIMENT="${1:?usage: sbatch scripts/run_assemble.sh <bo_benchmark|stress_sweep> <config.yaml> [key=value ...]}"
CONFIG="${2:?missing config.yaml}"
shift 2 || true

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_DIR}"
# shellcheck disable=SC1091
source "${REPO_DIR}/scripts/_job_common.sh"

job_activate
python -m pfns4neurostim "${EXPERIMENT}" --config "${CONFIG}" --only-cached --set "device=cpu" "$@"
echo "[run_assemble] done: ${EXPERIMENT} ${CONFIG}"
