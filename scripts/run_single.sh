#!/bin/bash
# SLURM dispatcher for runners that run as ONE process and write their own deliverables
# (no lanes, no --compute-only, no separate assemble job): `mechanism` (Hyp C) and `gt_sensitivity`.
#
# Usage (from a Mila login node, repo root; the USER runs sbatch):
#   sbatch scripts/run_single.sh mechanism configs/experiment/mechanism_update_rule_nhp.yaml
#   sbatch scripts/run_single.sh gt_sensitivity configs/experiment/gt_sensitivity_nhp.yaml n_reps=5
#
# Everything after the config path is forwarded to `--set`. The dataset is staged to $SLURM_TMPDIR.
#SBATCH --job-name=single
#SBATCH --output=logs/single_%j.out
#SBATCH --error=logs/single_%j.err
#SBATCH --partition=main
#SBATCH --signal=B:USR1@300
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
set -euo pipefail

EXPERIMENT="${1:?usage: sbatch scripts/run_single.sh <mechanism|gt_sensitivity> <config.yaml> [key=value ...]}"
CONFIG="${2:?missing config.yaml}"
shift 2 || true

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_DIR}"
# shellcheck disable=SC1091
source "${REPO_DIR}/scripts/_job_common.sh"

echo "[run_single] node=$(hostname) job=${SLURM_JOB_ID:-none} ${EXPERIMENT} config=${CONFIG}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

job_activate
job_stage_data "${CONFIG}"
srun python -m pfns4neurostim "${EXPERIMENT}" --config "${CONFIG}" --set "dataset.data_root=${STAGED_ROOT}" "$@"
echo "[run_single] done: ${EXPERIMENT} ${CONFIG}"
