#!/bin/bash
# SLURM dispatcher for runners that run as ONE process and write their own deliverables
# (no lanes, no --compute-only, no separate assemble job): `mechanism` (Hyp C).
#
# Usage (from a Mila login node, repo root; the USER runs sbatch):
#   sbatch scripts/run_single.sh mechanism configs/experiment/mechanism_update_rule_nhp.yaml
#
# Everything after the config path is forwarded to `--set`. The dataset is staged to $SLURM_TMPDIR.
#SBATCH --job-name=single
#SBATCH --output=logs/single_%j.out
#SBATCH --error=logs/single_%j.err
# Partition/account/GPU type are cluster-specific and come from the command line: sbatch $(bash scripts/cluster.sh flags gpu) ...
# No --signal: the runner has no USR1 handler, so a warning signal only killed the job 5 min early (both M10 runs of
# 2026-10-04 died at 5 h 55 min). Since 2026-10-06 (B3) every mechanism cell is persisted the moment it finishes
# (output/cells/<dataset>/mechanism_<analysis>/), so a job that hits its wall limit is RESUMED by submitting the same
# line again: finished cells are served from the cache and only the rest is computed. The limit no longer has to cover
# the whole config, only make steady progress.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
set -euo pipefail

EXPERIMENT="${1:?usage: sbatch scripts/run_single.sh <mechanism> <config.yaml> [key=value ...]}"
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
