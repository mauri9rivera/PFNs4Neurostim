#!/bin/bash
# SLURM dispatcher for a single-process experiment that needs NO GPU (added 2026-09-30).
#
# `scripts/run_single.sh` always asks for `--gres=gpu:1`, which is right for the mechanism analyses that
# embed with TabPFN (update rule, CKA) and wrong for the ones that do not: the MMD / sliced-W2 placement
# analysis declares `device: cpu` in its config and would otherwise hold one of the two GPUs the per-user
# cap allows for several hours while using none of it — delaying exactly the units that do need a GPU.
# `main-cpu` has its own separate cap, so this job runs *in addition to* the GPU ones.
#
# Usage (the USER runs sbatch, from a login node at the repo root):
#   sbatch scripts/run_single_cpu.sh mechanism configs/experiment/mechanism_placement_nhp.yaml
#   sbatch scripts/run_single_cpu.sh mechanism configs/experiment/mechanism_placement_5d_rat.yaml tag=5d-ctx
#
# Everything after the config path is forwarded to `--set`. Single-process analyses write their own
# deliverables (CSVs + figures) and have NO cell cache, so this job is not resumable: it either finishes
# inside the time limit or its work is lost.
#SBATCH --job-name=single-cpu
#SBATCH --output=logs/single_cpu_%j.out
#SBATCH --error=logs/single_cpu_%j.err
#SBATCH --partition=main-cpu
#SBATCH --signal=B:USR1@300
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
set -euo pipefail

EXPERIMENT="${1:?usage: sbatch scripts/run_single_cpu.sh <experiment> <config.yaml> [key=value ...]}"
CONFIG="${2:?missing config.yaml}"
shift 2 || true
OVERRIDES=("$@")

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_DIR}"
# shellcheck disable=SC1091
source "${REPO_DIR}/scripts/_job_common.sh"

echo "[run_single_cpu] node=$(hostname) job=${SLURM_JOB_ID:-none} experiment=${EXPERIMENT} config=${CONFIG}"

job_activate
job_stage_data "${CONFIG}"
srun python -m pfns4neurostim "${EXPERIMENT}" --config "${CONFIG}" --set "dataset.data_root=${STAGED_ROOT}" ${OVERRIDES[@]+"${OVERRIDES[@]}"}

echo "[run_single_cpu] done: ${EXPERIMENT} ${CONFIG}"
