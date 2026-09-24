#!/bin/bash
# SLURM dispatcher for CPU-only work (the GP models and the random baseline) on `main-cpu`.
#
# GPs run on CPU anyway (2.2x faster than CUDA for this Adam GP), so they need no GPU. `main-cpu`
# has its own per-user cap (8 CPUs / 64 GB), separate from the 2 GPUs on `main`: it runs in
# addition to the GPU jobs. One lane per core, each single-threaded.
#
# Usage (the USER runs sbatch):
#   sbatch scripts/run_cpu.sh bo_benchmark configs/experiment/hyp_a_5d_rat.yaml "models=[gp_mll,gp_naive,random]"
#   sbatch scripts/run_cpu.sh stress_sweep configs/experiment/stress_k2_channel_nhp.yaml "models=[gp_mll,gp_naive]"
#
# Everything after the config path is forwarded to `--set`. Cells are cached in output/cells and shared
# with the GPU jobs of the same config; assemble the union with a final `--only-cached` run.
#SBATCH --job-name=cpu-lanes
#SBATCH --output=logs/cpu_%j.out
#SBATCH --error=logs/cpu_%j.err
#SBATCH --partition=main-cpu
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
set -euo pipefail

EXPERIMENT="${1:?usage: sbatch scripts/run_cpu.sh <bo_benchmark|stress_sweep> <config.yaml> [key=value ...]}"
CONFIG="${2:?missing config.yaml}"
shift 2 || true
# Do NOT force `device=cpu` here: `device` is part of every cell's cache identity, so forcing it makes these cells invisible
# to the GPU jobs and to the assembly of the same config (found on Mila 2026-09-20). The GP models carry their own CPU pin
# (model_params.<gp>.device: cpu) and Random is numpy-only, so no GPU is touched either way.
OVERRIDES=("$@")

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_DIR}"
# shellcheck disable=SC1091
source "${REPO_DIR}/scripts/_job_common.sh"

echo "[run_cpu] node=$(hostname) job=${SLURM_JOB_ID:-none} experiment=${EXPERIMENT} config=${CONFIG}"
export LANES="${LANES:-${SLURM_CPUS_PER_TASK:-8}}"

job_activate
job_stage_data "${CONFIG}"
job_dispatch "${EXPERIMENT}" "${CONFIG}" "${OVERRIDES[@]}"

echo "[run_cpu] done"
