#!/bin/bash
# SLURM dispatcher for the BO benchmarks (Hyp 0 / Hyp A, acquisition tables, PFN benchmark).
#
# Usage (from a Mila login node, repo root; the USER runs sbatch):
#   sbatch scripts/run_bo_benchmark.sh configs/experiment/hyp_a_nhp.yaml n_reps=5 dataset.emgs=[0,1,2]
#   CONDA_ENV=pfns4neurostim-bench sbatch --export=ALL scripts/run_bo_benchmark.sh configs/experiment/hyp0_pfn_bench_nhp.yaml
#
# The external PFNs (TabICL v2, TabFM, TabFlex) run in the Python 3.11 bench env, selected
# with CONDA_ENV. Finished cells are cached in output/cells, so a requeued job resumes.
# Results land under output/benchmark/<dataset>/<family>-<tag>/.
#SBATCH --job-name=bo-bench
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err
#SBATCH --partition=main
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=12:00:00
set -euo pipefail

CONFIG="${1:?usage: sbatch scripts/run_bo_benchmark.sh <config.yaml> [key=value ...]}"
shift || true
OVERRIDES=("$@")

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_DIR}"
# shellcheck disable=SC1091
source "${REPO_DIR}/scripts/_job_common.sh"

echo "[run_bo_benchmark] node=$(hostname) job=${SLURM_JOB_ID:-none} config=${CONFIG}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

job_activate
job_stage_data "${CONFIG}"
job_dispatch bo_benchmark "${CONFIG}" "${OVERRIDES[@]}"

echo "[run_bo_benchmark] done; deliverables under output/benchmark/"
