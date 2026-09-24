#!/bin/bash
# SLURM dispatcher for the Hypothesis B stress sweeps (task #10, roadmap Phase 2).
#
# Usage (from a Mila login node, repo root; the USER runs sbatch):
#   sbatch scripts/run_stress_sweep.sh configs/experiment/stress_k2_channel_nhp.yaml
#   sbatch scripts/run_stress_sweep.sh configs/experiment/stress_k2_channel_5d_rat.yaml n_reps=10
#
# Everything after the config path is forwarded to `--set`. LANES=4 runs 4 sharded processes in the job. The dataset is staged to
# $SLURM_TMPDIR (node-local SSD). Finished cells are cached in output/cells, so a
# preempted, timed-out or crashed job resumes on requeue; results land under
# output/stress/<knob>/<dataset>/<family>-<tag>/ and are pulled home with
# scripts/export_results.sh, then re-plotted locally with `--replot`.
#SBATCH --job-name=stress
#SBATCH --output=logs/stress_%j.out
#SBATCH --error=logs/stress_%j.err
#SBATCH --partition=main
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=12:00:00
set -euo pipefail

CONFIG="${1:?usage: sbatch scripts/run_stress_sweep.sh <config.yaml> [key=value ...]}"
shift || true
OVERRIDES=("$@")

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_DIR}"
# shellcheck disable=SC1091
source "${REPO_DIR}/scripts/_job_common.sh"

echo "[run_stress_sweep] node=$(hostname) job=${SLURM_JOB_ID:-none} config=${CONFIG}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

job_activate
job_stage_data "${CONFIG}"
job_dispatch stress_sweep "${CONFIG}" "${OVERRIDES[@]}"

echo "[run_stress_sweep] done; deliverables under output/stress/"
