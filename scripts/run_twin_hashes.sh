#!/bin/bash
# SLURM dispatcher: fingerprint the synthetic twins a set of configs would build, ON A COMPUTE NODE (task plan A3 Step 15).
#
# Twins are refitted in every process and are not part of the cell key, so a unit whose TabPFN half runs on one cluster and
# whose GP half runs on another is only sound if both clusters build bit-identical twins. This job runs
# scripts/twin_hashes.py with the same BLAS threading as the BO lanes and writes output/twins/<label>_<cluster>_<jobid>.json;
# compare two of them with `python scripts/twin_hashes.py --compare A.json B.json`. No BO, no cells, about 2 CPU-minutes.
#
# Usage (run on the node class the real half will use -- the GPU partition for a TabPFN half, CPU for a GP half):
#   sbatch $(bash scripts/cluster.sh flags gpu) scripts/run_twin_hashes.sh synth configs/experiment/stress_k2_channel_synthetic_nhp.yaml configs/experiment/stress_k2_channel_synthetic_5d_rat.yaml
#   bash scripts/narval.sh do sbatch cpu --cpus 1 --mem 4G --time 00:15:00 --job-name twins -- scripts/run_twin_hashes.sh synth <configs...>
#SBATCH --job-name=twin-hashes
#SBATCH --output=logs/twins_%j.out
#SBATCH --error=logs/twins_%j.err
# Partition/account/GPU type are cluster-specific and come from the command line: sbatch $(bash scripts/cluster.sh flags cpu) ...
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail

LABEL="${1:?usage: sbatch scripts/run_twin_hashes.sh <label> <config.yaml> [config.yaml ...]}"
shift
[ "$#" -ge 1 ] || { echo "[run_twin_hashes] at least one config is required" >&2; exit 2; }
[[ "${LABEL}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "[run_twin_hashes] label must match [A-Za-z0-9_.-]+" >&2; exit 2; }

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_DIR}"
# shellcheck disable=SC1091
source "${REPO_DIR}/scripts/_job_common.sh"

CLUSTER_NAME="$(bash scripts/cluster.sh name)"
echo "[run_twin_hashes] node=$(hostname) job=${SLURM_JOB_ID:-none} cluster=${CLUSTER_NAME} label=${LABEL}"

job_activate
CONFIG_ARGS=()
for cfg in "$@"; do
  job_stage_data "${cfg}"
  CONFIG_ARGS+=(--config "${cfg}")
done
# Same numerics as the BO lanes (job_run_lanes / job_run): single-threaded BLAS.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT="output/twins/${LABEL}_${CLUSTER_NAME}_${SLURM_JOB_ID:-local}.json"
srun python scripts/twin_hashes.py "${CONFIG_ARGS[@]}" --set "dataset.data_root=${STAGED_ROOT}" --out "${OUT}"

echo "[run_twin_hashes] done: ${OUT}"
