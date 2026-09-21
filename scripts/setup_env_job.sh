#!/bin/bash
# Build a conda env inside a compute allocation. `conda env create` parses the multi-GB conda-forge index and is killed
# by the login node's per-process memory limit (observed 2026-09-20: "Killed" while collecting repodata).
#
#   sbatch scripts/setup_env_job.sh bench     # builds pfns4neurostim-bench from environment.bench.yml (default)
#   sbatch scripts/setup_env_job.sh main      # (re)builds the main env from environment.yml
#
# Submodules are cloned separately on the login node (git is light): bash scripts/mila_setup.sh submodules
#SBATCH --job-name=setup-env
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err
#SBATCH --partition=main-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
set -euo pipefail

WHICH="${1:-bench}"
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_DIR}"

echo "[setup_env_job] node=$(hostname) job=${SLURM_JOB_ID:-none} env=${WHICH}"
# Fail early and clearly if the compute node has no outbound network (pip and conda both need it).
if ! curl -sI --max-time 15 https://pypi.org > /dev/null; then
  echo "[setup_env_job] ERROR: no outbound network from this node; build the env from a login node instead." >&2
  exit 1
fi

bash scripts/mila_setup.sh env "${WHICH}"
echo "[setup_env_job] done"
