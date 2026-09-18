#!/bin/bash
# SLURM dispatcher for the Hypothesis B stress sweeps (task #10, roadmap Phase 2).
#
# Usage (from a Mila login node, repo root):
#   sbatch scripts/run_stress_sweep.sh configs/experiment/stress_k2_nhp.yaml
#   sbatch scripts/run_stress_sweep.sh configs/experiment/stress_k2_5d_rat.yaml n_reps=10
#
# Everything after the config path is forwarded to `--set`, so any resolved key
# can be overridden without editing the YAML.
#
# Data staging: the raw .mat trees are copied to $SLURM_TMPDIR (node-local SSD)
# and the run is pointed at them via dataset.data_root, per the documented Mila
# staging pattern. Results are written to $SCRATCH through the repo's output
# symlink; pull them home with
#   rsync -avz mila:~/scratch/pfns4neurostim/output/stress/ output/stress/
# and regenerate every figure locally with `--replot`.
#SBATCH --job-name=k2-stress
#SBATCH --output=logs/stress_%j.out
#SBATCH --error=logs/stress_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
set -euo pipefail

CONFIG="${1:?usage: sbatch scripts/run_stress_sweep.sh <config.yaml> [key=value ...]}"
shift || true
OVERRIDES=("$@")

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_DIR}"

echo "[run_stress_sweep] node=$(hostname) job=${SLURM_JOB_ID:-none} config=${CONFIG}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

module load anaconda/3
# shellcheck disable=SC1091
conda activate pfns4neurostim

# --- stage the dataset to node-local SSD -----------------------------------
DATASET=$(python -c "import sys,yaml; d=yaml.safe_load(open(sys.argv[1])); print(d['defaults']['dataset'])" "${CONFIG}")
declare -A DATA_DIRS=( [nhp]=monkeys [rat]=rat [spinal]=spinal [5d_rat]=5d_rat )
SUBDIR="${DATA_DIRS[${DATASET}]:-${DATASET}}"

STAGED_ROOT="${SLURM_TMPDIR:-${REPO_DIR}}/data"
mkdir -p "${STAGED_ROOT}"
echo "[run_stress_sweep] staging data/${SUBDIR} -> ${STAGED_ROOT}"
cp -r "${REPO_DIR}/data/${SUBDIR}" "${STAGED_ROOT}/"
du -sh "${STAGED_ROOT}/${SUBDIR}"

# --- run --------------------------------------------------------------------
srun python -m pfns4neurostim stress_sweep --config "${CONFIG}" --set "dataset.data_root=${STAGED_ROOT}" "${OVERRIDES[@]}"

echo "[run_stress_sweep] done; deliverables under output/stress/"
