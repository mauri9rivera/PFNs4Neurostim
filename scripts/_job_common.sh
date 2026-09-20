#!/bin/bash
# Shared helpers for the SLURM dispatchers (sourced, never run directly).
#
#   source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/_job_common.sh"
#   job_activate                       # conda env: $CONDA_ENV (default pfns4neurostim)
#   job_stage_data "${CONFIG}"         # copies the dataset to $SLURM_TMPDIR, sets STAGED_ROOT
#   job_run bo_benchmark "${CONFIG}" "${OVERRIDES[@]}"
#
# Preemption/timeouts: sbatch sends TERM 300 s before the limit (--signal=B:TERM@300).
# `job_run` forwards it to the experiment, then requeues the job. Every finished cell is
# already in the cell cache (output/cells), so the requeued job resumes where this one stopped.

job_activate() {
  # `module` and `conda activate` reference unset variables, so they abort under `set -u`
  # (observed on Mila, job scripts patched by hand on 2026-09-18).
  set +u
  module load anaconda/3
  # shellcheck disable=SC1091
  conda activate "${CONDA_ENV:-pfns4neurostim}"
  set -u
  echo "[job] env=${CONDA_ENV:-pfns4neurostim} python=$(python --version 2>&1)"
}

job_stage_data() {
  local config="${1:?config path}"
  local dataset subdir
  dataset=$(python -c "import sys,yaml; d=yaml.safe_load(open(sys.argv[1])); print(d['defaults']['dataset'])" "${config}")
  declare -A data_dirs=( [nhp]=monkeys [rat]=rat [spinal]=spinal [5d_rat]=5d_rat )
  subdir="${data_dirs[${dataset}]:-${dataset}}"
  STAGED_ROOT="${SLURM_TMPDIR:-${REPO_DIR}}/data"
  mkdir -p "${STAGED_ROOT}"
  echo "[job] staging data/${subdir} -> ${STAGED_ROOT}"
  cp -r "${REPO_DIR}/data/${subdir}" "${STAGED_ROOT}/"
  du -sh "${STAGED_ROOT}/${subdir}"
}

job_run() {
  local experiment="${1:?experiment}" config="${2:?config}"
  shift 2
  export CLUSTER_DIAG="${CLUSTER_DIAG:-1}"
  srun python -m pfns4neurostim "${experiment}" --config "${config}" --set "dataset.data_root=${STAGED_ROOT}" "$@" &
  local pid=$!
  trap 'echo "[job] TERM received: stopping and requeueing"; kill -TERM ${pid} 2>/dev/null || true; wait ${pid} || true; scontrol requeue "${SLURM_JOB_ID}"; exit 0' TERM
  wait "${pid}"
}
