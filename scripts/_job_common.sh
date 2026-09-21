#!/bin/bash
# Shared helpers for the SLURM dispatchers (sourced, never run directly).
#
#   source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/_job_common.sh"
#   job_activate                       # conda env: $CONDA_ENV (default pfns4neurostim)
#   job_stage_data "${CONFIG}"         # copies the dataset to $SLURM_TMPDIR, sets STAGED_ROOT
#   job_dispatch bo_benchmark "${CONFIG}" "${OVERRIDES[@]}"    # LANES=N (default 1) processes
#
# Signals: sbatch sends USR1 300 s before the time limit (--signal=B:USR1@300) -> stop and REQUEUE (resumes from the cell cache).
# TERM is what `scancel` and preemption send -> stop and EXIT, never requeue (an earlier version trapped TERM to requeue, so
# cancelling a job silently restarted it; found 2026-09-20 with job 10871694).
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
  trap 'echo "[job] time-limit warning (USR1): stopping and requeueing"; kill -TERM ${pid} 2>/dev/null || true; wait ${pid} || true; scontrol requeue "${SLURM_JOB_ID}"; exit 0' USR1
  trap 'echo "[job] TERM (scancel or preemption): stopping, NOT requeueing"; kill -TERM ${pid} 2>/dev/null || true; wait ${pid} || true; exit 143' TERM
  wait "${pid}"
}

# Multi-process lanes: N independent processes inside ONE job, each owning every N-th channel
# (`--shard i/N`). TabPFN uses ~5% of a GPU and ~1.3 GB of RAM, so the lanes share the job's GPU
# (measured ~1.7x aggregate throughput with 3 lanes); GP-only CPU jobs use one lane per core.
# Each lane runs single-threaded (lanes ARE the parallelism). A failed lane does not stop the
# others, its finished cells are already cached, and the job exits non-zero at the end.
LANE_PIDS=()

job_run_lanes() {
  local lanes="${1:?lanes}" experiment="${2:?experiment}" config="${3:?config}"
  shift 3
  export CLUSTER_DIAG="${CLUSTER_DIAG:-1}"
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  local i
  for ((i = 0; i < lanes; i++)); do
    python -m pfns4neurostim "${experiment}" --config "${config}" --set "dataset.data_root=${STAGED_ROOT}" "$@" --shard "${i}/${lanes}" > "logs/lane${i}_${SLURM_JOB_ID:-local}.out" 2>&1 &
    LANE_PIDS+=($!)
  done
  echo "[job] started ${lanes} lanes: logs/lane*_${SLURM_JOB_ID:-local}.out"
  trap 'echo "[job] time-limit warning (USR1): stopping lanes and requeueing"; kill -TERM "${LANE_PIDS[@]}" 2>/dev/null || true; wait || true; scontrol requeue "${SLURM_JOB_ID}"; exit 0' USR1
  trap 'echo "[job] TERM (scancel or preemption): stopping lanes, NOT requeueing"; kill -TERM "${LANE_PIDS[@]}" 2>/dev/null || true; wait || true; exit 143' TERM
  local failed=0 pid
  for pid in "${LANE_PIDS[@]}"; do
    wait "${pid}" || failed=1
  done
  return "${failed}"
}

# LANES=1 (default) keeps the single-process behaviour; LANES>1 runs that many sharded lanes.
job_dispatch() {
  local experiment="${1:?experiment}" config="${2:?config}"
  shift 2
  if [ "${LANES:-1}" -gt 1 ]; then
    job_run_lanes "${LANES}" "${experiment}" "${config}" "$@"
  else
    job_run "${experiment}" "${config}" "$@"
  fi
}
