#!/bin/bash
# Cluster abstraction: the ONE place that knows how Mila and Narval differ. Sourced by the job scripts and the
# login-node tools; also runnable for the two things a command line needs.
#
#   source scripts/cluster.sh                     # defines CLUSTER and the functions below
#   bash scripts/cluster.sh name                  # mila | narval | local
#   bash scripts/cluster.sh flags gpu|cpu         # sbatch flags this cluster needs (partition, account, GPU type)
#   bash scripts/cluster.sh python <env>          # interpreter path of an environment (prints nothing if absent)
#   bash scripts/cluster.sh setup                 # the setup script for this cluster
#
# Why the flags are not in the #SBATCH headers: a header can only be overridden on the command line, never removed,
# and Mila's `--partition=main` is an invalid partition on Narval (Alliance routes jobs to partitions itself). So the
# dispatchers carry only portable directives and every submission adds `$(bash scripts/cluster.sh flags gpu|cpu)`.
#
# Environments keep their conda names everywhere (pfns4neurostim, pfns4neurostim-bench, pfns4neurostim-latest), so
# CONDA_ENV, ExternalSpec.env and portfolio.py stay unchanged. On Narval each name is a virtualenv under
# $VENV_ROOT built by scripts/narval_setup.sh, because Alliance clusters ask users to use module Python + virtualenv
# rather than Anaconda.
#
# Overrides: CLUSTER, VENV_ROOT, NARVAL_ACCOUNT, NARVAL_GPU (e.g. a100, a100_3g.20gb), NARVAL_STDENV, PY_MAIN, PY_BENCH.

_detect_cluster() {
  if [ -n "${CLUSTER:-}" ]; then echo "${CLUSTER}"; return; fi
  # Alliance clusters export CC_CLUSTER on every node; Mila hostnames are *.server.mila.quebec / cn-*.
  if [ "${CC_CLUSTER:-}" = "narval" ] || hostname -f 2>/dev/null | grep -q 'narval'; then echo narval; return; fi
  if hostname -f 2>/dev/null | grep -qE 'mila\.quebec'; then echo mila; return; fi
  echo local
}
CLUSTER="$(_detect_cluster)"

VENV_ROOT="${VENV_ROOT:-${HOME}/venvs}"
NARVAL_ACCOUNT="${NARVAL_ACCOUNT:-def-bonizzat}"
NARVAL_GPU="${NARVAL_GPU:-a100}"
NARVAL_STDENV="${NARVAL_STDENV:-2023}"
# Mila and local keep the main env on Python 3.9 (CLAUDE.md section 3), but 3.9 is a reproducibility pin, not a
# requirement: every pinned package declares >=3.9 with no cap, the v1-prior dataset generator declares
# >=3.10,<3.13, and Narval offers 3.9 only under StdEnv/2020. Narval therefore runs the main env on
# Python 3.11 under StdEnv/2023 (decided 2026-10-02); record this before comparing Narval and Mila results.
# Set PY_MAIN=3.9 NARVAL_STDENV=2020 to get the pinned interpreter back.
PY_MAIN="${PY_MAIN:-3.11}"
PY_BENCH="${PY_BENCH:-3.11}"
PY_LATEST="${PY_LATEST:-3.11}"   # TabPFN-3.5 + Causilo (environment.latest.yml): python >= 3.10, torch >= 2.13

cluster_python_version() {   # cluster_python_version <env name>
  case "$1" in
    *-bench) echo "${PY_BENCH}" ;;
    *-latest) echo "${PY_LATEST}" ;;
    *) echo "${PY_MAIN}" ;;
  esac
}

cluster_sbatch_flags() {   # cluster_sbatch_flags <gpu|cpu>
  case "${CLUSTER}:${1:?gpu|cpu}" in
    mila:gpu) echo "--partition=main" ;;
    mila:cpu) echo "--partition=main-cpu" ;;
    narval:gpu) echo "--account=${NARVAL_ACCOUNT} --gres=gpu:${NARVAL_GPU}:1" ;;
    narval:cpu) echo "--account=${NARVAL_ACCOUNT}" ;;
    *) echo "" ;;
  esac
}

cluster_env_python() {   # cluster_env_python <env name>: the env's interpreter, or nothing
  local env="${1:?env name}" prefix
  case "${CLUSTER}" in
    narval) prefix="${VENV_ROOT}/${env}" ;;
    *)
      command -v conda >/dev/null 2>&1 || { set +u; module load anaconda/3 >/dev/null 2>&1 || true; set -u; }
      # Called DIRECTLY, never through `conda run`, which prints nothing in a non-interactive login shell (2026-09-30).
      prefix=$(conda env list 2>/dev/null | awk -v e="${env}" '$1 == e {print $NF}' | head -1)
      ;;
  esac
  [ -n "${prefix}" ] || return 0
  if [ -x "${prefix}/bin/python" ]; then echo "${prefix}/bin/python"
  elif [ -x "${prefix}/python.exe" ]; then echo "${prefix}/python.exe"   # Windows layout, for a local dry run
  fi
}

cluster_activate() {   # cluster_activate <env name>
  local env="${1:?env name}"
  # `module`, `conda activate` and venv activate scripts reference unset variables under `set -u` (Mila, 2026-09-18).
  set +u
  case "${CLUSTER}" in
    narval)
      module load "StdEnv/${NARVAL_STDENV}" "python/$(cluster_python_version "${env}")"
      # shellcheck disable=SC1090
      source "${VENV_ROOT}/${env}/bin/activate"
      ;;
    *)
      module load anaconda/3
      conda activate "${env}"
      ;;
  esac
  set -u
}

cluster_setup_script() {
  case "${CLUSTER}" in
    narval) echo "scripts/narval_setup.sh" ;;
    *) echo "scripts/mila_setup.sh" ;;
  esac
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case "${1:-}" in
    name) echo "${CLUSTER}" ;;
    flags) cluster_sbatch_flags "${2:-}" ;;
    python) cluster_env_python "${2:?env name}" ;;
    setup) cluster_setup_script ;;
    *) sed -n 2,9p "$0"; exit 2 ;;
  esac
fi
