#!/bin/bash
# Supervised access to Narval (Alliance Canada) over an SSH control socket -- scripts/mila.sh with Narval's host, paths
# and quota tool for everything read-only, plus `do`, the ONE sanctioned way for the agent to change cluster state.
#
# The USER opens the socket (password + Duo, once). `run` and the views below only read and refuse mutations. `do` is an
# allowlist, not a shell: each subcommand builds one fixed command from validated arguments, refuses everything else, and
# appends the exact remote command and its outcome to a local audit log, so every state change is reviewable afterwards.
#
#   bash scripts/narval.sh open | status | close
#   bash scripts/narval.sh queue | logs <jobid> [--tail N] | sacct <jobid> | eff <jobid> | share | avail | quota
#   bash scripts/narval.sh run -- <read-only cmd>
#   bash scripts/narval.sh pull [subpath under output/]              # results Narval -> local (scripts/export_results.sh)
#   bash scripts/narval.sh do pull                                   # git pull --ff-only + submodule update in the cluster checkout
#   bash scripts/narval.sh do setup <layout|stage|submodules|check-data|env|install|weights|verify> [main|bench|v1] [models]
#   bash scripts/narval.sh do sbatch <gpu|cpu> [--lanes N] [--conda-env E] [--gpu-type T] [--mem M] [--cpus C] [--time T]
#                                    [--dependency afterany:ID[:ID]] [--job-name N] -- <scripts/run_*.sh> <args...>
#
# Add NARVAL_DRY_RUN=1 to any `do` call to print the exact remote command and log it as a dry run without touching the cluster.
# Env: NARVAL_HOST (default narval), NARVAL_SOCKET (default ~/.ssh/cm-narval.sock),
#      NARVAL_CODE_DIR (default ~/projects/def-bonizzat/mauriv/my-projects/PFNs4Neurostim),
#      NARVAL_SCRATCH_ROOT (default ~/scratch/pfns4neurostim; data/ output/ logs/ live there),
#      NARVAL_AUDIT_LOG (default output/logs/narval_audit.log, local and gitignored).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

export MILA_HOST="${NARVAL_HOST:-narval}"
export MILA_SOCKET="${NARVAL_SOCKET:-$HOME/.ssh/cm-narval.sock}"
export MILA_REMOTE_ROOT="${NARVAL_SCRATCH_ROOT:-~/scratch/pfns4neurostim}"
export REMOTE_CODE_DIR="${NARVAL_CODE_DIR:-~/projects/def-bonizzat/mauriv/my-projects/PFNs4Neurostim}"
export REMOTE_QUOTA_CMD="diskusage_report"
AUDIT_LOG="${NARVAL_AUDIT_LOG:-${HERE}/../output/logs/narval_audit.log}"

die() { printf '[narval do] refused: %s\n' "$*" >&2; exit 2; }
need() { [[ "$2" =~ $3 ]] || die "$1 '$2' does not match $3"; }   # need <what> <value> <regex>

# Run one fixed remote command over the control socket and record it. Fails within seconds when the socket is down.
remote_do() {   # remote_do <label> <remote command>
  local label="$1" cmd="$2" rc=0 out=""
  mkdir -p "$(dirname "${AUDIT_LOG}")"
  if [ "${NARVAL_DRY_RUN:-0}" = "1" ]; then
    printf '%s | DRY-RUN | %s | %s\n' "$(date -Is)" "${label}" "${cmd}" >> "${AUDIT_LOG}"
    printf '%s\n' "${cmd}"; return 0
  fi
  out=$(ssh -S "${MILA_SOCKET}" -o ControlMaster=no -o BatchMode=yes -o ConnectTimeout=15 "${MILA_HOST}" "${cmd}" 2>&1) || rc=$?
  printf '%s | rc=%s | %s | %s | %s\n' "$(date -Is)" "${rc}" "${label}" "${cmd}" "$(printf '%s' "${out}" | tail -n 1 | cut -c1-160)" >> "${AUDIT_LOG}"
  printf '%s\n' "${out}"
  return "${rc}"
}

do_pull() {
  remote_do "pull" "cd ${REMOTE_CODE_DIR} && git pull --ff-only && git submodule update --init --recursive && git log -1 --oneline"
}

do_setup() {   # do_setup <subcommand> [env] [models]
  local sub="${1:?usage: do setup <subcommand> [main|bench|v1] [models]}"; shift
  need subcommand "${sub}" '^(layout|stage|submodules|check-data|env|install|weights|verify)$'
  local args=""
  for a in "$@"; do need argument "${a}" '^[A-Za-z0-9_,.-]+$'; args+=" ${a}"; done
  remote_do "setup ${sub}" "cd ${REMOTE_CODE_DIR} && bash scripts/narval_setup.sh ${sub}${args}"
}

do_sbatch() {   # do_sbatch <gpu|cpu> [options] -- <script> <args...>
  local kind="${1:?usage: do sbatch <gpu|cpu> [options] -- <script> <args...>}"; shift
  need kind "${kind}" '^(gpu|cpu)$'
  local lanes="" conda_env="pfns4neurostim" gpu_type="" extra=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
    case "$1" in
      --lanes) need lanes "$2" '^[0-9]{1,2}$'; lanes="$2" ;;
      --conda-env) need env "$2" '^pfns4neurostim(-bench|-v1)?$'; conda_env="$2" ;;
      --gpu-type) need gpu-type "$2" '^a100(_[0-9]g\.[0-9]+gb)?$'; gpu_type="$2" ;;
      --mem) need mem "$2" '^[0-9]{1,3}[MG]$'; extra+=("--mem=$2") ;;
      --cpus) need cpus "$2" '^[0-9]{1,2}$'; extra+=("--cpus-per-task=$2") ;;
      --time) need time "$2" '^([0-9]{1,2}-)?[0-9]{1,2}:[0-9]{2}(:[0-9]{2})?$'; extra+=("--time=$2") ;;
      --dependency) need dependency "$2" '^afterany(:[0-9]+)+$'; extra+=("--dependency=$2") ;;
      --job-name) need job-name "$2" '^[A-Za-z0-9_.-]+$'; extra+=("--job-name=$2") ;;
      *) die "unknown option $1" ;;
    esac
    shift 2
  done
  [ "${1:-}" = "--" ] || die "missing '--' before the script"; shift
  local script="${1:?missing script}" ; shift
  need script "${script}" '^scripts/run_[a-z_]+\.sh$'
  [ "${kind}" = "gpu" ] || [ -z "${gpu_type}" ] || die "--gpu-type only applies to a gpu job"
  local args=()
  for a in "$@"; do
    [[ "${a}" != *"'"* && "${a}" != *'`'* && "${a}" != *'$'* && "${a}" != *';'* && "${a}" != *'&'* && "${a}" != *'|'* && "${a}" != *'<'* && "${a}" != *'>'* ]] || die "argument '${a}' contains a shell metacharacter"
    args+=("$(printf '%q' "${a}")")
  done
  local envs="CONDA_ENV=${conda_env}"; [ -z "${lanes}" ] || envs+=" LANES=${lanes}"
  local flags="\$(${gpu_type:+NARVAL_GPU=${gpu_type} }bash scripts/cluster.sh flags ${kind})"
  remote_do "sbatch ${kind}" "cd ${REMOTE_CODE_DIR} && ${envs} sbatch --parsable ${flags} ${extra[*]:-} ${script} ${args[*]:-}"
}

case "${1:-}" in
  pull)
    shift
    export MILA_REMOTE_OUTPUT="${MILA_REMOTE_ROOT}/output"
    exec bash "${HERE}/export_results.sh" "$@" ;;
  do)
    shift
    case "${1:-}" in
      pull) do_pull ;;
      setup) shift; do_setup "$@" ;;
      sbatch) shift; do_sbatch "$@" ;;
      *) die "usage: do pull | do setup <subcommand> ... | do sbatch <gpu|cpu> ... -- <script> <args>" ;;
    esac
    exit $? ;;
esac
exec bash "${HERE}/mila.sh" "$@"
