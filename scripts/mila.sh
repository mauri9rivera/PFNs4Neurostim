#!/bin/bash
# Supervised, read-only access to the Mila cluster over an SSH control socket.
#
# Mila requires an OTP per SSH connection, so no unattended connection can be opened.
# The USER authenticates once (`open`), which creates a persistent control socket; the
# agent then reuses it with no prompt. Every `sbatch` stays with the user: `submit`
# only PRINTS the line to run.
#
#   bash scripts/mila.sh open                 # prints the command the USER runs (OTP)
#   bash scripts/mila.sh status               # is the socket live?
#   bash scripts/mila.sh close                # tear the socket down
#   bash scripts/mila.sh run -- <read-only cmd>
#   bash scripts/mila.sh logs <jobid> [--tail N]
#   bash scripts/mila.sh queue | sacct <jobid> | avail | quota
#   bash scripts/mila.sh submit <script> <config> [extra args]   # prints only
set -euo pipefail

MILA_HOST="${MILA_HOST:-mila}"
MILA_SOCKET="${MILA_SOCKET:-$HOME/.ssh/cm-mila.sock}"
MILA_REMOTE_ROOT="${MILA_REMOTE_ROOT:-\$SCRATCH/pfns4neurostim}"
# Defence in depth: `run` refuses anything that mutates state or submits work.
MILA_DENY_REGEX='(^|[;&|[:space:]])(sbatch|salloc|srun|scancel|scontrol|rm|mv|cp|chmod|kill|pkill|tee|dd)([[:space:]]|$)|pip[[:space:]]+(install|uninstall)|conda[[:space:]]+(create|install|remove|update|uninstall|env[[:space:]]+(create|update|remove))|>'

log() { printf '[mila] %s\n' "$*" >&2; }

# Fails within seconds (never hangs on an invisible OTP prompt) when the socket is down.
remote() {
  ssh -S "${MILA_SOCKET}" -o ControlMaster=no -o BatchMode=yes -o ConnectTimeout=10 "${MILA_HOST}" "$@"
}

cmd_open() {
  log "Run this yourself (it asks for your OTP once), then use 'status':"
  printf 'ssh -M -S %s -o ControlPersist=8h -Nf %s\n' "${MILA_SOCKET}" "${MILA_HOST}"
}

cmd_status() { ssh -S "${MILA_SOCKET}" -O check "${MILA_HOST}"; }
cmd_close() { ssh -S "${MILA_SOCKET}" -O exit "${MILA_HOST}"; }

cmd_run() {
  [ "${1:-}" = "--" ] && shift
  [ "$#" -ge 1 ] || { log "usage: run -- <command>"; exit 2; }
  local command="$*"
  # Harmless stderr redirects are not writes; strip them before the check.
  local check="${command//2>&1/}"
  check="${check//2>\/dev\/null/}"
  if [[ "${check}" =~ ${MILA_DENY_REGEX} ]]; then
    log "refused: '${command}' is not read-only. Ask the user to run it."
    exit 3
  fi
  remote "${command}"
}

cmd_logs() {
  local jobid="${1:-}" tail_n=""
  [[ "${jobid}" =~ ^[0-9]+$ ]] || { log "usage: logs <numeric jobid> [--tail N]"; exit 2; }
  shift
  if [ "${1:-}" = "--tail" ]; then
    tail_n="${2:?--tail needs a line count}"
    [[ "${tail_n}" =~ ^[0-9]+$ ]] || { log "--tail needs an integer"; exit 2; }
  fi
  local pattern="${MILA_REMOTE_ROOT}/logs/*_${jobid}.out ${MILA_REMOTE_ROOT}/logs/*_${jobid}.err"
  if [ -n "${tail_n}" ]; then
    remote "tail -n ${tail_n} ${pattern}"
  else
    remote "cat ${pattern}"
  fi
}

cmd_queue() { remote 'squeue --me'; }
cmd_sacct() {
  local jobid="${1:-}"
  [[ "${jobid}" =~ ^[0-9]+$ ]] || { log "usage: sacct <numeric jobid>"; exit 2; }
  remote "sacct -j ${jobid} --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode"
}
cmd_avail() { remote 'sinfo -o "%P %a %l %D %G" | head -20; savail 2>/dev/null || true'; }
cmd_quota() { remote 'disk-quota'; }

cmd_submit() {
  local script="${1:?usage: submit <script> <config> [extra args]}" config="${2:?missing config}"
  shift 2
  log "The agent never submits. Run this on the login node:"
  printf 'cd ~/projects/PFNs4Neurostim && sbatch %s %s %s\n' "${script}" "${config}" "$*"
}

sub="${1:-}"; shift || true
case "${sub}" in
  open) cmd_open ;;
  status) cmd_status ;;
  close) cmd_close ;;
  run) cmd_run "$@" ;;
  logs) cmd_logs "$@" ;;
  queue) cmd_queue ;;
  sacct) cmd_sacct "$@" ;;
  avail) cmd_avail ;;
  quota) cmd_quota ;;
  submit) cmd_submit "$@" ;;
  *) sed -n 2,17p "$0"; exit 2 ;;
esac
