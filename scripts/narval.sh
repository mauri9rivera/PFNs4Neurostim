#!/bin/bash
# Supervised, read-only access to Narval (Alliance Canada) over an SSH control socket -- scripts/mila.sh with
# Narval's host, paths and quota tool. Same rules: the USER opens the socket (password + Duo, once), the agent
# reuses it for read-only queries, and every sbatch/scancel stays with the user (`submit` only PRINTS the line).
#
#   bash scripts/narval.sh open | status | close
#   bash scripts/narval.sh queue | logs <jobid> [--tail N] | sacct <jobid> | eff <jobid> | share | avail | quota
#   bash scripts/narval.sh run -- <read-only cmd>
#   bash scripts/narval.sh submit <script> <config> [overrides]      # prints only
#   bash scripts/narval.sh pull [subpath under output/]              # results Narval -> local (scripts/export_results.sh)
#
# Env: NARVAL_HOST (default narval), NARVAL_SOCKET (default ~/.ssh/cm-narval.sock),
#      NARVAL_CODE_DIR (default ~/projects/def-bonizzat/mauriv/my-projects/PFNs4Neurostim),
#      NARVAL_SCRATCH_ROOT (default ~/scratch/pfns4neurostim; data/ output/ logs/ live there).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

export MILA_HOST="${NARVAL_HOST:-narval}"
export MILA_SOCKET="${NARVAL_SOCKET:-$HOME/.ssh/cm-narval.sock}"
export MILA_REMOTE_ROOT="${NARVAL_SCRATCH_ROOT:-~/scratch/pfns4neurostim}"
export REMOTE_CODE_DIR="${NARVAL_CODE_DIR:-~/projects/def-bonizzat/mauriv/my-projects/PFNs4Neurostim}"
export REMOTE_QUOTA_CMD="diskusage_report"

if [ "${1:-}" = "pull" ]; then
  shift
  export MILA_REMOTE_OUTPUT="${MILA_REMOTE_ROOT}/output"
  exec bash "${HERE}/export_results.sh" "$@"
fi
exec bash "${HERE}/mila.sh" "$@"
