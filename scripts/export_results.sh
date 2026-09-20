#!/bin/bash
# ============================================================
#  Pull results from Mila into the local ./output tree.
#
#  Run it from a shell that has rsync AND the SSH control socket. On the Windows dev box that
#  is WSL (Git Bash has no rsync, and the socket lives in WSL):
#
#    wsl -e bash -lc 'cd /mnt/c/workspace/PFNs4Neurostim && bash scripts/export_results.sh'
#    wsl -e bash -lc 'cd /mnt/c/workspace/PFNs4Neurostim && bash scripts/export_results.sh stress/k2_snr/nhp'
#
#  Usage:
#    bash scripts/export_results.sh [subpath under output/]     # default: benchmark/ stress/ cells/
#
#  It reuses the control socket opened with `bash scripts/mila.sh open`, so it never asks for an
#  OTP and fails within seconds if the socket is down. Transfer is one-way (Mila -> local) and
#  never deletes local files. Pulling output/cells/ also brings the per-cell cache home, so a
#  later local run reuses every finished cell and `--replot` rebuilds figures from tidy.csv.
#
#  Env: MILA_HOST (default mila), MILA_SOCKET (default ~/.ssh/cm-mila.sock),
#       MILA_REMOTE_OUTPUT (default ~/scratch/pfns4neurostim/output).
# ============================================================
set -euo pipefail

SUBPATH="${1:-}"
MILA_HOST="${MILA_HOST:-mila}"
MILA_SOCKET="${MILA_SOCKET:-$HOME/.ssh/cm-mila.sock}"
REMOTE_OUT="${MILA_REMOTE_OUTPUT:-~/scratch/pfns4neurostim/output}"
LOCAL_OUT="./output"

command -v rsync >/dev/null 2>&1 || { echo "rsync not found: run this from WSL/Linux." >&2; exit 1; }
SSH_CMD="ssh -S ${MILA_SOCKET} -o ControlMaster=no -o BatchMode=yes -o ConnectTimeout=10"

if [ -n "${SUBPATH}" ]; then
    TARGETS=("${SUBPATH}")
else
    TARGETS=(benchmark stress cells)
fi

for t in "${TARGETS[@]}"; do
    mkdir -p "${LOCAL_OUT}/${t}"
    echo "Syncing ${t}"
    rsync -avz --progress -e "${SSH_CMD}" "${MILA_HOST}:${REMOTE_OUT}/${t}/" "${LOCAL_OUT}/${t}/"
done

echo "Sync complete -> ${LOCAL_OUT}"
