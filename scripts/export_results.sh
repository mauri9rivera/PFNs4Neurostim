#!/bin/bash
# ============================================================
#  Run from the LOCAL machine to pull results from Mila.
#  Usage:
#    bash scripts/export_results.sh <mila_username>                # benchmark/, stress/ and cells/
#    bash scripts/export_results.sh <mila_username> stress/k2_snr  # one sub-tree of output/
#
#  The remote root matches scripts/mila_setup.sh ($SCRATCH/pfns4neurostim). Pulling
#  output/cells/ brings the per-cell cache home too, so figures regenerate with `--replot`
#  and a later local run reuses every finished cell.
# ============================================================
MILA_USER=${1:?Usage: $0 <mila_username> [subpath under output/]}
SUBPATH=${2:-}
REMOTE="$MILA_USER@login.server.mila.quebec"
REMOTE_OUT="${MILA_REMOTE_OUTPUT:-~/scratch/pfns4neurostim/output}"
LOCAL_OUT="./output"

if [ -n "$SUBPATH" ]; then
    TARGETS=("$SUBPATH")
else
    TARGETS=(benchmark stress cells)
fi

for t in "${TARGETS[@]}"; do
    mkdir -p "${LOCAL_OUT}/${t}"
    echo "Syncing ${t}"
    rsync -avz --progress "${REMOTE}:${REMOTE_OUT}/${t}/" "${LOCAL_OUT}/${t}/"
done

echo "Sync complete -> ${LOCAL_OUT}"
