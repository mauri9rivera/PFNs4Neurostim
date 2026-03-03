#!/bin/bash
# ============================================================
#  Run from LOCAL machine to pull results from Mila.
#  Usage: bash scripts/export_results.sh <mila_username>
# ============================================================
MILA_USER=${1:?Usage: $0 <mila_username>}
REMOTE="$MILA_USER@login.server.mila.quebec"
REMOTE_DIR="~/projects/PFNs4Neurostim/output/"
LOCAL_DIR="./output/"

rsync -avz --progress \
    "${REMOTE}:${REMOTE_DIR}results/"       "${LOCAL_DIR}results/"
rsync -avz --progress \
    "${REMOTE}:${REMOTE_DIR}fitness/"       "${LOCAL_DIR}fitness/"
rsync -avz --progress \
    "${REMOTE}:${REMOTE_DIR}optimization/"  "${LOCAL_DIR}optimization/"

echo "Sync complete -> ${LOCAL_DIR}"
