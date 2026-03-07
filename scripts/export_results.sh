#!/bin/bash
# ============================================================
#  Run from LOCAL machine to pull results from Mila.
#  Usage:
#    bash scripts/export_results.sh <mila_username>               # sync all runs
#    bash scripts/export_results.sh <mila_username> <tag_pattern> # sync matching runs only
#
#  Examples:
#    bash scripts/export_results.sh mauricio.rivera
#    bash scripts/export_results.sh mauricio.rivera nhp_inter_subject
#    bash scripts/export_results.sh mauricio.rivera nhp_inter_subject_subj0_ep20_lr1.00e-06_aug10_20260306_120154
# ============================================================
MILA_USER=${1:?Usage: $0 <mila_username> [tag_pattern]}
PATTERN=${2:-}
REMOTE="$MILA_USER@login.server.mila.quebec"
REMOTE_RUNS="~/scratch/projects/PFNs4Neurostim/output/runs/"
LOCAL_RUNS="./output/runs/"

mkdir -p "$LOCAL_RUNS"

if [ -n "$PATTERN" ]; then
    echo "Syncing runs matching: ${PATTERN}*"
    rsync -avz --progress \
        --filter="+ ${PATTERN}*/"   \
        --filter="+ ${PATTERN}*/**" \
        --filter="- *"              \
        "${REMOTE}:${REMOTE_RUNS}" "${LOCAL_RUNS}"
else
    echo "Syncing all runs"
    rsync -avz --progress "${REMOTE}:${REMOTE_RUNS}" "${LOCAL_RUNS}"
fi

echo "Sync complete -> ${LOCAL_RUNS}"
