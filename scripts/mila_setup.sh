#!/bin/bash
# Cluster setup for PFNs4Neurostim on the Mila cluster.
#
# Storage policy (docs.mila.quebec, see .claude/all_docs_condensed.md):
#   $HOME     100 GB, backed up daily, never purged   -> code + conda env
#   $SCRATCH  5 TB, NOT backed up, PURGED after 90 days without access
#                                                     -> working data + output
#   $ARCHIVE  5 TB, long-term, login/CPU nodes only   -> raw-data master copy
#   $SLURM_TMPDIR  node-local SSD, per job            -> staged data during a job
#
# A previous checkout lived entirely on $SCRATCH and was destroyed by the
# 90-day purge (2026-09-08), including the .git directory. This layout keeps
# code on $HOME and an immutable data master on $ARCHIVE so a purge only ever
# costs a re-stage.
#
# Usage (run on a Mila LOGIN node; $ARCHIVE is not mounted on GPU nodes):
#   bash scripts/mila_setup.sh layout     # create directories + repo symlinks
#   bash scripts/mila_setup.sh archive    # copy $SCRATCH data -> $ARCHIVE master
#   bash scripts/mila_setup.sh stage      # restore $ARCHIVE master -> $SCRATCH
#   bash scripts/mila_setup.sh env        # create/update the conda env + pip install -e .
#   bash scripts/mila_setup.sh verify     # print the whole layout + import check
#   bash scripts/mila_setup.sh touch      # refresh atimes so $SCRATCH is not purged
#
# Nothing in this script deletes anything.
set -euo pipefail

PROJECT_NAME="pfns4neurostim"
CODE_DIR="${HOME}/projects/PFNs4Neurostim"
SCRATCH_ROOT="${SCRATCH:-/network/scratch/${USER:0:1}/${USER}}/${PROJECT_NAME}"
ARCHIVE_ROOT="${ARCHIVE:-/network/archive/${USER:0:1}/${USER}}/${PROJECT_NAME}"
CONDA_ENV="pfns4neurostim"

log() { printf '[mila_setup] %s\n' "$*"; }

cmd_layout() {
  log "code dir:    ${CODE_DIR}"
  log "scratch root:${SCRATCH_ROOT}"
  log "archive root:${ARCHIVE_ROOT}"
  mkdir -p "${SCRATCH_ROOT}/data" "${SCRATCH_ROOT}/output" "${SCRATCH_ROOT}/logs"
  mkdir -p "${ARCHIVE_ROOT}/data"
  if [ ! -d "${CODE_DIR}/.git" ]; then
    log "ERROR: ${CODE_DIR} is not a git clone yet. Clone it first:"
    log "  git clone --recurse-submodules https://github.com/mauri9rivera/PFNs4Neurostim.git ${CODE_DIR}"
    exit 1
  fi
  # data/ and output/ are gitignored, so symlinks are safe inside the checkout.
  for name in data output logs; do
    target="${SCRATCH_ROOT}/${name}"
    link="${CODE_DIR}/${name}"
    if [ -L "${link}" ]; then
      log "symlink exists: ${link} -> $(readlink "${link}")"
    elif [ -e "${link}" ]; then
      log "WARNING: ${link} exists and is not a symlink; leaving it untouched."
    else
      ln -s "${target}" "${link}"
      log "linked ${link} -> ${target}"
    fi
  done
}

cmd_archive() {
  log "copying ${SCRATCH_ROOT}/data -> ${ARCHIVE_ROOT}/data (master copy)"
  rsync -a --info=progress2 "${SCRATCH_ROOT}/data/" "${ARCHIVE_ROOT}/data/"
  du -sh "${ARCHIVE_ROOT}/data"
}

cmd_stage() {
  log "restoring ${ARCHIVE_ROOT}/data -> ${SCRATCH_ROOT}/data (working copy)"
  rsync -a --info=progress2 "${ARCHIVE_ROOT}/data/" "${SCRATCH_ROOT}/data/"
  du -sh "${SCRATCH_ROOT}/data"
}

cmd_env() {
  cd "${CODE_DIR}"
  log "updating conda env '${CONDA_ENV}' from environment.yml"
  conda env update -f environment.yml -n "${CONDA_ENV}"
  log "installing the package editable"
  conda run -n "${CONDA_ENV}" pip install -e .
}

cmd_verify() {
  log "--- quotas ---"; disk-quota 2>/dev/null || log "(disk-quota unavailable)"
  log "--- code ---"
  cd "${CODE_DIR}"
  git status -sb | head -5
  git submodule status
  log "--- storage ---"
  du -sh "${SCRATCH_ROOT}"/* 2>/dev/null || true
  du -sh "${ARCHIVE_ROOT}"/* 2>/dev/null || true
  log "--- symlinks ---"
  ls -la "${CODE_DIR}" | grep -E '^l' || log "(no symlinks)"
  log "--- data files per dataset ---"
  for d in "${CODE_DIR}"/data/*/; do
    [ -d "${d}" ] && printf '  %-12s %s files\n' "$(basename "${d}")" "$(find "${d}" -type f | wc -l)"
  done
  log "--- import check ---"
  conda run -n "${CONDA_ENV}" python -c "import pfns4neurostim, torch; from pfns4neurostim.visualization import style; print('package OK, torch', torch.__version__, 'cuda', torch.cuda.is_available())"
}

cmd_touch() {
  # $SCRATCH is purged after 90 days without access. Refresh atimes monthly.
  log "refreshing access times under ${SCRATCH_ROOT}"
  find "${SCRATCH_ROOT}" -type f -exec touch -a {} +
  log "done"
}

case "${1:-}" in
  layout) cmd_layout ;;
  archive) cmd_archive ;;
  stage) cmd_stage ;;
  env) cmd_env ;;
  verify) cmd_verify ;;
  touch) cmd_touch ;;
  *) sed -n '1,30p' "$0"; exit 1 ;;
esac
