#!/bin/bash
# Clean the Mila cluster's data, cell cache and run directories after the 2026-09-25 changes (login node).
#
# Three independent retirements, each verified by its own tool, all dry-run unless --apply:
#   1. 5d_rat raw files of the superseded cohort: every top-level entry of data/5d_rat that is neither an
#      animal directory of the current cohort (loaders.rat_5d.SUBJECTS) nor the provenance zip is MOVED to
#      data/_retired/5d_rat-<old cohort>/ (the loader reads an explicit subject list, so this is hygiene).
#   2. Results of the superseded 5d_rat cohort (scripts/retire_dataset_cohort.py): cells DELETED (their
#      identity carries the old cohort, they can never be hits), run data QUARANTINED in <run>/_stale_cohort/
#      with figures left in place as a proxy.
#   3. Results the code can no longer request (scripts/prune_results.py): renamed/removed knobs, old K5
#      parameters, stale model versions (TabFM before 2026-09-25), pre-2026-09-23 shard dirs, the retired
#      hyp0-pfns4bo runs. MOVED to output/archive/<stamp>-pruned/ with a manifest (delete it once satisfied).
#
# Before --apply the new cohort must be on the cluster (the script refuses otherwise), and the repo must be at
# the commit that carries the cohort stamp (it checks the stamp). Upload from your machine, then stage:
#   see docs/mila_runbook.md section 3b.
#
#   bash scripts/clean_mila_results.sh            # dry run: report what each step would do
#   bash scripts/clean_mila_results.sh --apply    # perform all three
set -euo pipefail
cd "$(dirname "$0")/.."

APPLY=""
if [ "${1:-}" = "--apply" ]; then APPLY="--apply"; fi
ENV_NAME="pfns4neurostim"
DATASET="5d_rat"

# Activate the main env unless it is already active. Not `conda run`: the cluster's conda predates
# `--no-capture-output` (2026-09-25), and activation behaves the same on every conda version.
# Activation hooks reference unset variables and may return non-zero, so -e and -u are off around it
# (as in scripts/_job_common.sh); success is checked explicitly afterwards.
if [ "${CONDA_DEFAULT_ENV:-}" != "${ENV_NAME}" ]; then
  set +eu
  if ! command -v conda >/dev/null 2>&1; then module load anaconda/3; fi
  if [ "$(type -t conda)" != "function" ]; then source "$(conda info --base)/etc/profile.d/conda.sh"; fi
  conda activate "${ENV_NAME}"
  set -eu
  if [ "${CONDA_DEFAULT_ENV:-}" != "${ENV_NAME}" ]; then
    echo "[clean] could not activate ${ENV_NAME}; run 'conda activate ${ENV_NAME}' first" >&2; exit 1
  fi
fi
py() { python "$@"; }

COHORT=$(py -c "from pfns4neurostim.data.loaders import data_cohort; print(data_cohort('${DATASET}') or '')")
ANIMALS=$(py -c "from pfns4neurostim.data.loaders.rat_5d import SUBJECTS; print(' '.join(s.key for s in SUBJECTS))")
if [ -z "${COHORT}" ]; then
  echo "[clean] this checkout has no ${DATASET} cohort stamp: git pull first" >&2; exit 1
fi
DATA_DIR="data/${DATASET}"
echo "[clean] cohort to keep: ${COHORT}; animals: ${ANIMALS}"

# --- 1. old raw files -----------------------------------------------------------------------
missing=""
for a in ${ANIMALS}; do [ -d "${DATA_DIR}/${a}" ] || missing="${missing} ${a}"; done
if [ -n "${missing}" ]; then
  echo "[clean] new cohort NOT on the cluster (missing:${missing}); upload it first (runbook 3b)." >&2
  [ -n "${APPLY}" ] && exit 1
fi
RETIRED="data/_retired/${DATASET}-pre-${COHORT}"
stale=()
for entry in "${DATA_DIR}"/*; do
  name=$(basename "${entry}")
  case " ${ANIMALS} " in *" ${name} "*) continue ;; esac
  [[ "${name}" == *.zip ]] && continue
  stale+=("${entry}")
done
echo "[clean] 1. ${#stale[@]} superseded entries in ${DATA_DIR} -> ${RETIRED}/"
for entry in "${stale[@]}"; do
  echo "    $(basename "${entry}")"
  if [ -n "${APPLY}" ]; then mkdir -p "${RETIRED}"; mv "${entry}" "${RETIRED}/"; fi
done

# --- 2. results of the superseded cohort ------------------------------------------------------
echo "[clean] 2. retire ${DATASET} results not computed on ${COHORT}"
py scripts/retire_dataset_cohort.py --dataset "${DATASET}" --cohort "${COHORT}" ${APPLY}

# --- 3. results the code can no longer request ------------------------------------------------
echo "[clean] 3. prune results no config can request any more"
py scripts/prune_results.py ${APPLY}

# A one-off script copied here on 2026-09-24 (never committed); its configs no longer exist.
if [ -f scripts/fix_pfns4bo_resubmit.sh ]; then
  echo "[clean] one-off scripts/fix_pfns4bo_resubmit.sh is obsolete"
  if [ -n "${APPLY}" ]; then rm scripts/fix_pfns4bo_resubmit.sh; fi
fi
[ -z "${APPLY}" ] && echo "[clean] dry run: nothing touched. Re-run with --apply."
exit 0
