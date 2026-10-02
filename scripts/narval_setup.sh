#!/bin/bash
# Cluster setup for PFNs4Neurostim on Narval (Alliance Canada). The Narval counterpart of scripts/mila_setup.sh.
#
# Storage policy (Alliance):
#   project   ~/projects/def-bonizzat/...  group quota, backed up, never purged -> code + data master copy
#   $HOME     small quota, backed up                                         -> virtualenvs ($VENV_ROOT)
#   scratch   ~/scratch, large, NOT backed up, PURGED on a schedule          -> working data + output + logs
#   $SLURM_TMPDIR  node-local disk, per job                                  -> staged data during a job
# Mila's lesson carries over (a checkout on $SCRATCH was destroyed by its purge on 2026-09-08): code lives on
# project, scratch only holds what a re-stage can rebuild. There is no `touch` subcommand: refreshing access times
# to dodge the scratch purge is against Alliance policy. Results you need are pulled home (scripts/narval.sh pull).
#
# Usage (Narval LOGIN node, repo root). Login nodes have internet; compute nodes do not, so environments are built
# HERE, not in a job (unlike Mila's setup_env_job.sh). A venv build is light enough for a login node.
#   bash scripts/narval_setup.sh layout       # scratch dirs + data/ output/ logs/ symlinks in the checkout
#   bash scripts/narval_setup.sh shared-data  # shared_data/ of SYMLINKS to datasets already on the cluster (no copy, nothing moved)
#   bash scripts/narval_setup.sh stage        # data master (shared_data/) -> scratch working copy (symlinks followed)
#   bash scripts/narval_setup.sh submodules   # git submodule update --init --recursive (+ ticl excludes)
#   bash scripts/narval_setup.sh env [main|bench|v1]   # build/update the virtualenv from environment*.yml
#   bash scripts/narval_setup.sh install [main|bench|v1]   # only pip install -e . --no-deps
#   bash scripts/narval_setup.sh verify       # layout, quotas, imports
#   bash scripts/narval_setup.sh weights [main|bench|v1] [models]   # pre-download checkpoints (compute nodes are offline)
#
# The data master is shared_data/ next to the projects. NHP, rat and the 5d_rat noOutliers cohort already exist inside
# additive_neurostim/datasets (checked byte-identical to the local copies on 2026-10-02), so `shared-data` only links them;
# that project is never modified. Spinal differs there (1.6 GB against 3.9 GB local) and is NOT linked: upload it when needed.
#
# Env: CODE_DIR, SCRATCH_ROOT, DATA_MASTER, plus everything scripts/cluster.sh reads (VENV_ROOT, NARVAL_STDENV,
# PY_MAIN, PY_BENCH). Nothing in this script deletes anything.
set -euo pipefail
cd "$(dirname "$0")/.."
export CLUSTER=narval
# shellcheck disable=SC1091
source scripts/cluster.sh

CODE_DIR="${CODE_DIR:-$(pwd)}"
SCRATCH_ROOT="${SCRATCH_ROOT:-${HOME}/scratch/pfns4neurostim}"
PROJECTS_ROOT="${PROJECTS_ROOT:-${HOME}/projects/def-bonizzat/mauriv/my-projects}"
DATA_MASTER="${DATA_MASTER:-${PROJECTS_ROOT}/shared_data}"
ADDITIVE_DATASETS="${ADDITIVE_DATASETS:-${PROJECTS_ROOT}/additive_neurostim/datasets}"

log() { printf '[narval_setup] %s\n' "$*"; }

env_name() {   # env_name <main|bench|v1>
  case "${1:-main}" in
    main) echo "pfns4neurostim" ;;
    bench) echo "pfns4neurostim-bench" ;;
    v1) echo "pfns4neurostim-v1" ;;
    *) log "usage: env [main|bench|v1]"; exit 2 ;;
  esac
}

env_file() {
  case "${1:-main}" in
    main) echo "environment.yml" ;;
    bench) echo "environment.bench.yml" ;;
    v1) echo "environment.v1.yml" ;;
  esac
}

load_python() {   # load_python <env name>
  local py
  py="$(cluster_python_version "$1")"
  set +u
  if ! module load "StdEnv/${NARVAL_STDENV}" "python/${py}" 2>/dev/null; then
    set -u
    log "ERROR: no python/${py} module under StdEnv/${NARVAL_STDENV}."
    log "  See what exists:   module spider python"
    log "  Options: an older StdEnv that still has ${py} (NARVAL_STDENV=<year>), or a newer Python for this env"
    log "  (PY_MAIN=3.11). A newer Python changes the pinned main stack (CLAUDE.md section 3) relative to Mila,"
    log "  so record the choice before comparing Narval and Mila results."
    exit 1
  fi
  set -u
  log "loaded StdEnv/${NARVAL_STDENV} python/${py} -> $(python --version 2>&1)"
}

cmd_layout() {
  log "code dir:     ${CODE_DIR}"
  log "scratch root: ${SCRATCH_ROOT}"
  log "data master:  ${DATA_MASTER}"
  [ -d "${CODE_DIR}/.git" ] || { log "ERROR: ${CODE_DIR} is not a git clone. Clone it first:"; log "  git clone --recurse-submodules https://github.com/mauri9rivera/PFNs4Neurostim.git ${CODE_DIR}"; exit 1; }
  mkdir -p "${SCRATCH_ROOT}/data" "${SCRATCH_ROOT}/output" "${SCRATCH_ROOT}/logs" "${DATA_MASTER}"
  for name in data output logs; do
    local target="${SCRATCH_ROOT}/${name}" link="${CODE_DIR}/${name}"
    if [ -L "${link}" ]; then log "symlink exists: ${link} -> $(readlink "${link}")"
    elif [ -e "${link}" ]; then log "WARNING: ${link} exists and is not a symlink; leaving it untouched."
    else ln -s "${target}" "${link}"; log "linked ${link} -> ${target}"
    fi
  done
}

cmd_shared_data() {
  # Link the datasets that already live in additive_neurostim into shared_data/ under the names this project loads
  # (data/monkeys, data/rat, data/5d_rat/<animal>/5D_step4_OutliersRemoved.mat). Symlinks only: nothing is copied,
  # moved, replaced or deleted, and the other project's tree is only read.
  [ -d "${ADDITIVE_DATASETS}" ] || { log "ERROR: ${ADDITIVE_DATASETS} not found."; exit 1; }
  mkdir -p "${DATA_MASTER}/5d_rat"
  link() {   # link <target> <link>
    [ -e "$1" ] || { log "ERROR: link target missing: $1"; exit 1; }
    if [ -L "$2" ]; then log "link exists: $2 -> $(readlink "$2")"
    elif [ -e "$2" ]; then log "WARNING: $2 exists and is not a symlink; leaving it untouched."
    else ln -s "$1" "$2"; log "linked $2 -> $1"
    fi
  }
  link "${ADDITIVE_DATASETS}/nhp" "${DATA_MASTER}/monkeys"
  link "${ADDITIVE_DATASETS}/rat" "${DATA_MASTER}/rat"
  local cohort="${ADDITIVE_DATASETS}/5d_rat/datasets_noOutliers/datasets_noOutliers" animal
  for animal in BCI00 rCer1.5 rCer1.12 rCer1.14 rCer1.15; do
    link "${cohort}/${animal}" "${DATA_MASTER}/5d_rat/${animal}"
  done
  log "spinal is not linked (the copy in additive_neurostim differs from the local one); upload it separately when needed."
}

cmd_stage() {
  [ -n "$(ls -A "${DATA_MASTER}" 2>/dev/null)" ] || { log "ERROR: ${DATA_MASTER} is empty; upload the data master first (see the header)."; exit 1; }
  log "restoring ${DATA_MASTER} -> ${SCRATCH_ROOT}/data"
  rsync -aL --info=progress2 "${DATA_MASTER}/" "${SCRATCH_ROOT}/data/"   # -L: copy what the symlinks point at
  du -sh "${SCRATCH_ROOT}/data"
}

cmd_submodules() {
  cd "${CODE_DIR}"
  git submodule update --init --recursive
  local ticl_git="${CODE_DIR}/.git/modules/libs/ticl"
  if [ -d "${ticl_git}" ]; then
    mkdir -p "${ticl_git}/info"
    printf '%s\n' 'ticl/models_diff/' '*.cpkt' > "${ticl_git}/info/exclude"
  fi
  git submodule status
}

# environment*.yml -> pip requirements: conda pins `pkg=1.2.3` become `pkg==1.2.3`, the pip: block passes through,
# python/pip themselves are dropped (the module provides Python).
yml_to_requirements() {   # yml_to_requirements <yml> <out>
  python - "$1" "$2" <<'PY'
import re, sys
import yaml
spec = yaml.safe_load(open(sys.argv[1]))
lines = []
for dep in spec.get("dependencies", []):
    if isinstance(dep, dict):
        lines.extend(str(x) for x in dep.get("pip", []))
        continue
    name = re.split(r"[=<>]", dep, maxsplit=1)[0].strip()
    if name in ("python", "pip"):
        continue
    m = re.match(r"^([A-Za-z0-9_.\-]+)=([^=].*)$", dep)
    lines.append(f"{m.group(1)}=={m.group(2)}" if m else dep)
open(sys.argv[2], "w").write("\n".join(lines) + "\n")
PY
}

cmd_env() {
  local which="${1:-main}" env file venv req
  env="$(env_name "${which}")"; file="$(env_file "${which}")"; venv="${VENV_ROOT}/${env}"
  load_python "${env}"
  mkdir -p "${VENV_ROOT}"
  if [ -x "${venv}/bin/python" ]; then log "venv exists: ${venv} (updating)"
  else log "creating ${venv}"; virtualenv --no-download "${venv}"
  fi
  # shellcheck disable=SC1091
  set +u; source "${venv}/bin/activate"; set -u
  pip install --no-index --upgrade pip
  pip install pyyaml                  # for the yml conversion below (Alliance wheelhouse or PyPI)
  req="$(mktemp)"
  yml_to_requirements "${file}" "${req}"
  log "installing $(grep -cv '^--' "${req}") pinned packages from ${file}"
  # Alliance's wheelhouse is searched first via its pip config; pins it does not carry come from PyPI / the torch
  # index on the login node. A pin that resolves to neither fails here, loudly, not inside a job.
  pip install -r "${req}"
  rm -f "${req}"
  cmd_install "${which}"
}

cmd_install() {
  local env; env="$(env_name "${1:-main}")"
  "${VENV_ROOT}/${env}/bin/python" -m pip install -e "${CODE_DIR}" --no-deps
  "${VENV_ROOT}/${env}/bin/python" -c "import pfns4neurostim; print('pfns4neurostim', pfns4neurostim.__version__, 'at', pfns4neurostim.__file__)"
}

cmd_weights() {
  # Several PFNs fetch their checkpoints on first use (TabPFN, TabICL, TabFlex, ...). Compute nodes have no internet,
  # so the first cell of a job would fail. A tiny fit + predict here, on the login node, fills the caches under $HOME,
  # which compute nodes mount. Default models per env follow scripts/preflight.sh.
  local which="${1:-main}" env models
  env="$(env_name "${which}")"
  case "${which}" in
    main) models="${2:-tabpfn_v2_5}" ;;
    bench) models="${2:-tabpfn_v2_5,tabicl,tabfm}" ;;
    v1) models="${2:-tabpfn_v1}" ;;
  esac
  "${VENV_ROOT}/${env}/bin/python" - "${models}" <<'PYEOF'
import sys
import numpy as np
from pfns4neurostim.models.registry import build_surrogate
rng = np.random.default_rng(0)
X, y = rng.random((12, 2)), rng.random(12)  # [12, 2], [12]
for name in sys.argv[1].split(","):
    model = build_surrogate(name, device="cpu")
    model.fit(X, y)
    mu, sigma = model.predict_marginals(X[:3])  # [3], [3]
    print(f"  {name:<14} ok  (weights cached, mu[0]={float(mu[0]):.3f})")
PYEOF
}

cmd_verify() {
  log "--- quotas ---"; diskusage_report 2>/dev/null || log "(diskusage_report unavailable)"
  log "--- code ---"; git -C "${CODE_DIR}" status -sb | head -5; git -C "${CODE_DIR}" submodule status
  log "--- symlinks ---"; ls -la "${CODE_DIR}" | grep -E '^l' || log "(no symlinks: run layout)"
  log "--- data ---"
  for d in "${CODE_DIR}"/data/*/; do [ -d "${d}" ] && printf '  %-12s %s files\n' "$(basename "${d}")" "$(find "${d}" -type f | wc -l)"; done
  log "--- environments ---"
  local which env py
  for which in main bench v1; do
    env="$(env_name "${which}")"; py="$(cluster_env_python "${env}")"
    if [ -z "${py}" ]; then printf '  %-22s not built\n' "${env}"; continue; fi
    printf '  %-22s ' "${env}"
    "${py}" -c "import pfns4neurostim, torch; print('ok, torch', torch.__version__)" 2>&1 | tail -1
  done
  log "next: bash scripts/preflight.sh"
}

case "${1:-}" in
  layout) cmd_layout ;;
  shared-data) cmd_shared_data ;;
  stage) cmd_stage ;;
  submodules) cmd_submodules ;;
  env) cmd_env "${2:-main}" ;;
  install) cmd_install "${2:-main}" ;;
  verify) cmd_verify ;;
  weights) cmd_weights "${2:-main}" "${3:-}" ;;
  *) sed -n 2,28p "$0"; exit 2 ;;
esac
