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
#   bash scripts/narval_setup.sh check-data   # verify shared_data/ against scripts/data_manifest.sha256; fails loudly on any change
#   bash scripts/narval_setup.sh stage        # copy ONLY this project's datasets from shared_data/ to scratch, then verify the copy
#   bash scripts/narval_setup.sh submodules   # git submodule update --init --recursive (+ ticl excludes)
#   bash scripts/narval_setup.sh env [main|bench|latest]   # build/update the virtualenv from environment*.yml
#   bash scripts/narval_setup.sh install [main|bench|latest]   # only pip install -e . --no-deps
#   bash scripts/narval_setup.sh verify       # layout, quotas, imports
#   bash scripts/narval_setup.sh weights [main|bench|latest] [models]   # pre-download checkpoints (compute nodes are offline)
#
# The data master is shared_data/ next to the projects, and other projects write to it (additive_neurostim moved its datasets
# there on 2026-10-02), so nothing here trusts it blindly: scripts/data_manifest.sha256 lists the sha256 of every file this
# project loads (NHP, rat, the five 5d_rat noOutliers animals), `check-data` fails if any differs, and `stage` copies only
# those datasets and re-verifies the copy. Spinal joined the manifest on 2026-10-04: the shared_data/ copy and the local one
# have identical sha256 for all 11 .pkl files (the 1.6 GB vs 3.9 GB of the plan was `du` disk usage against file size).
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
MANIFEST="${MANIFEST:-${CODE_DIR}/scripts/data_manifest.sha256}"

log() { printf '[narval_setup] %s\n' "$*"; }

env_name() {   # env_name <main|bench|latest>
  case "${1:-main}" in
    main) echo "pfns4neurostim" ;;
    bench) echo "pfns4neurostim-bench" ;;
    latest) echo "pfns4neurostim-latest" ;;
    *) log "usage: env [main|bench|latest]"; exit 2 ;;
  esac
}

env_file() {
  case "${1:-main}" in
    main) echo "environment.yml" ;;
    bench) echo "environment.bench.yml" ;;
    latest) echo "environment.latest.yml" ;;
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

verify_manifest() {   # verify_manifest <data root>: every manifest file must exist there with the recorded sha256
  [ -f "${MANIFEST}" ] || { log "ERROR: ${MANIFEST} not found."; exit 1; }
  if (cd "$1" && sha256sum -c --quiet "${MANIFEST}" > /dev/null 2>&1); then
    log "data OK: $(wc -l < "${MANIFEST}") files in $1 match scripts/data_manifest.sha256"
  else
    log "ERROR: data in $1 differs from scripts/data_manifest.sha256. Files that do not match:"
    (cd "$1" && sha256sum -c "${MANIFEST}" 2>&1 | grep -v ': OK$' || true) | sed 's/^/  /'
    log "Another project may have changed them. Do not stage or run until this is understood."
    exit 1
  fi
}

cmd_check_data() { verify_manifest "${DATA_MASTER}"; }

cmd_stage() {
  # Copy only the datasets this project loads, never the whole of shared_data/ (it also holds other projects' data), then verify.
  verify_manifest "${DATA_MASTER}"
  local dirs; dirs=$(awk -F'[ *]+' '{print $NF}' "${MANIFEST}" | xargs -n1 dirname | sort -u)
  local d
  for d in ${dirs}; do
    mkdir -p "${SCRATCH_ROOT}/data/${d}"
    log "staging ${d}"
    rsync -a "${DATA_MASTER}/${d}/" "${SCRATCH_ROOT}/data/${d}/"
  done
  verify_manifest "${SCRATCH_ROOT}/data"
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

# Narval-only substitutions for exact pins the Alliance wheelhouse does not carry as compiled wheels ("pin substitute").
# The environment*.yml files stay the single validated spec for Mila and local; only the Narval build swaps these, and every
# swap is logged. Pure-Python pins missing from the wheelhouse (botorch, jaxtyping, tabpfn) are not listed: pip fetches them
# from PyPI on the login node, as the main env already does. Checked 2026-10-07 with `avail_wheels --python 3.11`: the bench
# env could not be built (2026-10-05) because numpy==2.2.6 is absent; 2.2.2 is the same minor release.
NARVAL_PIN_SUBSTITUTES=(
  "numpy==2.2.6 numpy==2.2.2"
  "matplotlib==3.11.2 matplotlib==3.11.1"
  "statsmodels==0.15.0 statsmodels==0.14.6"
)

apply_pin_substitutes() {   # apply_pin_substitutes <requirements file>
  local req="${1:?requirements file}" pair from to
  for pair in "${NARVAL_PIN_SUBSTITUTES[@]}"; do
    from="${pair%% *}"; to="${pair##* }"
    if grep -qxF "${from}" "${req}"; then
      awk -v f="${from}" -v t="${to}" '$0 == f {print t; next} {print}' "${req}" > "${req}.sub" && mv "${req}.sub" "${req}"
      log "pin substitute (not in the Alliance wheelhouse): ${from} -> ${to}"
    fi
  done
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
  apply_pin_substitutes "${req}"
  log "installing $(grep -cv '^--' "${req}") pinned packages from ${file}"
  # Alliance's wheelhouse is searched first via its pip config; pins it does not carry come from PyPI / the torch
  # index on the login node. A pin that resolves to neither fails here, loudly, not inside a job.
  # pfns4bo 0.1.5 declares scikit-learn<1.2 while the stack pins 1.6.1 (CLAUDE.md section 3). The conda environments on Mila and
  # locally carry exactly this conflict (`pip check` reports it) and PFNs4BO runs on them, but pip's resolver refuses it. So the
  # package goes in last with --no-deps, together with the two packages it would pull in at the versions those environments have.
  local nodeps; nodeps=$(grep -E '^pfns4bo==' "${req}" || true)
  if [ -n "${nodeps}" ]; then grep -vE '^pfns4bo==' "${req}" > "${req}.main" && mv "${req}.main" "${req}"; fi
  pip install -r "${req}"
  if [ -n "${nodeps}" ]; then
    pip install --no-deps "${nodeps}" "bayesmark==0.0.8" "configspace==1.2.2"
    pip install pyparsing more_itertools typing_extensions requests
  fi
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
    latest) models="${2:-tabpfn_v3_5,causilo}" ;;
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
  for which in main bench latest; do
    env="$(env_name "${which}")"; py="$(cluster_env_python "${env}")"
    if [ -z "${py}" ]; then printf '  %-22s not built\n' "${env}"; continue; fi
    printf '  %-22s ' "${env}"
    "${py}" -c "import pfns4neurostim, torch; print('ok, torch', torch.__version__)" 2>&1 | tail -1
  done
  log "next: bash scripts/preflight.sh"
}

case "${1:-}" in
  layout) cmd_layout ;;
  check-data) cmd_check_data ;;
  stage) cmd_stage ;;
  submodules) cmd_submodules ;;
  env) cmd_env "${2:-main}" ;;
  install) cmd_install "${2:-main}" ;;
  verify) cmd_verify ;;
  weights) cmd_weights "${2:-main}" "${3:-}" ;;
  *) sed -n 2,28p "$0"; exit 2 ;;
esac
