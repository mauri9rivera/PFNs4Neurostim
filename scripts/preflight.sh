#!/bin/bash
# Check every prerequisite a deployment stage depends on, and print a ready/blocked table. Submits NOTHING.
#
#   bash scripts/preflight.sh              # the whole table
#   bash scripts/preflight.sh nhp          # just that stage; exit 1 when it is not ready
#
# Why this exists: three things have failed at the *first cell of a queued job* rather than at submission —
# a wrong conda environment (#8 rule 3), a dataset that was never staged, and a job asking for less memory
# than its model needs (nine units, found 2026-09-30). All three are answerable in seconds on a login node.
# `require_models` now covers the environment case inside the runners; this script covers the rest, and
# checks them per stage so a blocked stage never hides a ready one.
#
# Read-only by construction: it runs `git`, `ls`, `conda env list`, `df` and one in-process import check.
set -uo pipefail
cd "$(dirname "$0")/.."

STAGE_FILTER="${1:-all}"
PASS=0
FAIL=0

# shellcheck disable=SC1091
source scripts/cluster.sh          # CLUSTER, cluster_env_python, cluster_setup_script
SETUP="bash $(cluster_setup_script)"

say() {   # say <ok|no|note> <what> <detail>
  case "$1" in
    ok)   printf '  [ ok ] %-34s %s\n' "$2" "$3"; PASS=$((PASS + 1)) ;;
    no)   printf '  [FAIL] %-34s %s\n' "$2" "$3"; FAIL=$((FAIL + 1)) ;;
    note) printf '  [note] %-34s %s\n' "$2" "$3" ;;
  esac
}

# ---------------------------------------------------------------- repository
check_repo() {
  echo "repository"
  local branch head upstream dirty
  branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
  head=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
  # --ignore-submodules=dirty: libs/ submodules carry generated __pycache__, which is not a reason to
  # block a stage. A submodule POINTER change still shows, which is the thing that would matter.
  dirty=$(git status --porcelain --untracked-files=no --ignore-submodules=dirty 2>/dev/null | wc -l | tr -d ' ')
  say note "branch / HEAD" "${branch} @ ${head}"
  if [ "${dirty}" = "0" ]; then
    say ok "working tree" "clean (tracked files)"
  else
    say no "working tree" "${dirty} modified tracked file(s) — the cluster runs what is PUSHED, not what is here"
  fi
  upstream=$(git rev-parse --abbrev-ref '@{upstream}' 2>/dev/null || true)
  if [ -z "${upstream}" ]; then
    say note "upstream" "none configured; cannot tell whether this commit is on the remote"
  elif [ "$(git rev-parse HEAD)" = "$(git rev-parse "${upstream}" 2>/dev/null)" ]; then
    say ok "synced with ${upstream}" "same commit"
  else
    say no "synced with ${upstream}" "HEAD differs — run a git pull on the login node, or push from your machine"
  fi
}

# ---------------------------------------------------------------- data
check_data() {   # check_data <dataset> <subdir>
  local name="$1" subdir="$2" size
  if [ -d "data/${subdir}" ]; then
    size=$(du -sh "data/${subdir}" 2>/dev/null | cut -f1)
    say ok "data/${subdir} (${name})" "present, ${size}"
  else
    say no "data/${subdir} (${name})" "missing — ${SETUP} stage"
  fi
}

# ---------------------------------------------------------------- environments
check_env() {   # check_env <env name> <models to import, comma separated>
  local env="$1" models="$2" py out which
  which="${env##*-}"; [ "${env}" = "pfns4neurostim" ] && which=main
  # The env's interpreter is called DIRECTLY (conda env on Mila, virtualenv on Narval), never through `conda run`:
  # in a non-interactive login shell conda run returns no output at all, which read as a blocked environment on a
  # perfectly good one (2026-09-30). The same lesson is recorded for clean_mila_results.sh.
  py=$(cluster_env_python "${env}")
  if [ -z "${py}" ]; then
    say no "env ${env}" "not built — ${SETUP} env ${which}"
    return
  fi
  out=$("${py}" -c "
import sys
from pfns4neurostim.models.registry import require_models
try:
    require_models('${models}'.split(','))
except Exception as exc:
    print(str(exc).replace(chr(10), ' '))
    sys.exit(1)
print('ok')
" 2>&1 | tail -1)
  if [ "${out}" = "ok" ]; then
    say ok "env ${env}" "models importable: ${models}"
  else
    say no "env ${env}" "${out:-no output from ${py}}"
  fi
}

# ---------------------------------------------------------------- misc
check_submodules() {
  echo "submodules (Hyp C reference banks)"
  local missing=0
  for path in libs/tabpfn-v1-prior libs/PFNs4BO; do
    if [ -d "${path}" ] && [ -n "$(ls -A "${path}" 2>/dev/null)" ]; then
      say ok "${path}" "initialised"
    else
      say no "${path}" "empty — git submodule update --init --recursive"
      missing=1
    fi
  done
  return ${missing}
}

check_space() {
  echo "disk"
  local avail
  avail=$(df -h . 2>/dev/null | awk 'NR==2 {print $4}')
  say note "free on $(pwd | cut -c1-24)..." "${avail:-unknown}"
  if [ -d output/cells ]; then
    say note "output/cells" "$(du -sh output/cells 2>/dev/null | cut -f1) in $(find output/cells -name '*.json' 2>/dev/null | wc -l | tr -d ' ') cells"
  fi
}

# ---------------------------------------------------------------- stages
stage_audit()     { check_repo; }
stage_nhp()       { check_repo; check_data nhp monkeys; }
stage_rat()       { check_repo; check_data 5d_rat 5d_rat; }
stage_hypc()      { check_repo; check_data nhp monkeys; check_submodules || true; }
stage_externals() { check_repo; check_data nhp monkeys; check_data 5d_rat 5d_rat; echo "environments"; check_env pfns4neurostim-bench tabicl,tabfm; }
stage_spinal()    { check_repo; check_data spinal spinal; }
stage_v1()        { echo "environments"; check_env pfns4neurostim-v1 tabpfn_v1; }

case "${STAGE_FILTER}" in
  all)
    check_repo
    echo "data"
    check_data nhp monkeys
    check_data 5d_rat 5d_rat
    check_data spinal spinal
    echo "environments"
    check_env pfns4neurostim tabpfn_v2_5,gp_mll,pfns4bo
    check_env pfns4neurostim-bench tabicl,tabfm
    check_env pfns4neurostim-v1 tabpfn_v1
    check_submodules || true
    check_space
    ;;
  audit|nhp|rat|hypc|externals|spinal|v1) "stage_${STAGE_FILTER}" ;;
  *) echo "usage: bash scripts/preflight.sh [all|audit|nhp|rat|hypc|externals|spinal|v1]" >&2; exit 2 ;;
esac

echo
echo "  ${PASS} ok, ${FAIL} blocking"
if [ "${FAIL}" -gt 0 ]; then
  echo "  a [FAIL] above blocks the stage(s) that need it; every other stage is unaffected."
  exit 1
fi
echo "  next: bash scripts/submit_next.sh status"
