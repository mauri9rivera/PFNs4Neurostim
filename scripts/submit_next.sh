#!/bin/bash
# Submit the remaining portfolio in stages, each gated on its prerequisite (login node, repo root).
#
#   bash scripts/submit_next.sh status        # what is submitted, what is ready, what is still blocked
#   bash scripts/submit_next.sh now           # everything with no open prerequisite
#   bash scripts/submit_next.sh v1-test       # ONE TabPFN v1 cell (needs the pfns4neurostim-v1 env)
#   bash scripts/submit_next.sh v1            # E7, E8, P16 (needs a finished v1 cell)
#   bash scripts/submit_next.sh tabfm-spinal  # P15 (needs E3's cells from the fixed TabFM wrapper)
#
# Units come from the generated scripts/submit_*.sh (one `submit_unit "<ID>. ..."` line each); this script only
# selects lines, so scripts/portfolio.py stays the single source. Each stage records itself in
# logs/submitted_<stage>.txt and refuses to run twice (add --force to override). Held back everywhere:
# S5b, until the Demo 1 collapse check is fixed (task plan #10 Step 15).
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:-status}"
FORCE="${2:-}"
CELLS="output/cells"
V1_ENV="pfns4neurostim-v1"

# Run the unit lines of a generated submit script whose ID matches (keep) or does not match (drop) a regex.
run_units() {
  local script="$1" mode="$2" regex="$3" tmp
  tmp=$(mktemp)
  awk -v mode="${mode}" -v re="^(${regex})$" '
    /^submit_(unit|single) "/ {
      id = $0; sub(/^submit_(unit|single) "/, "", id); sub(/\..*/, "", id)
      hit = (id ~ re)
      if ((mode == "keep") != hit) next
      print "echo \"[submit_next] " id "\"" }
    { print }' "${script}" > "${tmp}"
  bash "${tmp}"
  rm -f "${tmp}"
}

has_cell() {   # has_cell <dataset> <model>: a cached bo_benchmark cell of that model exists
  grep -rlq "\"model\": \"$2\"" "${CELLS}/$1/bo_benchmark" 2>/dev/null
}
env_exists() { conda env list 2>/dev/null | grep -qE "^${V1_ENV}\s"; }
marker() { echo "logs/submitted_$1.txt"; }
guard() {   # guard <stage>: refuse a second submission of the same stage
  if [ -f "$(marker "$1")" ] && [ "${FORCE}" != "--force" ]; then
    echo "[submit_next] stage '$1' was already submitted ($(head -1 "$(marker "$1")")); add --force to resubmit" >&2
    exit 1
  fi
}
record() { mkdir -p logs; date "+%F %T" > "$(marker "$1")"; }
if ! command -v conda >/dev/null 2>&1; then set +u; module load anaconda/3; set -u; fi

case "${STAGE}" in
  status)
    for s in now v1-test v1 tabfm-spinal; do
      if [ -f "$(marker "$s")" ]; then echo "  ${s}: submitted $(head -1 "$(marker "$s")")"; continue; fi
      case "$s" in
        now) echo "  now: READY" ;;
        v1-test) env_exists && echo "  v1-test: READY" || echo "  v1-test: waiting for env ${V1_ENV} (sbatch scripts/setup_env_job.sh v1)" ;;
        v1) has_cell nhp tabpfn_v1 && echo "  v1: READY" || echo "  v1: waiting for a finished v1 cell (stage v1-test)" ;;
        tabfm-spinal) has_cell nhp tabfm && echo "  tabfm-spinal: READY" || echo "  tabfm-spinal: waiting for E3 (TabFM NHP) cells" ;;
      esac
    done
    echo "  held back: S5b (task plan #10 Step 15)"
    ;;
  now)
    guard now
    [ -d data/spinal ] || { echo "[submit_next] data/spinal missing on the cluster" >&2; exit 1; }
    run_units scripts/submit_portfolio.sh drop 'S5b'          # 5d_rat stress S1b-S4b
    run_units scripts/submit_bench.sh drop '__none__'         # 5d_rat B0-B2
    run_units scripts/submit_externals.sh drop 'E7|E8'        # E1, E3, E4, E6
    run_units scripts/submit_spinal.sh drop 'P15|P16'         # spinal P1-P14
    record now
    ;;
  v1-test)
    guard v1-test
    env_exists || { echo "[submit_next] env ${V1_ENV} not built yet: sbatch scripts/setup_env_job.sh v1" >&2; exit 1; }
    # One cell of E7's grid (same identity), so E7 later reuses it as a cache hit.
    CONDA_ENV="${V1_ENV}" sbatch scripts/run_bo_benchmark.sh configs/experiment/hyp0_pfn_bench_nhp.yaml "models=[tabpfn_v1]" "dataset.subjects=[1]" "dataset.emgs=[0]" n_reps=1 tag=v1-smoke
    record v1-test
    echo "[submit_next] when it finishes: bash scripts/submit_next.sh status   (v1 turns READY if the cell succeeded)"
    ;;
  v1)
    guard v1
    has_cell nhp tabpfn_v1 || { echo "[submit_next] no finished TabPFN v1 cell yet: run stage v1-test and read its log" >&2; exit 1; }
    run_units scripts/submit_externals.sh keep 'E7|E8'
    run_units scripts/submit_spinal.sh keep 'P16'
    record v1
    ;;
  tabfm-spinal)
    guard tabfm-spinal
    has_cell nhp tabfm || { echo "[submit_next] no TabFM cell from E3 yet" >&2; exit 1; }
    run_units scripts/submit_spinal.sh keep 'P15'
    record tabfm-spinal
    ;;
  *)
    echo "usage: bash scripts/submit_next.sh [status|now|v1-test|v1|tabfm-spinal] [--force]" >&2; exit 2 ;;
esac
