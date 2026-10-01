#!/bin/bash
# Submit the portfolio one stage at a time, each stage individually gated (login node, repo root).
#
#   bash scripts/submit_next.sh status      # rank, cost, gate and submission state of every stage
#   bash scripts/submit_next.sh audit       # 1. y-scaling arms + the invariance check   (2.7 CPU-h)
#   bash scripts/submit_next.sh nhp         # 2. every NHP deliverable, canonical arm    (~52 GPU-h)
#   bash scripts/submit_next.sh rat         # 3. 5d_rat stress + bench                   (~45 GPU-h)
#   bash scripts/submit_next.sh hypc        # 4. Hyp C context sweeps C1-C3              (~13 GPU-h)
#   bash scripts/submit_next.sh externals   # 5. PFN benchmark, non-v1 models            (~9 GPU-h)
#   bash scripts/submit_next.sh spinal --force   # 6. spinal P1-P14 (DEFERRED: needs --force)  (~141 GPU-h)
#
# `spinal` is DEFERRED and needs --force even on its first submission. It is 54 % of the remaining GPU
# time and 59 % of the CPU time for a *supporting* dataset, its data is staged so its gate passes, and the
# converged `gp_mll` (P0.10) tripled its CPU column -- seven of its units now exceed the 12 h wall and will
# requeue. Decide it deliberately, and consider n_reps=10 instead of 20, which halves it.
#
# Gated sub-stages, waiting on a specific finished thing rather than on a rank:
#   v1-test -> v1     one TabPFN v1 cell, then E7/E8/P16   (needs the v1 env, then that cell)
#   tabfm-spinal      P15                                   (needs E3's TabFM cells)
#   rat-demo1         S5b                                   (needs the Demo 1 collapse check, #10 Step 15)
#   hypc-5drat        C4                                     (needs one hand-timed 5d_rat channel, #18 Step 5)
#
# The ranking is priority, not order: every stage is independent of the ones below it, so submit them in
# whatever order suits you. `bash scripts/preflight.sh <stage>` checks a stage's prerequisites without
# submitting anything; run it first. Stage 0 needs no cluster at all: `bash scripts/run_local.sh`.
#
# Units come from the generated scripts/submit_<group>.sh (one `submit_unit`/`submit_single` line each); this
# script only selects lines, so scripts/portfolio.py stays the single source of every config, override,
# environment and resource figure. Each stage records itself in logs/submitted_<stage>.txt and refuses a
# second run without --force.
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:-status}"
FORCE="${2:-}"
CELLS="output/cells"
V1_ENV="pfns4neurostim-v1"

# Run the unit lines of a generated submit script whose ID matches (keep) or does not match (drop) a regex.
run_units() {
  local script="$1" mode="$2" regex="$3" tmp
  [ -f "${script}" ] || { echo "[submit_next] ${script} is missing: regenerate it with scripts/portfolio.py" >&2; exit 1; }
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
env_exists() { conda env list 2>/dev/null | grep -qE "^$1[[:space:]]"; }
have_data() { [ -d "data/$1" ]; }
marker() { echo "logs/submitted_$1.txt"; }
guard() {   # guard <stage>: refuse a second submission of the same stage
  if [ -f "$(marker "$1")" ] && [ "${FORCE}" != "--force" ]; then
    echo "[submit_next] stage '$1' was already submitted ($(head -1 "$(marker "$1")")); add --force to resubmit" >&2
    exit 1
  fi
}
record() { mkdir -p logs; date "+%F %T" > "$(marker "$1")"; }
need_data() {   # need_data <subdir> <hint>
  have_data "$1" || { echo "[submit_next] blocked: data/$1 is missing — $2" >&2; exit 1; }
}
if ! command -v conda >/dev/null 2>&1; then set +u; module load anaconda/3 >/dev/null 2>&1 || true; set -u; fi

# gate_<stage> prints READY, or the reason it is not.
gate_audit()     { echo "READY"; }
gate_nhp()       { have_data monkeys && echo "READY" || echo "needs data/monkeys (bash scripts/mila_setup.sh stage)"; }
gate_rat()       { have_data 5d_rat && echo "READY" || echo "needs data/5d_rat staged"; }
gate_hypc()      { have_data monkeys && echo "READY" || echo "needs data/monkeys"; }
gate_externals() { env_exists pfns4neurostim-bench && echo "READY" || echo "needs the bench env (bash scripts/mila_setup.sh env bench)"; }
gate_spinal()    {   # deferred by decision, not by a missing prerequisite
  have_data spinal || { echo "needs data/spinal staged (#14 Step 3)"; return; }
  echo "DEFERRED - ready, but needs --force (54% of the remaining compute)"
}
gate_v1_test()   { env_exists "${V1_ENV}" && echo "READY" || echo "needs the v1 env (sbatch scripts/setup_env_job.sh v1)"; }
gate_v1()        { has_cell nhp tabpfn_v1 && echo "READY" || echo "needs a finished v1 cell (stage v1-test)"; }
gate_tabfm()     { has_cell nhp tabfm && echo "READY" || echo "needs E3's TabFM cells (stage externals)"; }

case "${STAGE}" in
  status)
    printf '%-14s %-5s %-11s %s\n' STAGE RANK COST STATE
    while read -r name rank cost gate; do
      if [ -f "$(marker "${name}")" ]; then
        state="submitted $(head -1 "$(marker "${name}")")"
      else
        state=$("${gate}")
      fi
      printf '%-14s %-5s %-11s %s\n' "${name}" "${rank}" "${cost}" "${state}"
    done <<'STAGES'
audit 1 6.6cpu-h gate_audit
nhp 2 58gpu-h gate_nhp
rat 3 45gpu-h gate_rat
hypc 4 9.5gpu-h gate_hypc
externals 5 9gpu-h gate_externals
spinal 6 141gpu-h gate_spinal
v1-test - 1cell gate_v1_test
v1 - ?gpu-h gate_v1
tabfm-spinal - ?gpu-h gate_tabfm
STAGES
    echo
    echo "COST is the GPU column (CPU-only for audit); the CPU column roughly TRIPLED with the converged"
    echo "gp_mll of P0.10 -- audit 6.6, nhp 36, rat 31, spinal 110 CPU-h. Full table: python scripts/portfolio.py"
    echo
    echo "deferred by decision:       spinal (needs --force; 54% of the remaining compute)"
    echo "held back by a code task:   rat-demo1 (S5b, #10 Step 15), hypc-5drat (C4, #18 Step 5)"
    echo "run locally, not here:      hypc (C1/C2/C3) -- bash scripts/run_local.sh hypc-gpu | hypc-cpu"
    echo "stage 0 needs no cluster:   bash scripts/run_local.sh"
    echo "a stage's gates in detail:  bash scripts/preflight.sh <stage>"
    ;;

  # ---- 1. the cheapest gating work: what the y-scaling decision rests on ----
  audit)
    guard audit
    run_units scripts/submit_audit.sh drop '__none__'
    record audit
    ;;

  # ---- 2. every NHP deliverable under the canonical arm (the flip made the old cells unaddressable) ----
  nhp)
    guard nhp
    need_data monkeys "bash scripts/mila_setup.sh stage"
    run_units scripts/submit_nhp.sh drop '__none__'
    record nhp
    ;;

  # ---- 3. the second dataset, which is what makes a cross-dataset claim possible ----
  rat)
    guard rat
    need_data 5d_rat "bash scripts/mila_setup.sh stage"
    run_units scripts/submit_portfolio.sh drop 'S5b'      # 5d_rat stress S1b-S4b (S5b held back)
    run_units scripts/submit_bench.sh drop '__none__'     # 5d_rat B0-B2
    record rat
    ;;
  rat-demo1)
    guard rat-demo1
    if [ "${FORCE}" != "--force" ]; then
      echo "[submit_next] S5b needs the Demo 1 collapse check first (#10 Step 15); confirm it, then --force" >&2
      exit 1
    fi
    run_units scripts/submit_portfolio.sh keep 'S5b'
    record rat-demo1
    ;;

  # ---- 4. Hyp C with the context sweeps (C2 is the long, non-resumable one) ----
  hypc)
    guard hypc
    need_data monkeys "bash scripts/mila_setup.sh stage"
    run_units scripts/submit_hypc.sh drop 'C4'
    record hypc
    ;;
  hypc-5drat)
    guard hypc-5drat
    if [ "${FORCE}" != "--force" ]; then
      echo "[submit_next] C4's cost is unmeasured: time ONE 5d_rat placement channel first (#18 Step 5), then --force" >&2
      exit 1
    fi
    run_units scripts/submit_hypc.sh keep 'C4'
    record hypc-5drat
    ;;

  # ---- 5. the PFN benchmark, minus the v1 arm ----
  externals)
    guard externals
    run_units scripts/submit_externals.sh drop 'E7|E8'
    record externals
    ;;

  # ---- 6. the supporting dataset, and by far the most expensive stage ----
  spinal)
    guard spinal
    need_data spinal "#14 Step 3 stages it from your machine"
    if [ "${FORCE}" != "--force" ]; then
      echo "[submit_next] spinal is DEFERRED: 14 units, ~141 GPU-h + ~110 CPU-h, for the supporting" >&2
      echo "[submit_next] dataset -- 54% of everything left. Seven units exceed the 12 h wall and will" >&2
      echo "[submit_next] requeue. Submit the NHP and 5d_rat stages first, consider n_reps=10 (halves it)," >&2
      echo "[submit_next] then: bash scripts/submit_next.sh spinal --force" >&2
      exit 1
    fi
    run_units scripts/submit_spinal.sh drop 'P15|P16'
    record spinal
    ;;

  # ---- gated sub-stages ----
  v1-test)
    guard v1-test
    env_exists "${V1_ENV}" || { echo "[submit_next] env ${V1_ENV} not built: sbatch scripts/setup_env_job.sh v1" >&2; exit 1; }
    # One cell of E7's grid (same identity), so E7 later reuses it as a cache hit rather than recomputing it.
    CONDA_ENV="${V1_ENV}" sbatch scripts/run_bo_benchmark.sh configs/experiment/hyp0_pfn_bench_nhp.yaml "models=[tabpfn_v1]" "dataset.subjects=[1]" "dataset.emgs=[0]" n_reps=1 tag=v1-smoke
    record v1-test
    echo "[submit_next] when it finishes: bash scripts/submit_next.sh status   (v1 turns READY if the cell succeeded)"
    ;;
  v1)
    guard v1
    has_cell nhp tabpfn_v1 || { echo "[submit_next] no finished TabPFN v1 cell yet: run stage v1-test and read its log" >&2; exit 1; }
    run_units scripts/submit_externals.sh keep 'E7|E8'
    if have_data spinal; then
      run_units scripts/submit_spinal.sh keep 'P16'
    else
      echo "[submit_next] P16 skipped: data/spinal is not staged"
    fi
    record v1
    ;;
  tabfm-spinal)
    guard tabfm-spinal
    has_cell nhp tabfm || { echo "[submit_next] no TabFM cell from E3 yet (stage externals)" >&2; exit 1; }
    need_data spinal "#14 Step 3 stages it from your machine"
    run_units scripts/submit_spinal.sh keep 'P15'
    record tabfm-spinal
    ;;

  *)
    echo "usage: bash scripts/submit_next.sh [status|audit|nhp|rat|hypc|externals|spinal|v1-test|v1|tabfm-spinal|rat-demo1|hypc-5drat] [--force]" >&2
    exit 2 ;;
esac
