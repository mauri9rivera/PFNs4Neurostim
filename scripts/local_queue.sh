#!/bin/bash
# Local extra-work queue (run unattended). Phases run one after another so the single GPU is not oversubscribed; GP-only work
# overlaps the GPU work on the CPU; a failed phase never blocks the next; every phase ends by assembling its unit from the cache.
#
#   nohup bash scripts/local_queue.sh > output/logs/local_queue.log 2>&1 &
#
# Priority order (TabFM, the slowest PFN, is deliberately LAST). Env: PY (main env python), PYB (bench env python).
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
PYB="${PYB:-python}"
BENCH_CFG=configs/experiment/hyp0_pfn_bench_nhp.yaml

stamp() { echo "[queue] $1: $(date)"; }
assemble() { local py="$1" exp="$2" cfg="$3"; "${py}" -m pfns4neurostim "${exp}" --config "${cfg}" --only-cached > "output/logs/assemble_$(basename "${cfg}" .yaml).log" 2>&1 || echo "[queue] ASSEMBLY FAILED for ${cfg} (see output/logs)"; }

# phase <label> <py> <exp> <cfg> <gpu-models|-> <gpu-lanes> <cpu-models|-> <cpu-lanes>
phase() {
  local label="$1" py="$2" exp="$3" cfg="$4" gm="$5" gl="$6" cm="$7" cl="$8" pids=()
  stamp "START ${label}"
  if [ "${gm}" != "-" ]; then PY="${py}" LABEL="q-${label}" bash scripts/run_local_lanes.sh "${exp}" "${cfg}" "${gm}" "${gl}" & pids+=($!); fi
  if [ "${cm}" != "-" ]; then PY="${py}" LABEL="q-${label}" bash scripts/run_local_lanes.sh "${exp}" "${cfg}" "${cm}" "${cl}" & pids+=($!); fi
  for p in "${pids[@]}"; do wait "${p}" || echo "[queue] a lane group of ${label} reported failures"; done
  assemble "${py}" "${exp}" "${cfg}"
  stamp "DONE ${label}"
}

# Wait for the K6-budget chain (D1 assembly + K6 lanes) started earlier.
until grep -q "K6 assembled" output/logs/local_chain.log 2>/dev/null; do sleep 30; done
stamp "K6-budget chain finished; starting the queue"

phase d3-tabicl    "${PYB}" bo_benchmark  "${BENCH_CFG}"                                   tabicl      3  -                  0
phase acq-core     "${PY}"  bo_benchmark  configs/experiment/hyp0_acq_core_nhp.yaml        tabpfn_v2_5 3  gp_mll,gp_naive,random 4
phase k6-dropout   "${PY}"  stress_sweep  configs/experiment/stress_k6_dropout_nhp.yaml    tabpfn_v2_5 3  gp_mll,gp_naive    3
phase ucb-kappa    "${PY}"  bo_benchmark  configs/experiment/hyp0_ucb_kappa_nhp.yaml       tabpfn_v2_5 3  -                  0
phase k5-nhp-synth "${PY}"  stress_sweep  configs/experiment/stress_k5_nhp.yaml            tabpfn_v2_5 3  gp_mll,gp_naive    3
phase d3-tabfm     "${PYB}" bo_benchmark  "${BENCH_CFG}"                                   tabfm       2  -                  0
# The D3 unit also pulls TabPFN and GP-MLL from the D1 cache: assemble once more with everything present.
assemble "${PYB}" bo_benchmark "${BENCH_CFG}"
stamp "QUEUE FINISHED"
