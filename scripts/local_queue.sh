#!/bin/bash
# Local queue (run unattended on the RTX 3060). Phases run one after another so the single GPU is not oversubscribed; GP-only work
# overlaps the GPU work on the CPU; a failed phase never blocks the next; every phase ends by assembling its unit from the cache.
#
#   nohup bash scripts/local_queue.sh > output/logs/local_queue.log 2>&1 &
#
# Re-written 2026-09-21 for the meeting deadline, after the re-query fix (cache_version 2). Priority order, all at the config's
# default 10 reps. Deliberately NOT here: D3 NHP (the cluster runs it: TabPFN-2.5, TabICL, GP-MLL) and TabFM (3 h 17 min for 10 reps).
# A phase that has not finished when you need the numbers can still be assembled from whatever cells exist:
#   python -m pfns4neurostim <experiment> --config <config.yaml> --only-cached
# Env: PY (python of the main env).
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"

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

mkdir -p output/logs
# 1. Deliverable 1, NHP (not in the cluster portfolio). ~40 min at 10 reps.
phase d1-nhp       "${PY}" bo_benchmark  configs/experiment/hyp_a_nhp.yaml                 tabpfn_v2_5 3  gp_mll,gp_naive,random 4
# 2. Acquisition table (ei, ucb, ts_marginal, random) - the input to the headline-acquisition decision. ~50 min.
phase acq-core     "${PY}" bo_benchmark  configs/experiment/hyp0_acq_core_nhp.yaml         tabpfn_v2_5 3  gp_mll,gp_naive,random 4
# 3. K6 electrode dropout, 6 levels at budget 96 (the heaviest local phase: ~3 h for TabPFN alone).
phase k6-dropout   "${PY}" stress_sweep  configs/experiment/stress_k6_dropout_nhp.yaml    tabpfn_v2_5 3  gp_mll,gp_naive        3
# 4. UCB fixed-kappa grid (TabPFN only).
phase ucb-kappa    "${PY}" bo_benchmark  configs/experiment/hyp0_ucb_kappa_nhp.yaml       tabpfn_v2_5 3  -                      0
# 5. K5 outliers on NHP (synthetic heavy-tail: caption it as such).
phase k5-nhp-synth "${PY}" stress_sweep  configs/experiment/stress_k5_nhp.yaml            tabpfn_v2_5 3  gp_mll,gp_naive        3
stamp "QUEUE FINISHED"
