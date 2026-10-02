#!/bin/bash
# Local queue for the GP / random halves that were cancelled on Mila's `main-cpu` (2026-10-01). CPU only: the GP models carry
# their own `device: cpu` pin, so these cells share the cache identity of the cluster's CPU cells and of the GPU halves.
#
#   nohup bash scripts/local_queue_gp.sh bo    > output/logs/local_queue_gp_bo.log    2>&1 &   # benchmarks, GP on CPU
#   nohup bash scripts/local_queue_gp.sh knobs > output/logs/local_queue_gp_knobs.log 2>&1 &   # K-knob sweeps, GP on CUDA
#
# The two groups use different resources and run side by side. The knob group overrides the config's `model_params.<gp>.device: cpu`
# pin with cuda (device is part of the cell identity, so those cells are separate from CPU-run GP cells; latency is not reported).
# Cheapest first (portfolio.py CPU estimates on 8 lanes). Phases run sequentially so the lanes never oversubscribe the cores; a
# failed phase never blocks the next; every phase ends by assembling its unit from the cache (the PFN cells only appear in that
# assembly once the cluster's output/cells has been exported here).
# Env: PY (python of the main env), LANES (default 6: leaves cores free for a concurrent local GPU job).
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
GROUP="${1:?usage: local_queue_gp.sh <bo|knobs>}"
LANES="${LANES:-6}"
GPU_LANES="${GPU_LANES:-3}"   # ~3 processes share one GPU for the best aggregate throughput

stamp() { echo "[queue-gp] $1: $(date)"; }
assemble() { "${PY}" -m pfns4neurostim "$1" --config "$2" --only-cached > "output/logs/assemble_$(basename "$2" .yaml).log" 2>&1 || echo "[queue-gp] ASSEMBLY FAILED for $2 (see output/logs)"; }

# phase <label> <experiment> <config> <models>
phase() {
  stamp "START $1"
  PY="${PY}" LABEL="qgp-$1" bash scripts/run_local_lanes.sh "$2" "$3" "$4" "${LANES}" "${@:5}" || echo "[queue-gp] a lane of $1 reported failures"
  assemble "$2" "$3"
  stamp "DONE $1"
}

mkdir -p output/logs
if [ "${GROUP}" = "bo" ]; then
  phase pfn-bench-nhp   bo_benchmark configs/experiment/hyp0_pfn_bench_nhp.yaml       gp_mll
  phase hyp-a-nhp       bo_benchmark configs/experiment/hyp_a_nhp.yaml                gp_mll,gp_naive,random
  phase hyp-a-5d        bo_benchmark configs/experiment/hyp_a_5d_rat.yaml             gp_mll,gp_naive,random
  phase pfn-bench-5d    bo_benchmark configs/experiment/hyp0_pfn_bench_5d_rat.yaml    gp_mll
  phase acq-core-5d     bo_benchmark configs/experiment/hyp0_acq_core_5d_rat.yaml     gp_mll,gp_naive,random
else
  LANES="${GPU_LANES}"
  GPU_GP=("model_params.gp_mll.device=cuda" "model_params.gp_naive.device=cuda")
  phase k7-nhp          stress_sweep configs/experiment/stress_k7_nhp.yaml            gp_mll,gp_naive,random "${GPU_GP[@]}"
  phase k6-failure-5d   stress_sweep configs/experiment/stress_k6_failure_5d_rat.yaml gp_mll,gp_naive "${GPU_GP[@]}"
  phase k5-5d           stress_sweep configs/experiment/stress_k5_5d_rat.yaml         gp_mll,gp_naive "${GPU_GP[@]}"
  phase k2-global-5d    stress_sweep configs/experiment/stress_k2_global_5d_rat.yaml  gp_mll,gp_naive "${GPU_GP[@]}"
fi
stamp "QUEUE FINISHED ($GROUP)"
