#!/bin/bash
# Local counterpart of the Mila lane runner: N sharded processes of one experiment on this machine.
#
#   bash scripts/run_local_lanes.sh <bo_benchmark|stress_sweep> <config.yaml> <models,comma,separated> <lanes> [key=value ...]
#
# Each lane owns every N-th channel (--shard i/N), runs single-threaded, and writes to the shared cell cache, so lanes,
# and separate invocations (e.g. a GPU-model call and a CPU-model call), never conflict. Afterwards assemble the union with
#   python -m pfns4neurostim <experiment> --config <config.yaml> --only-cached
# Env: PY = python interpreter (default: python), LABEL = run-tag prefix (default: local).
set -euo pipefail

EXPERIMENT="${1:?usage: run_local_lanes.sh <experiment> <config> <models> <lanes> [key=value ...]}"
CONFIG="${2:?config}"
MODELS="${3:?models}"
LANES="${4:?lanes}"
shift 4 || true

PY="${PY:-python}"
LABEL="${LABEL:-local}"
mkdir -p output/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
pids=()
for ((i = 0; i < LANES; i++)); do
  "${PY}" -u -m pfns4neurostim "${EXPERIMENT}" --config "${CONFIG}" --set "models=[${MODELS}]" "$@" --shard "${i}/${LANES}" > "output/logs/${LABEL}_${MODELS//,/-}_lane${i}.log" 2>&1 &
  pids+=($!)
done
echo "[run_local_lanes] ${LANES} lanes of ${EXPERIMENT} (${MODELS}); logs: output/logs/${LABEL}_${MODELS//,/-}_lane*.log"
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
exit "${failed}"
