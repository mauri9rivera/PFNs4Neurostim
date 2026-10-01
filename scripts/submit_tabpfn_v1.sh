#!/bin/bash
# Bring TabPFN v1 into the PFN benchmark on all three neurostim datasets, in prerequisite order.
#
# TabPFN v1 is ONE MODEL ROW of hyp0_pfn_bench_{nhp,spinal,5d_rat}.yaml, not an experiment of its own, so
# every stage below submits the ordinary bo_benchmark unit with `models=[tabpfn_v1]` and lets the existing
# assemble step merge the row into the run that already holds the other models.
#
#   bash scripts/submit_tabpfn_v1.sh status      # what is done, ready, or blocked
#   bash scripts/submit_tabpfn_v1.sh env         # build pfns4neurostim-v1 (needs generous memory: see below)
#   bash scripts/submit_tabpfn_v1.sh smoke       # ONE cell, to execute the wrapper for the first time
#   bash scripts/submit_tabpfn_v1.sh nhp         # E7  (run first: it measures the wall time)
#   bash scripts/submit_tabpfn_v1.sh spinal      # P16 (needs data/spinal staged)
#   bash scripts/submit_tabpfn_v1.sh 5d_rat      # E8
#   bash scripts/submit_tabpfn_v1.sh reassemble  # merge v1 into the three PFN-bench tables (no compute)
#
# WHY THE STAGES ARE GATED.
#  1. The env build has already failed once by OOM (job 10936144, 2026-09-25): the cluster runs conda 4.8.3,
#     whose classic SAT solver is memory-hungry, and setup_env_job.sh asks for only 16 G. This script
#     requests ENV_MEM (default 48G) for that job.
#  2. The v1 wrapper has NEVER EXECUTED anywhere (task plan #8 Step 5a): its API calls
#     (TabPFNClassifier(device, N_ensemble_configurations, seed), fit(..., overwrite_warning)) are as
#     documented upstream, not as run. So `smoke` runs exactly one cell and the other stages refuse to
#     start until a finished v1 cell exists in the cache.
#  3. NHP goes before 5d_rat because v1's per-rep cost is unmeasured; read E7's wall time from `sacct`
#     before committing the 5D grid.
#
# Every v1 row is a CLASSIFICATION-HEAD ADAPTATION (guardrail G3): v1 is a classifier and the wrapper reads
# at most MAX_CLASSES=10 quantile bins as a bar distribution, so its calibration numbers are bounded by the
# bin count. Label it as such in every table and caption; see docs/tabpfn_v1_adaptation.md.
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:-status}"
FORCE="${2:-}"
V1_ENV="pfns4neurostim-v1"
ENV_MEM="${ENV_MEM:-48G}"
CELLS="output/cells"
SMOKE_TAG="v1-smoke"

log() { printf '[submit_tabpfn_v1] %s\n' "$*" >&2; }

# Cluster login nodes reach conda through a module; off the cluster there is no `module`, and `status`
# must still work there, so this is best-effort rather than fatal.
if ! command -v conda >/dev/null 2>&1; then
  set +u +e; module load anaconda/3 >/dev/null 2>&1 || true; set -u -e
fi

env_exists()  { conda env list 2>/dev/null | grep -qE "^${V1_ENV}\s"; }
# A finished v1 cell anywhere in the cache means the wrapper has actually run to completion once.
has_v1_cell() { grep -rlq '"model": "tabpfn_v1"' "${CELLS}" 2>/dev/null; }
marker()      { echo "logs/submitted_v1_$1.txt"; }
record()      { mkdir -p logs; date "+%F %T" > "$(marker "$1")"; }
guard() {
  if [ -f "$(marker "$1")" ] && [ "${FORCE}" != "--force" ]; then
    log "stage '$1' was already submitted ($(head -1 "$(marker "$1")")); add --force to resubmit"
    exit 1
  fi
}
need_env() {
  env_exists || { log "env ${V1_ENV} does not exist yet: bash scripts/submit_tabpfn_v1.sh env"; exit 2; }
}
need_smoke() {
  has_v1_cell || { log "no finished TabPFN v1 cell yet: run stage 'smoke' and read its log first"; exit 2; }
}

# submit_dataset <config> <label>: one GPU unit + its assemble job, v1 only, in the v1 env.
submit_dataset() {
  local cfg="$1" label="$2" gpu_id
  [ -f "${cfg}" ] || { log "missing config ${cfg}"; exit 2; }
  gpu_id=$(CONDA_ENV="${V1_ENV}" LANES=4 sbatch --parsable scripts/run_bo_benchmark.sh "${cfg}" "models=[tabpfn_v1]")
  gpu_id="${gpu_id%%;*}"
  local asm
  asm=$(CONDA_ENV="${V1_ENV}" sbatch --parsable --dependency="afterany:${gpu_id}" scripts/run_assemble.sh bo_benchmark "${cfg}")
  log "${label}: compute job ${gpu_id}, assemble job ${asm%%;*}"
}

case "${STAGE}" in
  status)
    if env_exists; then echo "  env:    built"; else echo "  env:    MISSING  -> bash scripts/submit_tabpfn_v1.sh env"; fi
    if has_v1_cell; then echo "  smoke:  a v1 cell has completed"; else echo "  smoke:  never executed -> stage 'smoke'"; fi
    for s in nhp spinal 5d_rat; do
      if [ -f "$(marker "$s")" ]; then echo "  ${s}:    submitted $(head -1 "$(marker "$s")")"
      elif has_v1_cell; then echo "  ${s}:    READY"
      else echo "  ${s}:    blocked on 'smoke'"; fi
    done
    echo "  note:   data/spinal must be staged on the cluster before the spinal stage"
    ;;
  env)
    guard env
    if env_exists; then log "env ${V1_ENV} already exists; nothing to do"; exit 0; fi
    log "building ${V1_ENV} with --mem=${ENV_MEM} (the 16G default OOM-killed conda 4.8.3's solver)"
    sbatch --mem="${ENV_MEM}" scripts/setup_env_job.sh v1
    record env
    log "when it finishes: bash scripts/submit_tabpfn_v1.sh status"
    ;;
  smoke)
    guard smoke
    need_env
    # One channel, one rep: the cheapest call that exercises fit/predict and the bar-distribution path.
    # Its own tag, so a failed probe can never touch the real PFN-bench run directory.
    CONDA_ENV="${V1_ENV}" sbatch scripts/run_bo_benchmark.sh configs/experiment/hyp0_pfn_bench_nhp.yaml \
      "models=[tabpfn_v1]" "dataset.subjects=[1]" "dataset.emgs=[0]" n_reps=1 tag="${SMOKE_TAG}"
    record smoke
    log "read the log before going further; the v1 API has never been executed"
    ;;
  nhp)
    guard nhp; need_env; need_smoke
    submit_dataset configs/experiment/hyp0_pfn_bench_nhp.yaml "E7 NHP"
    record nhp
    log "read E7's wall time (sacct) before submitting 5d_rat: v1's per-rep cost is unmeasured"
    ;;
  spinal)
    guard spinal; need_env; need_smoke
    [ -d data/spinal ] || { log "data/spinal is not staged on this machine"; exit 2; }
    submit_dataset configs/experiment/hyp0_pfn_bench_spinal.yaml "P16 spinal"
    record spinal
    ;;
  5d_rat)
    guard 5d_rat; need_env; need_smoke
    submit_dataset configs/experiment/hyp0_pfn_bench_5d_rat.yaml "E8 5d_rat"
    record 5d_rat
    ;;
  reassemble)
    # No compute: merge every cached cell (v1 included) into one table per dataset, in the MAIN env, since
    # cell identity carries the model version and not the environment (#8 rule 6).
    for ds in nhp spinal 5d_rat; do
      log "re-assembling hyp0_pfn_bench_${ds} from cache"
      python -m pfns4neurostim bo_benchmark --config "configs/experiment/hyp0_pfn_bench_${ds}.yaml" --only-cached
    done
    ;;
  *)
    sed -n '2,32p' "$0"; exit 2 ;;
esac
