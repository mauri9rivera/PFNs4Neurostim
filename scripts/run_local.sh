#!/bin/bash
# Everything that needs NO cluster: re-render every existing run, rebuild the post-hoc metrics, export the
# tables. Free, idempotent, and the right first step before spending a GPU-hour — it shows the current state
# and it exercises new figure code on real runs rather than on a synthetic smoke.
#
#   bash scripts/run_local.sh                 # every FREE step, in order (never the hypc runs)
#   bash scripts/run_local.sh replot          # one step
#   bash scripts/run_local.sh list            # what the steps are and what each needs
#
# The hypc steps below are the exception: they are real experiments, hours long, and opt-in only. They are
# here because Hyp C is the one group that belongs on this machine rather than on the cluster -- see
# "Hyp C locally" in the step list.
#
# Steps
#   replot    every run directory under output/ re-rendered from its tidy.csv + trajectories.pkl. Picks up
#             the 2026-09-30 figure work with no recompute: A8 queries-to-target, A9 anytime regret, the
#             chromatic floor/ceiling tokens, the context-faceted CKA panels.
#   tables    the cross-run summary tables (scripts/export_summary_tables.py).
#   bridge    the Demo 1 <-> Demo 2 synthetic/real bridge for any dataset whose Demo 1 K2 run exists.
#   verify    the fast test suite, plus the slow y-affine-invariance and mechanism tests that the fast suite
#             deselects. This is the one step that wants the local GPU.
#
# Hyp C locally (opt-in, NOT part of `all`)
#   hypc-gpu  C1 (update rule, ~4 h) then C2 (CKA + placement ladder, ~5.5 h), SEQUENTIALLY: both want the
#             GPU and running them together would only make each slower. Those two estimates are this
#             machine's own numbers -- C1-C3 last ran here on 2026-09-24/25 -- so they need no cluster
#             translation. Nothing else may use the GPU while they run.
#   hypc-cpu  C3 (placement MMD / W2, ~3 h). Its config declares `device: cpu`, so it can run at the same
#             time as hypc-gpu.
#   hypc-link M8: join C1's per-channel mechanism metrics with a finished BO run's tidy.csv. Takes the
#             tidy.csv as its argument. This is what makes "run Hyp C now, combine with the PFN results
#             later" work: the link is computed OUTSIDE the recompute guard, so --replot adds it to an
#             existing C1 run in seconds instead of re-running the 4 h grid.
#
#             bash scripts/run_local.sh hypc-link output/benchmark/nhp/hyp-a-nhp/tidy.csv
#
#   C4 (5d_rat placement) stays out: its cost has never been measured (#18 Step 5 times one channel first).
#
# Nothing here writes to output/cells, so it can run while cluster jobs are in flight; it only reads the run
# directories you have exported. `--replot` is refused on a run with no tidy.csv, so a half-exported run is
# reported rather than silently skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

STEP="${1:-all}"
PY="${PYTHON:-python}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

# Every step shells out to ${PY}. If that interpreter cannot import the package, each step fails in its
# own way and the failures read like data problems rather than a wrong interpreter -- `tables` even
# reported a missing run (2026-09-30). Check once, loudly, before anything runs.
if ! "${PY}" -c "import pfns4neurostim" >/dev/null 2>&1; then
  echo "[run_local] ${PY} cannot import pfns4neurostim." >&2
  echo "[run_local] activate the env (conda activate pfns4neurostim) or point PYTHON at its interpreter:" >&2
  echo "[run_local]   PYTHON=/path/to/envs/pfns4neurostim/bin/python bash scripts/run_local.sh ${STEP}" >&2
  exit 1
fi

step_replot() {
  echo "== replot: every run directory from its own CSVs =="
  "${PY}" scripts/replot_all.py --run
}

step_tables() {
  echo "== tables: cross-run summary tables =="
  if "${PY}" scripts/export_summary_tables.py; then
    echo "   wrote the summary tables"
  else
    echo "   SKIPPED: export_summary_tables.py failed; it hardcodes its inputs, so the usual cause is a" >&2
    echo "   run it cites not existing yet (task plan #1 Step 14). Read the traceback above to be sure." >&2
  fi
}

step_bridge() {
  echo "== bridge: Demo 1 surface with the Demo 2 channels overlaid (S10) =="
  local cfg dataset source ran=0
  for cfg in configs/experiment/stress_k2_channel_demo1_*.yaml; do
    dataset=$("${PY}" -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['defaults']['dataset'])" "${cfg}")
    # The Demo 1 run must exist too: --replot rebuilds the Demo 1 figures and overlays the Demo 2 source
    # on them, so without it the step fails on a missing tidy.csv rather than reporting nothing to do.
    target=$("${PY}" -c "import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); print(c['tag'])" "${cfg}")
    if ! ls output/stress/k2_channel/"${dataset}"/*"${target}"*/tidy.csv >/dev/null 2>&1; then
      echo "   skipped ${cfg}: its own Demo 1 run has not been produced yet"
      continue
    fi
    for source in output/stress/k2_channel/"${dataset}"/*/; do
      [ -f "${source}/tidy.csv" ] || continue
      echo "   ${cfg}  <-  ${source}"
      "${PY}" -m pfns4neurostim stress_sweep --config "${cfg}" --replot --bridge "${source%/}"
      ran=1
    done
  done
  [ "${ran}" = "1" ] || echo "   nothing to bridge yet: no Demo 2 K2-channel run under output/stress/k2_channel/"
}

step_hypc_gpu() {
  echo "== hypc-gpu: C1 then C2, sequentially (hours; nothing else should use the GPU) =="
  "${PY}" -m pfns4neurostim mechanism --config configs/experiment/mechanism_update_rule_nhp.yaml
  "${PY}" -m pfns4neurostim mechanism --config configs/experiment/mechanism_cka_nhp.yaml
}

step_hypc_cpu() {
  echo "== hypc-cpu: C3 placement MMD / W2 (device: cpu in the config) =="
  "${PY}" -m pfns4neurostim mechanism --config configs/experiment/mechanism_placement_nhp.yaml
}

step_hypc_link() {   # step_hypc_link <tidy.csv of a finished BO run>
  local tidy="${1:-}"
  if [ -z "${tidy}" ] || [ ! -f "${tidy}" ]; then
    echo "usage: bash scripts/run_local.sh hypc-link <path to a finished BO run's tidy.csv>" >&2
    exit 2
  fi
  echo "== hypc-link: M8 predictive link, C1 metrics x ${tidy} =="
  "${PY}" -m pfns4neurostim mechanism --config configs/experiment/mechanism_update_rule_nhp.yaml --replot --set "update_rule.link.tidy_csv=${tidy}"
}

step_verify() {
  echo "== verify: fast suite, then the slow/GPU tests the fast suite deselects =="
  "${PY}" -m pytest tests -m "not slow and not gpu and not legacy" -q
  "${PY}" -m pytest tests/models/test_y_affine_invariance.py tests/analysis/test_mechanism_runner.py -q
}

case "${STEP}" in
  all)     step_replot; step_tables; step_bridge; step_verify ;;
  replot)  step_replot ;;
  tables)  step_tables ;;
  bridge)  step_bridge ;;
  verify)  step_verify ;;
  hypc-gpu)  step_hypc_gpu ;;
  hypc-cpu)  step_hypc_cpu ;;
  hypc-link) step_hypc_link "${2:-}" ;;
  list)
    echo "free steps:  replot | tables | bridge | verify (local GPU, minutes)"
    echo "hypc steps:  hypc-gpu (C1+C2, ~9.5 h GPU) | hypc-cpu (C3, ~3 h CPU) | hypc-link <tidy.csv>"
    echo "runs found under output/:"
    find output -maxdepth 4 -name tidy.csv -printf '  %h\n' 2>/dev/null | sort || true
    ;;
  *) echo "usage: bash scripts/run_local.sh [all|replot|tables|bridge|verify|hypc-gpu|hypc-cpu|hypc-link|list]" >&2; exit 2 ;;
esac
