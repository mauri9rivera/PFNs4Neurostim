#!/bin/bash
# ============================================================
#  SLURM job — finetune TabPFN and run evaluation on Mila.
#
#  All hyperparameters come from configs/ YAML files.
#  Set FAMILY=<key> and any runtime overrides, then sbatch.
#
#  ── Family 1 — per held_out_subject (finetuning, inter-subject) ──
#    FAMILY=1 DATASET=nhp sbatch --array=0%1 scripts/run_experiment.sh
#    FAMILY=1 DATASET=rat sbatch --array=0-1%2 scripts/run_experiment.sh
#    # TASK_ID indexes into subject list: nhp=[1], rat=[0,5]
#    # Config: configs/{dataset}_optimization.yaml
#
#  ── Family 2 — aug sweep (single job, all values serial) ─────────
#    FAMILY=2 DATASET=nhp HELD_OUT_SUBJ=1 sbatch --time=10:00:00 scripts/run_experiment.sh
#    # aug_pct_sweep values come from configs/nhp_aug_sweep.yaml
#    # Override sweep values: AUG_PCT_SWEEP="0.1 0.5 1.0" FAMILY=2 ...
#    # Config: configs/{dataset}_aug_sweep.yaml
#
#  ── Family 3 — budget sweep (single job, finetunes once) ─────────
#    FAMILY=3 DATASET=nhp HELD_OUT_SUBJ=1 sbatch scripts/run_experiment.sh
#    # budgets come from configs/nhp_optimization_budget.yaml
#    # Override: BUDGET_VALUES_STR="10 50 100" FAMILY=3 ...
#    # Config: configs/{dataset}_optimization_budget.yaml  (nhp only)
#
#  ── Family 4 — per held_out_emg (intra-EMG split) ────────────────
#    FAMILY=4 DATASET=nhp sbatch --array=0-5%2 scripts/run_experiment.sh
#    # TASK_ID = EMG index directly (0-5 for nhp)
#    # Config: configs/{dataset}_optimization.yaml
#
#  ── Family 5 — ID/OOD B1-B7 full suite ──────────────────────────
#    FAMILY=5 sbatch --gres=gpu:rtx8000:1 --mem-per-gpu=16G --cpus-per-task=4 --time=12:00:00 scripts/run_experiment.sh
#    # Config: configs/id_ood_bfull.yaml
#    # Override dataset subset: DATASETS="nhp rat" FAMILY=5 sbatch ...
#
#  ── Family 6 — LoRA per held_out_subject (mirrors Family 1) ──────
#    FAMILY=6 DATASET=nhp sbatch --array=0%1 scripts/run_experiment.sh
#    FAMILY=6 DATASET=rat sbatch --array=0-1%2 scripts/run_experiment.sh
#    # Config: configs/nhp_lora_ablation.yaml (nhp); set CONFIG= for others
#    # LoRA aug sweep: CONFIG=configs/nhp_lora_aug_sweep.yaml FAMILY=6 ...
#
#  ── Family 7 — vanilla benchmark per held_out_subject ────────────
#    FAMILY=7 DATASET=nhp sbatch --array=0%1 scripts/run_experiment.sh
#    FAMILY=7 DATASET=rat sbatch --array=0-1%2 scripts/run_experiment.sh
#    FAMILY=7 DATASET=spinal sbatch --array=0-10%4 scripts/run_experiment.sh
#    # Config: configs/{dataset}_vanilla_benchmark.yaml
#    # Also accepts VANILLA_CONFIG=<path> for backward compatibility
#
#  ── Family 8 — post-hoc aggregation (no GPU) ─────────────────────
#    FAMILY=8 AGG_CONFIG=configs/nhp_vanilla_benchmark.yaml sbatch --gres='' --cpus-per-task=4 --mem=8G scripts/run_experiment.sh
#
#  ── Family 9 — dense CKA layer sweep ─────────────────────────────
#    FAMILY=9 sbatch --gres=gpu:rtx8000:1 --mem-per-gpu=16G --cpus-per-task=4 --time=8:00:00 scripts/run_experiment.sh
#    # Config: configs/id_ood_b4dense.yaml
#
#  ── Family 0 — legacy single-run (backward compatible) ───────────
#    FAMILY=0 SPLIT=inter_subject MODE=optimization HELD_OUT_SUBJ=1 sbatch scripts/run_experiment.sh
#    # No config auto-resolved; set CONFIG=<path> to use a YAML.
#    # Note: EPOCHS, LR, N_AUG are no longer shell defaults — use --config.
#
#  ── Global overrides (any family) ────────────────────────────────
#    CONFIG=<path>     Override the auto-resolved YAML config
#    BUDGET=<int>      Override budget from YAML
#    N_REPS=<int>      Override n_reps from YAML
#    DIAGNOSTICS=1     Enable gradient/CKA monitoring
#    CLUSTER_DIAG=0    Disable HPC efficiency summary (on by default)
# ============================================================
#SBATCH --job-name=pfn4neurostim
#SBATCH --output=logs/slurm_%A_%a_%x.out
#SBATCH --error=logs/slurm_%A_%a_%x.err
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-gpu=7G
#SBATCH --time=4:00:00

set -euo pipefail
export CLUSTER_DIAG=${CLUSTER_DIAG:-1}

# ── Shared runtime vars ────────────────────────────────────────────────────────
DATASET=${DATASET:-nhp}
BUDGET=${BUDGET:-}
N_REPS=${N_REPS:-}
FAMILY=${FAMILY:-0}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
USE_LORA=0

# ── Subject lookup tables (TASK_ID → held-out subject index per dataset) ───────
# Families 1, 6, 7 index into _SUBJECTS via TASK_ID.
# Required --array spec: nhp → --array=0%1, rat → --array=0-1%2, spinal → --array=0-10%4
case "$DATASET" in
    nhp)    _SUBJECTS=(1) ;;
    rat)    _SUBJECTS=(0 5) ;;
    spinal) _SUBJECTS=(0 1 2 3 4 5 6 7 8 9 10) ;;
    *)      _SUBJECTS=() ;;
esac

# ── Family-specific parameter resolution ──────────────────────────────────────
if [ "$FAMILY" = "0" ]; then
    # Legacy single-run — no config auto-resolved; set CONFIG= to use a YAML.
    SPLIT=${SPLIT:-inter_subject}
    MODE=${MODE:-optimization}
    HELD_OUT_EMG=${HELD_OUT_EMG:-}
    HELD_OUT_SUBJ=${HELD_OUT_SUBJ:-}
    CFG=${CONFIG:-}

elif [ "$FAMILY" = "1" ]; then
    [ "$TASK_ID" -ge "${#_SUBJECTS[@]}" ] && { echo "ERROR: TASK_ID=$TASK_ID out of range for DATASET=$DATASET (max index $((${#_SUBJECTS[@]}-1)))" >&2; exit 1; }
    HELD_OUT_SUBJ=${_SUBJECTS[$TASK_ID]}
    HELD_OUT_EMG=
    SPLIT=inter_subject
    MODE=optimization
    CFG=${CONFIG:-configs/${DATASET}_optimization.yaml}

elif [ "$FAMILY" = "2" ]; then
    : "${HELD_OUT_SUBJ:?HELD_OUT_SUBJ must be set for FAMILY=2}"
    HELD_OUT_EMG=
    SPLIT=inter_subject
    MODE=aug_sweep_optimization
    CFG=${CONFIG:-configs/${DATASET}_aug_sweep.yaml}

elif [ "$FAMILY" = "3" ]; then
    : "${HELD_OUT_SUBJ:?HELD_OUT_SUBJ must be set for FAMILY=3}"
    HELD_OUT_EMG=
    SPLIT=inter_subject
    MODE=optimization_budget
    CFG=${CONFIG:-configs/${DATASET}_optimization_budget.yaml}

elif [ "$FAMILY" = "4" ]; then
    HELD_OUT_EMG=$TASK_ID
    HELD_OUT_SUBJ=
    SPLIT=intra_emg
    MODE=optimization
    CFG=${CONFIG:-configs/${DATASET}_optimization.yaml}

elif [ "$FAMILY" = "5" ]; then
    CFG=${CONFIG:-configs/id_ood_bfull.yaml}

elif [ "$FAMILY" = "9" ]; then
    CFG=${CONFIG:-configs/id_ood_b4dense.yaml}

elif [ "$FAMILY" = "6" ]; then
    [ "$TASK_ID" -ge "${#_SUBJECTS[@]}" ] && { echo "ERROR: TASK_ID=$TASK_ID out of range for DATASET=$DATASET (max index $((${#_SUBJECTS[@]}-1)))" >&2; exit 1; }
    HELD_OUT_SUBJ=${_SUBJECTS[$TASK_ID]}
    HELD_OUT_EMG=
    SPLIT=inter_subject
    MODE=optimization
    USE_LORA=1
    # Only nhp has a dedicated LoRA config; for other datasets set CONFIG= explicitly.
    case "$DATASET" in
        nhp) CFG=${CONFIG:-configs/nhp_lora_ablation.yaml} ;;
        *)   CFG=${CONFIG:-} ;;
    esac

elif [ "$FAMILY" = "7" ]; then
    [ "${#_SUBJECTS[@]}" -eq 0 ] && { echo "ERROR: No subjects defined for DATASET=$DATASET" >&2; exit 1; }
    [ "$TASK_ID" -ge "${#_SUBJECTS[@]}" ] && { echo "ERROR: TASK_ID=$TASK_ID out of range for DATASET=$DATASET (max index $((${#_SUBJECTS[@]}-1)))" >&2; exit 1; }
    HELD_OUT_SUBJ=${_SUBJECTS[$TASK_ID]}
    # VANILLA_CONFIG accepted for backward compatibility alongside CONFIG.
    CFG=${VANILLA_CONFIG:-${CONFIG:-configs/${DATASET}_vanilla_benchmark.yaml}}

elif [ "$FAMILY" = "8" ]; then
    : "${AGG_CONFIG:?AGG_CONFIG must be set for FAMILY=8}"

else
    echo "Unknown FAMILY=$FAMILY. Must be 0-9." >&2
    exit 1
fi

# ── Build optional flags (used by finetuning families 0-4, 6) ────────────────
EXTRA_FLAGS=""
[ -n "${HELD_OUT_EMG:-}" ] && EXTRA_FLAGS="$EXTRA_FLAGS --held_out_emg $HELD_OUT_EMG"
[ -n "${HELD_OUT_SUBJ:-}" ] && EXTRA_FLAGS="$EXTRA_FLAGS --held_out_subj $HELD_OUT_SUBJ"
[ -n "${BUDGET_VALUES_STR:-}" ] && EXTRA_FLAGS="$EXTRA_FLAGS --budgets $BUDGET_VALUES_STR"
[ -n "${BUDGET:-}" ] && EXTRA_FLAGS="$EXTRA_FLAGS --budget $BUDGET"
[ -n "${N_REPS:-}" ] && EXTRA_FLAGS="$EXTRA_FLAGS --n_reps $N_REPS"
[ "${DIAGNOSTICS:-0}" = "1" ] && EXTRA_FLAGS="$EXTRA_FLAGS --diagnostics"
[ "$USE_LORA" = "1" ] && EXTRA_FLAGS="$EXTRA_FLAGS --lora"
[ -n "${AUG_PCT_SWEEP:-}" ] && EXTRA_FLAGS="$EXTRA_FLAGS --aug_pct_sweep $AUG_PCT_SWEEP"

# ── Environment ───────────────────────────────────────────────────────────────
module load miniconda/3
conda activate pfns4neurostim

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs output/runs

# ── Run ───────────────────────────────────────────────────────────────────────
if [ "$FAMILY" = "5" ] || [ "$FAMILY" = "9" ]; then
    DATASETS=${DATASETS:-"nhp rat spinal"}
    echo "[$(date)] family=$FAMILY config=$CFG datasets=$DATASETS"
    mkdir -p output/id_ood
    srun python src/id_ood_analysis.py --config "$CFG" --datasets $DATASETS
    echo "[$(date)] Done. Results in output/id_ood/"

elif [ "$FAMILY" = "7" ]; then
    echo "[$(date)] family=7 task=$TASK_ID dataset=$DATASET subj=$HELD_OUT_SUBJ config=$CFG"
    VFLAGS="--device cuda --held_out_subj $HELD_OUT_SUBJ --save"
    [ -n "${BUDGET:-}" ] && VFLAGS="$VFLAGS --budget $BUDGET"
    [ -n "${N_REPS:-}" ] && VFLAGS="$VFLAGS --n_reps $N_REPS"
    srun python src/vanilla_benchmark.py --config "$CFG" $VFLAGS
    echo "[$(date)] Done. Results in output/runs/"

elif [ "$FAMILY" = "8" ]; then
    echo "[$(date)] family=8 config=$AGG_CONFIG"
    mkdir -p output/aggregated
    srun python src/aggregate.py --config "$AGG_CONFIG"
    echo "[$(date)] Done. Results in output/aggregated/"

else
    echo "[$(date)] family=$FAMILY task=$TASK_ID dataset=$DATASET split=$SPLIT mode=$MODE ${HELD_OUT_SUBJ:+subj=$HELD_OUT_SUBJ} ${HELD_OUT_EMG:+emg=$HELD_OUT_EMG} ${CFG:+config=$CFG}"
    FFLAGS="--dataset $DATASET --split $SPLIT --mode $MODE --device cuda --save"
    [ -n "${CFG:-}" ] && FFLAGS="$FFLAGS --config $CFG"
    srun python src/finetuning.py $FFLAGS $EXTRA_FLAGS
    echo "[$(date)] Done. Results in output/runs/"
fi
