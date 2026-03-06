#!/bin/bash
# ============================================================
#  SLURM job — finetune TabPFN and run evaluation on Mila
#
#  Usage (single run, legacy):
#    sbatch scripts/run_experiment.sh
#
#  Usage (job array families):
#    # Family 0 — single run (legacy)
#    sbatch scripts/run_experiment.sh
#
#    # Family 1 — per held_out_subject
#    FAMILY=1 DATASET=nhp N_AUG=25 BUDGET=100 sbatch --array=0-3 scripts/run_experiment.sh
#    FAMILY=1 DATASET=rat N_AUG=25 BUDGET=100 sbatch --array=0-5 scripts/run_experiment.sh
#
#    # Family 2 — aug sweep (5 values → array 0-4)
#    FAMILY=2 DATASET=nhp HELD_OUT_SUBJ=1 BUDGET=100 \
#      AUG_VALUES_STR="1 5 10 25 50" sbatch --array=0-4 scripts/run_experiment.sh
#
#    # Family 3 — budget sweep (6 values → array 0-5)
#    FAMILY=3 DATASET=nhp HELD_OUT_SUBJ=1 N_AUG=25 \
#      BUDGET_VALUES_STR="10 25 50 100 150 200" sbatch --array=0-5 scripts/run_experiment.sh
#
#    # Family 4 — per held_out_emg (nhp has 6 EMGs → array 0-5)
#    FAMILY=4 DATASET=nhp N_AUG=25 BUDGET=100 sbatch --array=0-5 scripts/run_experiment.sh
# ============================================================
#SBATCH --job-name=pfn4neurostim
#SBATCH --output=logs/slurm_%A_%a_%x.out
#SBATCH --error=logs/slurm_%A_%a_%x.err
#SBATCH --partition=main
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00

set -euo pipefail

# ── Shared defaults ──────────────────────────────────────────────────────────
DATASET=${DATASET:-nhp}
EPOCHS=${EPOCHS:-20}
LR=${LR:-1e-6}
N_REPS=${N_REPS:-30}
FAMILY=${FAMILY:-0}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# ── Family-specific parameter resolution ─────────────────────────────────────
if [ "$FAMILY" = "0" ]; then
    # Legacy single-run (backward compatible with original script)
    SPLIT=${SPLIT:-inter_subject}
    MODE=${MODE:-fit}
    BUDGET=${BUDGET:-100}
    N_AUG=${N_AUG:-25}
    HELD_OUT_EMG=${HELD_OUT_EMG:-}
    HELD_OUT_SUBJ=${HELD_OUT_SUBJ:-}

elif [ "$FAMILY" = "1" ]; then
    # Per held_out_subject — inter_subject, fit+optimization
    # sbatch --array=0-3  FAMILY=1 DATASET=nhp N_AUG=25 BUDGET=100
    # sbatch --array=0-5  FAMILY=1 DATASET=rat N_AUG=25 BUDGET=100
    HELD_OUT_SUBJ=$TASK_ID
    HELD_OUT_EMG=
    SPLIT=inter_subject
    MODE=fit,optimization
    N_AUG=${N_AUG:-25}
    BUDGET=${BUDGET:-100}

elif [ "$FAMILY" = "2" ]; then
    # Aug sweep — fixed held_out_subj, optimization, n_aug varies
    # sbatch --array=0-4  FAMILY=2 DATASET=nhp HELD_OUT_SUBJ=1 BUDGET=100 \
    #                     AUG_VALUES_STR="1 5 10 25 50"
    : "${HELD_OUT_SUBJ:?HELD_OUT_SUBJ must be set for FAMILY=2}"
    AUG_VALUES_STR=${AUG_VALUES_STR:-"1 5 10 25 50"}
    read -ra _AUG <<< "$AUG_VALUES_STR"
    N_AUG=${_AUG[$TASK_ID]}
    HELD_OUT_EMG=
    SPLIT=inter_subject
    MODE=optimization
    BUDGET=${BUDGET:-100}

elif [ "$FAMILY" = "3" ]; then
    # Budget sweep — fixed held_out_subj and n_aug, optimization, budget varies
    # sbatch --array=0-5  FAMILY=3 DATASET=nhp HELD_OUT_SUBJ=1 N_AUG=25 \
    #                     BUDGET_VALUES_STR="10 25 50 100 150 200"
    : "${HELD_OUT_SUBJ:?HELD_OUT_SUBJ must be set for FAMILY=3}"
    BUDGET_VALUES_STR=${BUDGET_VALUES_STR:-"10 25 50 100 150 200"}
    read -ra _BUD <<< "$BUDGET_VALUES_STR"
    BUDGET=${_BUD[$TASK_ID]}
    HELD_OUT_EMG=
    SPLIT=inter_subject
    MODE=optimization
    N_AUG=${N_AUG:-25}

elif [ "$FAMILY" = "4" ]; then
    # Per held_out_emg — intra_emg, fit+optimization
    # sbatch --array=0-5  FAMILY=4 DATASET=nhp N_AUG=25 BUDGET=100
    HELD_OUT_EMG=$TASK_ID
    HELD_OUT_SUBJ=
    SPLIT=intra_emg
    MODE=fit,optimization
    N_AUG=${N_AUG:-25}
    BUDGET=${BUDGET:-100}

else
    echo "Unknown FAMILY=$FAMILY. Must be 0-4." >&2
    exit 1
fi

# ── Build optional flags ──────────────────────────────────────────────────────
EXTRA_FLAGS=""
[ -n "${HELD_OUT_EMG:-}"  ] && EXTRA_FLAGS="$EXTRA_FLAGS --held_out_emg $HELD_OUT_EMG"
[ -n "${HELD_OUT_SUBJ:-}" ] && EXTRA_FLAGS="$EXTRA_FLAGS --held_out_subj $HELD_OUT_SUBJ"

# ── Environment ───────────────────────────────────────────────────────────────
module load miniconda/3
conda activate pfns4neurostim

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs output/runs

# ── Run ───────────────────────────────────────────────────────────────────────
echo "[$(date)] family=$FAMILY task=$TASK_ID dataset=$DATASET split=$SPLIT \
mode=$MODE n_aug=$N_AUG budget=$BUDGET \
${HELD_OUT_SUBJ:+subj=$HELD_OUT_SUBJ} ${HELD_OUT_EMG:+emg=$HELD_OUT_EMG}"

python src/finetuning.py \
    --dataset         "$DATASET"  \
    --split           "$SPLIT"    \
    --mode            "$MODE"     \
    --device          cuda        \
    --epochs          "$EPOCHS"   \
    --lr              "$LR"       \
    --n_augmentations "$N_AUG"    \
    --budget          "$BUDGET"   \
    --n_reps          "$N_REPS"   \
    --save            \
    $EXTRA_FLAGS

echo "[$(date)] Done. Results in output/runs/"
