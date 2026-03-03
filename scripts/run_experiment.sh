#!/bin/bash
# ============================================================
#  SLURM job — finetune TabPFN and run evaluation on Mila
#
#  Usage: sbatch scripts/run_experiment.sh
#  Or override params via environment variables:
#    DATASET=rat SPLIT=intra_emg HELD_OUT_EMG=3 MODE=fit,optimization \
#      sbatch scripts/run_experiment.sh
# ============================================================
#SBATCH --job-name=pfn4neurostim
#SBATCH --output=logs/slurm_%j_%x.out
#SBATCH --error=logs/slurm_%j_%x.err
#SBATCH --partition=main
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

# --- configurable via environment variables ---
DATASET=${DATASET:-nhp}
SPLIT=${SPLIT:-inter_subject}
MODE=${MODE:-fit}
EPOCHS=${EPOCHS:-20}
LR=${LR:-1e-6}
N_AUG=${N_AUG:-25}
BUDGET=${BUDGET:-100}
N_REPS=${N_REPS:-30}
HELD_OUT_EMG=${HELD_OUT_EMG:-}
HELD_OUT_SUBJ=${HELD_OUT_SUBJ:-}

# --- environment ---
module load miniconda/3
conda activate pfns4neurostim

# --- directories ---
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs output/results output/fitness output/optimization

# --- build optional extra flags ---
EXTRA_FLAGS=""
[ -n "$HELD_OUT_EMG"  ] && EXTRA_FLAGS="$EXTRA_FLAGS --held_out_emg $HELD_OUT_EMG"
[ -n "$HELD_OUT_SUBJ" ] && EXTRA_FLAGS="$EXTRA_FLAGS --held_out_subj $HELD_OUT_SUBJ"

# --- run ---
echo "[$(date)] Starting experiment: dataset=$DATASET split=$SPLIT mode=$MODE"

python src/finetuning.py \
    --dataset   "$DATASET"  \
    --split     "$SPLIT"    \
    --mode      "$MODE"     \
    --device    cuda        \
    --epochs    "$EPOCHS"   \
    --lr        "$LR"       \
    --n_augmentations "$N_AUG" \
    --budget    "$BUDGET"   \
    --n_reps    "$N_REPS"   \
    --save      \
    $EXTRA_FLAGS

echo "[$(date)] Done. Results in output/results/"
