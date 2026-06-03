#!/bin/bash
# ============================================================
#  SLURM job — Phase 1 (vanilla benchmark) and Phase 2
#  (ID/OOD analysis) experiments for PFNs4Neurostim.
#
#  Usage: set JOB=<key> and submit.  All hyperparameters come from
#  the canonical YAML configs under configs/; env vars are runtime
#  overrides only.
#
#  ── Phase 1 — Vanilla Benchmark (GPU, ~4h) ───────────────────
#    JOB=A1 sbatch scripts/run_all_experiments.sh
#    JOB=A2 sbatch scripts/run_all_experiments.sh
#    JOB=A3 sbatch scripts/run_all_experiments.sh
#    JOB=A4 sbatch scripts/run_all_experiments.sh
#
#  Phase 1 env-var overrides (optional):
#    VANILLA_CONFIG=<path>   YAML config (per-job default listed below)
#    SUBJECTS=held_out|all   'all' triggers LOO over every subject
#    HELD_OUT_SUBJ=<int>     Single subject; overrides SUBJECTS
#  Examples:
#    JOB=A2 SUBJECTS=all sbatch scripts/run_all_experiments.sh
#    JOB=A2 HELD_OUT_SUBJ=0 sbatch scripts/run_all_experiments.sh
#    JOB=A3 VANILLA_CONFIG=configs/rat_vanilla_benchmark.yaml sbatch scripts/run_all_experiments.sh
#
#  ── Phase 2 — ID/OOD Analysis ─────────────────────────────────
#  Each B-job delegates entirely to its config in configs/id_ood_b*.yaml.
#  Override config:        B_CONFIG=<path> JOB=B1 sbatch ...
#  Override dataset subset: DATASETS="nhp rat spinal" JOB=B1 sbatch ...
#
#    GPU jobs (~2-4h, --mem-per-gpu=16G):
#      JOB=B1 sbatch --gres=gpu:rtx8000:1 --cpus-per-task=4 --mem-per-gpu=16G --time=4:00:00 scripts/run_all_experiments.sh
#      JOB=B3 sbatch --gres=gpu:rtx8000:1 --cpus-per-task=4 --mem-per-gpu=16G --time=4:00:00 scripts/run_all_experiments.sh
#      JOB=B4 sbatch --gres=gpu:rtx8000:1 --cpus-per-task=4 --mem-per-gpu=16G --time=4:00:00 scripts/run_all_experiments.sh
#      JOB=B4dense sbatch --gres=gpu:rtx8000:1 --cpus-per-task=4 --mem-per-gpu=24G --time=6:00:00 scripts/run_all_experiments.sh
#      JOB=B6 sbatch --gres=gpu:rtx8000:1 --cpus-per-task=4 --mem-per-gpu=16G --time=4:00:00 scripts/run_all_experiments.sh
#      JOB=B7 sbatch --gres=gpu:rtx8000:1 --cpus-per-task=4 --mem-per-gpu=16G --time=8:00:00 scripts/run_all_experiments.sh
#      JOB=Bfull sbatch --gres=gpu:rtx8000:1 --cpus-per-task=4 --mem-per-gpu=24G --time=12:00:00 scripts/run_all_experiments.sh
#
#    CPU job (~4h, no TabPFN inference, pure-NumPy distance metrics):
#      JOB=B2 sbatch --gres='' --cpus-per-task=4 --mem=16G --time=4:00:00 scripts/run_all_experiments.sh
#
# ── Compute resource knobs ─────────────────────────────────────
#  Change these SBATCH lines to tune resources per job:
#    --gres=gpu:rtx8000:1   GPU model + count  (use --gres='' for B2)
#    --cpus-per-task=2      CPU cores          (use 4 for Phase 2 jobs)
#    --mem-per-gpu=7G       RAM per GPU        (use 16G/24G for Phase 2)
#    --time=4:00:00         Wall-clock limit   (use 8h for B7, 12h for Bfull)
#    --partition=main       Queue
# ============================================================

#SBATCH --job-name=pfn4neurostim
#SBATCH --output=logs/slurm_%A_%x.out
#SBATCH --error=logs/slurm_%A_%x.err
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-gpu=7G
#SBATCH --time=4:00:00

set -euo pipefail
export CLUSTER_DIAG=1

JOB=${JOB:-}

if [ -z "$JOB" ]; then
    echo "ERROR: JOB must be set. Valid values: A1 A2 A3 A4 B1 B2 B3 B4 B4dense B6 B7 Bfull" >&2
    exit 1
fi

# ── Phase 1 overrides ──────────────────────────────────────────────────────────
SUBJECTS=${SUBJECTS:-}
HELD_OUT_SUBJ=${HELD_OUT_SUBJ:-}
VANILLA_OVERRIDES=""
[ -n "$SUBJECTS" ] && VANILLA_OVERRIDES="$VANILLA_OVERRIDES --subjects $SUBJECTS"
[ -n "$HELD_OUT_SUBJ" ] && VANILLA_OVERRIDES="$VANILLA_OVERRIDES --held_out_subj $HELD_OUT_SUBJ"

# ── Phase 2 overrides ──────────────────────────────────────────────────────────
# Default datasets for all B-jobs: nhp spinal.
# Override: DATASETS="nhp rat spinal" JOB=B1 sbatch ...
DATASETS=${DATASETS:-"nhp spinal"}

# ── Environment ───────────────────────────────────────────────────────────────
module load miniconda/3
conda activate pfns4neurostim

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs output/runs output/id_ood

# ── Phase 1 — Vanilla Benchmark ───────────────────────────────────────────────

if [ "$JOB" = "A1" ] || [ "$JOB" = "A2" ]; then
    CFG=${VANILLA_CONFIG:-configs/nhp_vanilla_benchmark.yaml}
    echo "[$(date)] $JOB — NHP vanilla optimization config=$CFG ${VANILLA_OVERRIDES:+overrides=$VANILLA_OVERRIDES}"
    srun python src/vanilla_benchmark.py --config "$CFG" --mode optimization --save $VANILLA_OVERRIDES

elif [ "$JOB" = "A3" ]; then
    CFG=${VANILLA_CONFIG:-configs/rat_vanilla_benchmark.yaml}
    echo "[$(date)] A3 — Rat vanilla benchmark config=$CFG ${VANILLA_OVERRIDES:+overrides=$VANILLA_OVERRIDES}"
    srun python src/vanilla_benchmark.py --config "$CFG" --save $VANILLA_OVERRIDES

elif [ "$JOB" = "A4" ]; then
    CFG=${VANILLA_CONFIG:-configs/nhp_vanilla_benchmark.yaml}
    echo "[$(date)] A4 — NHP budget sweep config=$CFG ${VANILLA_OVERRIDES:+overrides=$VANILLA_OVERRIDES}"
    srun python src/vanilla_benchmark.py --config "$CFG" --mode optimization_budget --save $VANILLA_OVERRIDES

# ── Phase 2 — ID/OOD Analysis ─────────────────────────────────────────────────

elif [ "$JOB" = "B1" ]; then
    CFG=${B_CONFIG:-configs/id_ood_b1.yaml}
    echo "[$(date)] B1 — Shannon Entropy datasets=$DATASETS config=$CFG"
    srun python src/id_ood_analysis.py --config "$CFG" --datasets $DATASETS

elif [ "$JOB" = "B2" ]; then
    CFG=${B_CONFIG:-configs/id_ood_b2.yaml}
    echo "[$(date)] B2+B5 — MMD + Wasserstein-2 datasets=$DATASETS config=$CFG"
    srun python src/id_ood_analysis.py --config "$CFG" --datasets $DATASETS

elif [ "$JOB" = "B3" ]; then
    CFG=${B_CONFIG:-configs/id_ood_b3.yaml}
    echo "[$(date)] B3 — Mahalanobis Distance datasets=$DATASETS config=$CFG"
    srun python src/id_ood_analysis.py --config "$CFG" --datasets $DATASETS

elif [ "$JOB" = "B4" ]; then
    CFG=${B_CONFIG:-configs/id_ood_b4.yaml}
    echo "[$(date)] B4 — Layer-wise CKA (layers 4,13,17) datasets=$DATASETS config=$CFG"
    srun python src/id_ood_analysis.py --config "$CFG" --datasets $DATASETS

elif [ "$JOB" = "B4dense" ]; then
    CFG=${B_CONFIG:-configs/id_ood_b4dense.yaml}
    echo "[$(date)] B4dense — Dense Layer-wise CKA (10-layer sweep) datasets=$DATASETS config=$CFG"
    srun python src/id_ood_analysis.py --config "$CFG" --datasets $DATASETS

elif [ "$JOB" = "B6" ]; then
    CFG=${B_CONFIG:-configs/id_ood_b6.yaml}
    echo "[$(date)] B6 — RSA datasets=$DATASETS config=$CFG"
    srun python src/id_ood_analysis.py --config "$CFG" --datasets $DATASETS

elif [ "$JOB" = "B7" ]; then
    CFG=${B_CONFIG:-configs/id_ood_b7.yaml}
    echo "[$(date)] B7 — Procrustes BO-Trajectory datasets=$DATASETS config=$CFG"
    srun python src/id_ood_analysis.py --config "$CFG" --datasets $DATASETS

elif [ "$JOB" = "Bfull" ]; then
    CFG=${B_CONFIG:-configs/id_ood_bfull.yaml}
    echo "[$(date)] Bfull — ID/OOD full suite (B1-B7) datasets=$DATASETS config=$CFG"
    srun python src/id_ood_analysis.py --config "$CFG" --datasets $DATASETS

else
    echo "ERROR: Unknown JOB='$JOB'. Valid values: A1 A2 A3 A4 B1 B2 B3 B4 B4dense B6 B7 Bfull" >&2
    exit 1
fi

echo "[$(date)] Done."
