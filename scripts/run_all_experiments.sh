#!/bin/bash
# ============================================================
#  SLURM job — run Phase 1 (vanilla benchmark) and Phase 2
#  (ID/OOD analysis) experiments for PFNs4Neurostim.
#
#  Run one job at a time by setting JOB=<key> before sbatch.
#
#  ── Phase 1 — Vanilla Benchmark (GPU, ~4h, 7G) ───────────
#    JOB=A1 sbatch scripts/run_all_experiments.sh
#    JOB=A2 sbatch scripts/run_all_experiments.sh
#    JOB=A3 sbatch scripts/run_all_experiments.sh
#    JOB=A4 sbatch scripts/run_all_experiments.sh
#
#  Phase 1 env-var overrides (optional; default = held-out subjects of the per-job config):
#    VANILLA_CONFIG=<path>     YAML config (A1/A2/A4 default NHP, A3 default rat)
#    SUBJECTS=held_out|all     'all' triggers LOO over every subject in the dataset
#    HELD_OUT_SUBJ=<int>       Single subject (overrides SUBJECTS); use with sbatch arrays
#  Examples:
#    JOB=A2 SUBJECTS=all sbatch scripts/run_all_experiments.sh
#    JOB=A2 HELD_OUT_SUBJ=0 sbatch scripts/run_all_experiments.sh
#    JOB=A2 VANILLA_CONFIG=configs/rat_vanilla_benchmark.yaml sbatch scripts/run_all_experiments.sh
#
#  ── Phase 2 — ID/OOD Analysis ─────────────────────────────
#  GPU split (B1/B3/B4/B4dense/B6/B7 hit TabPFN forward passes; B2/B5 are pure-NumPy):
#
#    GPU jobs (~2-4h, 16G):
#      JOB=B1 sbatch --gres=gpu:rtx8000:1 --cpus-per-task=4 --mem=16G --time=4:00:00 scripts/run_all_experiments.sh
#      JOB=B3 sbatch --gres=gpu:rtx8000:1 --cpus-per-task=4 --mem=16G --time=4:00:00 scripts/run_all_experiments.sh
#      JOB=B4 sbatch --gres=gpu:rtx8000:1 --cpus-per-task=4 --mem=16G --time=4:00:00 scripts/run_all_experiments.sh
#      JOB=B4dense sbatch --gres=gpu:rtx8000:1 --cpus-per-task=4 --mem=24G --time=6:00:00 scripts/run_all_experiments.sh
#      JOB=B6 sbatch --gres=gpu:rtx8000:1 --cpus-per-task=4 --mem=16G --time=4:00:00 scripts/run_all_experiments.sh
#      JOB=B7 sbatch --gres=gpu:rtx8000:1 --cpus-per-task=4 --mem=16G --time=8:00:00 scripts/run_all_experiments.sh
#      JOB=Bfull sbatch --gres=gpu:rtx8000:1 --cpus-per-task=4 --mem=24G --time=12:00:00 scripts/run_all_experiments.sh
#
#    CPU jobs (~4h, 16G — no TabPFN inference, pure-NumPy distance/projection metrics):
#      JOB=B2 sbatch --gres='' --cpus-per-task=4 --mem=16G --time=4:00:00 scripts/run_all_experiments.sh
#
# ── Compute resource knobs ─────────────────────────────────
#  Change these SBATCH lines to tune resources per job:
#
#    --gres=gpu:rtx8000:1   GPU model + count  (use --gres='' for CPU-only Phase 2 jobs)
#    --cpus-per-task=2       CPU cores          (use 4 for Phase 2 / Bfull)
#    --mem=7G                RAM                (use 16G for Phase 2 / Bfull)
#    --time=4:00:00          Wall-clock limit   (use 8:00:00 for B-jobs, 12:00:00 for Bfull)
#    --partition=main        Queue              (change to long/unkillable as needed)
# ============================================================



#SBATCH --job-name=pfn4neurostim
#SBATCH --output=logs/slurm_%A_%x.out
#SBATCH --error=logs/slurm_%A_%x.err
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=7G
#SBATCH --time=4:00:00

set -euo pipefail

JOB=${JOB:-}

if [ -z "$JOB" ]; then
    echo "ERROR: JOB must be set. Valid values: A1 A2 A3 A4 B1 B2 B3 B4 B4dense B6 B7 Bfull" >&2
    exit 1
fi

# ── Phase 1 (A-jobs) overrides ────────────────────────────────────────────────
#   VANILLA_CONFIG   path to a yaml config — overrides the per-job default (A1/A2/A4 default to NHP, A3 defaults to rat).
#   SUBJECTS         'held_out' (default in the config) | 'all' (LOO over all subjects).
#   HELD_OUT_SUBJ    integer subject index — single-subject run (overrides SUBJECTS).

SUBJECTS=${SUBJECTS:-}
HELD_OUT_SUBJ=${HELD_OUT_SUBJ:-}
VANILLA_CONFIG=${VANILLA_CONFIG:-}

VANILLA_OVERRIDES=""
[ -n "$SUBJECTS" ]      && VANILLA_OVERRIDES="$VANILLA_OVERRIDES --subjects $SUBJECTS"
[ -n "$HELD_OUT_SUBJ" ] && VANILLA_OVERRIDES="$VANILLA_OVERRIDES --held_out_subj $HELD_OUT_SUBJ"

# ── Phase 2 (B-jobs) overrides ────────────────────────────────────────────────
#   DATASETS         space-separated list of dataset names — choices: rat | nhp | spinal.
#                    Default: 'nhp spinal'.  Set to a single name to halve memory/time.
DATASETS=${DATASETS:-"nhp spinal"}

# ── Environment ───────────────────────────────────────────────────────────────
module load miniconda/3
conda activate pfns4neurostim

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs output/runs output/id_ood

# ── Phase 1 — Vanilla Benchmark ───────────────────────────────────────────────

if [ "$JOB" = "A1" ]; then
    CFG=${VANILLA_CONFIG:-configs/nhp_vanilla_benchmark.yaml}
    echo "[$(date)] A1 — vanilla fit (TabPFN v2 vs GP, inter-subject, n_reps=30) config=$CFG ${VANILLA_OVERRIDES:+overrides=$VANILLA_OVERRIDES}"
    # shellcheck disable=SC2086
    python src/vanilla_benchmark.py \
        --config "$CFG" \
        --mode fit \
        --save \
        $VANILLA_OVERRIDES

elif [ "$JOB" = "A2" ]; then
    CFG=${VANILLA_CONFIG:-configs/nhp_vanilla_benchmark.yaml}
    echo "[$(date)] A2 — vanilla optimization (BO regret, inter-subject, n_reps=30) config=$CFG ${VANILLA_OVERRIDES:+overrides=$VANILLA_OVERRIDES}"
    # shellcheck disable=SC2086
    python src/vanilla_benchmark.py \
        --config "$CFG" \
        --mode optimization \
        --save \
        $VANILLA_OVERRIDES

elif [ "$JOB" = "A3" ]; then
    CFG=${VANILLA_CONFIG:-configs/rat_vanilla_benchmark.yaml}
    echo "[$(date)] A3 — Rat vanilla benchmark (fit + optimization) config=$CFG ${VANILLA_OVERRIDES:+overrides=$VANILLA_OVERRIDES}"
    # shellcheck disable=SC2086
    python src/vanilla_benchmark.py \
        --config "$CFG" \
        --save \
        $VANILLA_OVERRIDES

elif [ "$JOB" = "A4" ]; then
    CFG=${VANILLA_CONFIG:-configs/nhp_vanilla_benchmark.yaml}
    echo "[$(date)] A4 — vanilla budget sweep (budgets=10 30 50 100 200) config=$CFG ${VANILLA_OVERRIDES:+overrides=$VANILLA_OVERRIDES}"
    # shellcheck disable=SC2086
    python src/vanilla_benchmark.py \
        --config "$CFG" \
        --mode optimization \
        --budgets 10 30 50 100 200 \
        --save \
        $VANILLA_OVERRIDES

# ── Phase 2 — ID/OOD Analysis ─────────────────────────────────────────────────

elif [ "$JOB" = "B1" ]; then
    # GPU job: bar-distribution entropy is one TabPFN forward pass per
    # subject/EMG and per synthetic dataset (500 of them).
    echo "[$(date)] B1 — Shannon Entropy datasets=$DATASETS"
    python src/id_ood_analysis.py \
        --datasets $DATASETS \
        --analyses entropy \
        --prior_source tabpfn_prior \
        --device cuda \
        --n_synthetic 500 \
        --n_context 100 \
        --save

elif [ "$JOB" = "B2" ]; then
    # CPU job: pure-NumPy MMD² + sliced Wasserstein-2 on (X, y) feature
    # vectors — no TabPFN inference. GPU would not help.
    echo "[$(date)] B2+B5 — MMD + Wasserstein-2 (joint 2-panel figure) datasets=$DATASETS"
    python src/id_ood_analysis.py \
        --datasets $DATASETS \
        --analyses mmd wasserstein \
        --prior_source tabpfn_prior \
        --device cpu \
        --n_synthetic 500 \
        --save

elif [ "$JOB" = "B3" ]; then
    # GPU job: extract_embeddings_frozen() runs a TabPFN forward pass
    # with a forward hook on transformer_encoder.layers.17 per dataset.
    echo "[$(date)] B3 — Mahalanobis Distance datasets=$DATASETS"
    python src/id_ood_analysis.py \
        --datasets $DATASETS \
        --analyses mahalanobis \
        --prior_source tabpfn_prior \
        --device cuda \
        --n_synthetic 500 \
        --n_context 50 \
        --save

elif [ "$JOB" = "B4" ]; then
    # GPU job: same hook pattern as B3, repeated per analyzed layer.
    echo "[$(date)] B4 — Layer-wise CKA (3-layer default: 4, 13, 17) datasets=$DATASETS"
    python src/id_ood_analysis.py \
        --datasets $DATASETS \
        --analyses cka \
        --prior_source tabpfn_prior \
        --device cuda \
        --n_synthetic 500 \
        --seed 42 \
        --save

elif [ "$JOB" = "B4dense" ]; then
    # GPU job: 10-layer sweep multiplies the embedding cost ~3x vs B4.
    echo "[$(date)] B4dense — Layer-wise CKA (dense 10-layer sweep: 0 2 4 6 8 10 12 14 16 17) datasets=$DATASETS"
    python src/id_ood_analysis.py \
        --datasets $DATASETS \
        --analyses cka \
        --cka_layers 0 2 4 6 8 10 12 14 16 17 \
        --prior_source tabpfn_prior \
        --device cuda \
        --n_synthetic 500 \
        --n_context 50 \
        --save

elif [ "$JOB" = "B6" ]; then
    # GPU job: same hook pattern as B4, RSA replaces CKA on the embeddings.
    echo "[$(date)] B6 — RSA (Representational Similarity Analysis) datasets=$DATASETS"
    python src/id_ood_analysis.py \
        --datasets $DATASETS \
        --analyses rsa \
        --prior_source tabpfn_prior \
        --device cuda \
        --n_synthetic 500 \
        --n_context 50 \
        --save

elif [ "$JOB" = "B7" ]; then
    # GPU job: budget sweep extracts embeddings at each (budget × dataset).
    echo "[$(date)] B7 — Procrustes BO-Trajectory (embedding geometry vs budget) datasets=$DATASETS"
    python src/id_ood_analysis.py \
        --datasets $DATASETS \
        --analyses procrustes \
        --prior_source tabpfn_prior \
        --device cuda \
        --proc_budgets 2 10 30 50 100 \
        --proc_layer 17 \
        --proc_n_synthetic 20 \
        --save

elif [ "$JOB" = "Bfull" ]; then
    # GPU job: combines all seven analyses into a single dashboard run.
    echo "[$(date)] B1–B7 full suite + summary dashboard datasets=$DATASETS"
    python src/id_ood_analysis.py \
        --datasets $DATASETS \
        --analyses entropy mmd wasserstein mahalanobis cka rsa procrustes \
        --prior_source tabpfn_prior \
        --device cuda \
        --n_synthetic 500 \
        --n_context 50 \
        --proc_budgets 2 10 30 50 100 \
        --proc_layer 17 \
        --proc_n_synthetic 20 \
        --save

else
    echo "ERROR: Unknown JOB='$JOB'. Valid values: A1 A2 A3 A4 B1 B2 B3 B4 B4dense B6 B7 Bfull" >&2
    exit 1
fi

echo "[$(date)] Done."
