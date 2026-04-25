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
#  ── Phase 2 — ID/OOD Analysis (CPU-only, ~8h, 16G) ───────
#  Override the GPU/mem/time defaults by passing flags to sbatch:
#
#    JOB=B1 sbatch --gres='' --cpus-per-task=4 --mem=16G --time=8:00:00 scripts/run_all_experiments.sh
#    JOB=B2 sbatch --gres='' --cpus-per-task=4 --mem=16G --time=8:00:00 scripts/run_all_experiments.sh
#    JOB=B3 sbatch --gres='' --cpus-per-task=4 --mem=16G --time=8:00:00 scripts/run_all_experiments.sh
#    JOB=B4 sbatch --gres='' --cpus-per-task=4 --mem=16G --time=8:00:00 scripts/run_all_experiments.sh
#    JOB=B4dense sbatch --gres='' --cpus-per-task=4 --mem=16G --time=8:00:00 scripts/run_all_experiments.sh
#    JOB=B6 sbatch --gres='' --cpus-per-task=4 --mem=16G --time=8:00:00 scripts/run_all_experiments.sh
#    JOB=B7 sbatch --gres=gpu:rtx8000:1 --cpus-per-task=4 --mem=16G --time=8:00:00 scripts/run_all_experiments.sh
#    JOB=Bfull sbatch --gres=gpu:rtx8000:1 --cpus-per-task=4 --mem=16G --time=12:00:00 scripts/run_all_experiments.sh
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

# ── Environment ───────────────────────────────────────────────────────────────
module load miniconda/3
conda activate pfns4neurostim

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs output/runs output/id_ood

# ── Phase 1 — Vanilla Benchmark ───────────────────────────────────────────────

if [ "$JOB" = "A1" ]; then
    echo "[$(date)] A1 — NHP vanilla fit (TabPFN v2 vs GP, inter-subject, n_reps=30)"
    python src/vanilla_benchmark.py \
        --config configs/nhp_vanilla_benchmark.yaml \
        --mode fit \
        --save

elif [ "$JOB" = "A2" ]; then
    echo "[$(date)] A2 — NHP vanilla optimization (BO regret, inter-subject, n_reps=30)"
    python src/vanilla_benchmark.py \
        --config configs/nhp_vanilla_benchmark.yaml \
        --mode optimization \
        --save

elif [ "$JOB" = "A3" ]; then
    echo "[$(date)] A3 — Rat vanilla benchmark (fit + optimization)"
    python src/vanilla_benchmark.py \
        --config configs/rat_vanilla_benchmark.yaml \
        --save

elif [ "$JOB" = "A4" ]; then
    echo "[$(date)] A4 — NHP budget sweep (budgets=10 30 50 100 200)"
    python src/vanilla_benchmark.py \
        --config configs/nhp_vanilla_benchmark.yaml \
        --mode optimization \
        --budgets 10 30 50 100 200 \
        --save

# ── Phase 2 — ID/OOD Analysis ─────────────────────────────────────────────────

elif [ "$JOB" = "B1" ]; then
    echo "[$(date)] B1 — Shannon Entropy"
    python src/id_ood_analysis.py \
        --datasets nhp rat \
        --analyses entropy \
        --prior_source tabpfn_prior \
        --n_synthetic 500 \
        --n_context 100 \
        --seed 42 \
        --save

elif [ "$JOB" = "B2" ]; then
    echo "[$(date)] B2+B5 — MMD + Wasserstein-2 (joint 2-panel figure)"
    python src/id_ood_analysis.py \
        --datasets nhp rat \
        --analyses mmd wasserstein \
        --prior_source tabpfn_prior \
        --n_synthetic 500 \
        --seed 42 \
        --save

elif [ "$JOB" = "B3" ]; then
    echo "[$(date)] B3 — Mahalanobis Distance"
    python src/id_ood_analysis.py \
        --datasets nhp rat \
        --analyses mahalanobis \
        --prior_source tabpfn_prior \
        --n_synthetic 500 \
        --n_context 50 \
        --seed 42 \
        --save

elif [ "$JOB" = "B4" ]; then
    echo "[$(date)] B4 — Layer-wise CKA (3-layer default: 4, 13, 17)"
    python src/id_ood_analysis.py \
        --datasets nhp rat \
        --analyses cka \
        --prior_source tabpfn_prior \
        --n_synthetic 500 \
        --n_context 50 \
        --seed 42 \
        --save

elif [ "$JOB" = "B4dense" ]; then
    echo "[$(date)] B4dense — Layer-wise CKA (dense 10-layer sweep: 0 2 4 6 8 10 12 14 16 17)"
    python src/id_ood_analysis.py \
        --datasets nhp rat \
        --analyses cka \
        --cka_layers 0 2 4 6 8 10 12 14 16 17 \
        --prior_source tabpfn_prior \
        --n_synthetic 500 \
        --n_context 50 \
        --seed 42 \
        --save

elif [ "$JOB" = "B6" ]; then
    echo "[$(date)] B6 — RSA (Representational Similarity Analysis)"
    python src/id_ood_analysis.py \
        --datasets nhp rat \
        --analyses rsa \
        --prior_source tabpfn_prior \
        --n_synthetic 500 \
        --n_context 50 \
        --seed 42 \
        --save

elif [ "$JOB" = "B7" ]; then
    # B7 extracts transformer embeddings per BO budget step — benefits from GPU.
    # Keep --gres=gpu:rtx8000:1 when submitting (override --mem=16G --time=8:00:00).
    echo "[$(date)] B7 — Procrustes BO-Trajectory (embedding geometry vs budget)"
    python src/id_ood_analysis.py \
        --datasets nhp rat \
        --analyses procrustes \
        --prior_source tabpfn_prior \
        --device cuda \
        --proc_budgets 2 10 30 50 100 \
        --proc_layer 17 \
        --proc_n_synthetic 20 \
        --seed 42 \
        --save

elif [ "$JOB" = "Bfull" ]; then
    echo "[$(date)] B1–B7 full suite + summary dashboard"
    python src/id_ood_analysis.py \
        --datasets nhp rat \
        --analyses entropy mmd wasserstein mahalanobis cka rsa procrustes \
        --prior_source tabpfn_prior \
        --device cuda \
        --n_synthetic 500 \
        --n_context 50 \
        --proc_budgets 2 10 30 50 100 \
        --proc_layer 17 \
        --proc_n_synthetic 20 \
        --seed 42 \
        --save

else
    echo "ERROR: Unknown JOB='$JOB'. Valid values: A1 A2 A3 A4 B1 B2 B3 B4 B4dense B6 B7 Bfull" >&2
    exit 1
fi

echo "[$(date)] Done."
