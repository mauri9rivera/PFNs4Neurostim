# PFNs4Neurostim

## Overview

Current Gaussian Process Bayesian Optimization (GPBO) applications in neurostimulation are limited by GP's cubic time complexity, which hinders scalability for real-time and large-scale experiments. This project evaluates whether Prior-data Fitted Networks (PFNs) can serve as viable replacements by benchmarking their performance on neurostimulation tasks across multiple animal modalities (non-human primates, rats, and spinal cord models) and varying input dimensions.

We address two key research questions:

1. **Approximation**: Can the PFN accurately approximate EMG responses from neural stimulation data?
2. **Optimization**: Can the PFN perform Bayesian Optimization experiments on par with GP?

## Quick start

```bash
pip install -e .
python -m pfns4neurostim bo_benchmark --config configs/experiment/hyp_a_nhp.yaml
python -m pfns4neurostim stress_sweep --config configs/experiment/stress_k2_channel_nhp.yaml
```

Every runner writes a tidy `tidy.csv`, a trajectory pickle, the resolved `config.yaml` and its
figures into `output/<experiment>/<dataset>/`. Figures regenerate from the CSV alone:

```bash
python -m pfns4neurostim stress_sweep --config configs/experiment/stress_k2_channel_nhp.yaml --replot
```

On the Mila cluster, `scripts/mila_setup.sh` creates the storage layout (code on `$HOME`, data
master on `$ARCHIVE`, working copies on `$SCRATCH`) and `scripts/run_stress_sweep.sh` submits a
sweep, staging the dataset to `$SLURM_TMPDIR` first.

```bash
sbatch scripts/run_stress_sweep.sh configs/experiment/stress_k2_channel_nhp.yaml
```

## Repository layout

The code is a single installable package, `src/pfns4neurostim/`: `data/` (channels, splits,
stress knobs), `models/` (GP and PFN surrogates + registry), `acquisition/` (EI, UCB, PI,
Thompson, greedy, random), `evaluation/` (BO loop, metrics, robustness statistics),
`experiments/` (one runner per experiment type), `visualization/` (`style.py` is the single
source of figure style), `analysis/` (Hypothesis C), and `legacy_code/` (superseded code kept
for reproducing earlier results). See CLAUDE.md section 5 for the full tree.

## Installation

### 1. Clone the repository with submodules

```bash
git clone --recurse-submodules https://gitlab.com/scineurotech-group/PFNs4Neurostim.git
cd PFNs4Neurostim
```

If you already cloned without submodules, initialize them with:

```bash
git submodule init
git submodule update
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate pfns4neurostim
```

### 3. GPU Support (Optional)

The environment is configured for CUDA 11.8. If you have a compatible NVIDIA GPU, computations will automatically use GPU acceleration. CPU-only execution is also supported.

### 4. Data

The neural datasets are available upon request. Please contact mauricio.rivera@umontreal.ca to obtain access to the data files.

### 5. Running the project

```bash
python src/main.py
```

## Project Structure

```
PFNs4Neurostim/
├── src/
│   ├── main.py              # Main entry point and evaluation functions
│   ├── models/
│   │   └── gaussians.py     # GPyTorch ExactGP implementation
│   └── utils/
│       ├── data_utils.py    # Data loading and preprocessing
│       ├── gpbo_utils.py    # GPBO helper functions
│       └── visualization.py # Plotting and result visualization
├── libs/                    # External PFN libraries (git submodules)
├── data/                    # Neural stimulation datasets
├── environment.yml          # Conda environment specification
└── LICENSE                  # MIT License
```

### Key Functions in `main.py`

| Function | Description |
|----------|-------------|
| `evaluate_fit()` | Benchmarks model fitting quality by measuring R² scores on mean response of EMG maps |
| `evaluate_optimization()` | Tests Bayesian Optimization performance |
| `run_bo_loop()` | Performs the BO learning loop with EI-based acquisition |
| `fit_budget()` | Sweeps multiple budget sizes and plots R² curves with confidence intervals |
| `optimization_budget()` | Compares regret metrics (final simple regret or cumulative regret) across budgets |
| `main()` | Main runner that orchestrates experiments across datasets and models |
