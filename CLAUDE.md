# CLAUDE.md — PFNs4Neurostim

## Session Start Protocol

The following files are imported, with these conditions:

@.claude/task_plan.md, automatically:

@.claude/roadmap.md, only when you are specified to ultrathink

@.claude/research_design.md, when the user explicitely specifies it.

---

## 1. Project Overview

**PFNs4Neurostim** evaluates whether Tabular Prior-data Fitted Networks (TabPFNs) can replace
Gaussian Process Bayesian Optimization (GPBO) in neurostimulation applications, where GP's
O(n³) complexity hinders real-time and large-scale use.

**Two evaluation axes:**
1. **Prediction quality** — R² score across subjects/channels (fit task)
2. **Bayesian optimization** — cumulative regret vs query budget (optimization task)

**Animal modalities:** rat, non-human primate (NHP), spinal cord

**ID/OOD analysis:** determine whether neurostimulation datasets fall within TabPFN's
pretraining distribution using entropy, MMD, Mahalanobis, CKA, and Wasserstein divergence
metrics.

**Conference target:** TBD

---

## 2. Role & Context

- **Role:** Senior Research Engineer preparing code for a top-tier conference submission
  (NeurIPS/ICLR/CVPR)
- **Goal:** Clean, reproducible, readable code that other researchers can clone, run, and extend
- **Priority order:** Reproducibility > Readability > Performance

---

## 3. Tech Stack & Environment

| Component | Version |
|-----------|---------|
| Python | 3.9.25 |
| PyTorch | 2.5.1 (CUDA 11.8) |
| TabPFN | 6.3.2 |
| pfns4bo | 0.1.5 |
| NumPy | 1.26.4 |
| SciPy | 1.13.1 |
| pandas | 2.3.3 |
| scikit-learn | 1.6.1 |
| Matplotlib | 3.9.2 |
| Seaborn | 0.13.2 |

**Environment:** `conda activate pfns4neurostim` (see `environment.yml`)

**Experiment tracking:** Local CSV + pickle files under `output/runs/<tag>/`. No W&B or MLflow.

---

## 4. Coding Conventions

### Type Hints
- **All new code must have strict type hints** on both parameters and return values.
- Existing code does not need to be retrofitted unless you are substantially modifying a function.

```python
# Correct — new function
def evaluate_r2(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_reps: int = 30,
) -> dict[str, float]:
    ...

# Incorrect — missing parameter types
def evaluate_r2(predictions, targets, n_reps=30):
    ...
```

### Docstrings
- Use **Google-style** docstrings for all new and modified functions.
- If a function implements a specific equation from the paper, reference it:
  `"Implements Eq. 4 from Section 3.2"`.

```python
def run_bo_loop(
    model: TabPFNRegressor,
    dataset: np.ndarray,
    budget: int = 100,
) -> dict[str, list[float]]:
    """Run a Bayesian optimization loop using UCB acquisition.

    Args:
        model: Fitted TabPFN regressor used as surrogate.
        dataset: Full search space, shape [N, D+1] (D features + 1 target).
        budget: Number of sequential queries.

    Returns:
        Dict with keys 'regret', 'timing', 'best_found'.
    """
```

### Tensor Shape Annotations
- Annotate tensor shapes in comments on all tensor operations.

```python
x = torch.tensor(data, dtype=torch.float32)  # [N, D]
x = x.unsqueeze(1)                            # [N, 1, D]
```

### Random Seeds
- Always set **all three** random seeds at script/function entry when reproducibility matters.
- Default seed: `42`.

```python
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across torch, numpy, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

### Fail Fast on NaN/Inf
- Never silently swallow NaN or Inf values. Raise immediately with a descriptive message.

```python
if torch.isnan(loss):
    raise RuntimeError(
        f"NaN loss at epoch {epoch}. Last grad norm: {grad_norm:.4f}"
    )
```

---

## 5. Architecture & File Structure

```
PFNs4Neurostim/
├── CLAUDE.md                    ← This file
├── environment.yml              ← Conda environment (pfns4neurostim)
├── configs/                     ← Canonical YAML experiment configs
│   ├── nhp_fit.yaml
│   ├── nhp_optimization.yaml
│   └── rat_optimization.yaml
├── src/
│   ├── finetuning.py            ← CLI entry: finetune_tabpfn(), run_experiment()
│   ├── evaluation.py            ← Core eval: gp_baseline, finetuned_fit/optimization, budget sweeps
│   ├── vanilla_benchmark.py     ← Vanilla TabPFN v2 vs GP benchmark (Hypothesis A)
│   ├── aggregate.py             ← Post-hoc aggregation CLI: --config <yaml> → output/aggregated/
│   ├── id_ood_analysis.py       ← ID/OOD analysis CLI entry
│   ├── models/
│   │   ├── regressors.py        ← GradientMonitoredRegressor, extract_inference_model
│   │   ├── gaussians.py         ← ExactGP (gpytorch)
│   │   └── lora.py              ← LoRA parameter-efficient finetuning
│   ├── utils/
│   │   ├── bo_loops.py          ← run_gpbo_loop, run_finetunedbo_loop
│   │   ├── data_utils.py        ← Data loading, augmentation, split constants
│   │   ├── gpbo_utils.py        ← UCB acquisition function
│   │   ├── query_transforms.py  ← Data transforms (ZScore, MinMax, YeoJohnson, BoxCox)
│   │   └── visualization.py     ← All publication figures
│   └── analysis/
│       ├── id_ood.py            ← run_id_ood_analysis() — statistical divergence metrics
│       ├── id_ood_visualization.py
│       ├── synthetic_gp.py
│       ├── synthetic_noise.py
│       └── synthetic_tabpfn_prior.py
├── data/
│   ├── cortical/
│   ├── monkeys/
│   ├── rat/
│   └── spinal/
├── output/
│   ├── runs/<tag>/              ← diagnostics/, fitness/, optimization/, results/
│   └── aggregated/<dataset>-<family>/  ← CSVs + plots from aggregate.py
├── libs/                        ← Git submodules — READ-ONLY, never modify
│   ├── PFNs/
│   ├── PFNs4BO/
│   └── tabpfn-v1-prior/
└── scripts/
    ├── run_experiment.sh
    └── export_results.sh
```

### Dependency Graph (no cycles)

```
tabpfn (external)
  └─▶ models/regressors.py
        └─▶ utils/bo_loops.py
              └─▶ evaluation.py
                    └─▶ finetuning.py          (CLI entry)
                    └─▶ vanilla_benchmark.py   (CLI entry, Hypothesis A)
      models/gaussians.py    ──▶ evaluation.py
      utils/data_utils.py    ──▶ evaluation.py, finetuning.py, vanilla_benchmark.py, aggregate.py
      utils/visualization.py ──▶ evaluation.py, finetuning.py, vanilla_benchmark.py, aggregate.py
      utils/gpbo_utils.py    ──▶ utils/bo_loops.py
      utils/query_transforms.py ──▶ (pending integration)
      aggregate.py           ──▶ utils/data_utils.aggregate_results(), utils/visualization (post-hoc)
```

### Split Constants (`data_utils.py`)

```python
HELD_OUT_SUBJECTS = {'rat': [0, 5], 'nhp': [1]}
TRAIN_SUBJECTS    = {'rat': [1, 2, 3, 4], 'nhp': [0, 3]}
ALL_SUBJECTS      = {'rat': [0, 1, 2, 3, 4, 5], 'nhp': [0, 1, 3]}
# NHP subject 2 excluded — pure noise signal
```

---

## 6. Config Pattern

All canonical experiment hyperparameters live in `configs/` as YAML files.

**Naming convention:** `{dataset}_{mode}[_{split}].yaml`

All four CLI scripts support `--config <path>`. YAML keys are loaded as defaults;
any CLI flag that is explicitly provided overrides the YAML value.

**Each YAML must have a `family:` key** — this is used by `aggregate.py` to identify
which run directories belong to a given experiment family.

```bash
# Use canonical config
python src/finetuning.py --config configs/nhp_optimization.yaml

# Override a single parameter at runtime
python src/finetuning.py --config configs/nhp_optimization.yaml --epochs 100
```

**Do not hardcode hyperparameter values in function bodies.** All values must be reachable
via function arguments, CLI flags, or YAML config keys.

---

## 7. Plotting & Figures

- **Format:** SVG (primary, vector) + PNG (fallback, 300 dpi minimum)
- **Colorblind-friendly:** use `seaborn.color_palette("colorblind")` or `"tab10"`
- **Axes:** always label with units (e.g., `"R² Score"`, `"Cumulative Regret"`, `"Query Budget"`)
- **LaTeX rendering:** only when explicitly requested by the user
  (`plt.rc('text', usetex=True)`)
- **Fonts:** TrueType (Type 1) by default

---

## 8. Negative Constraints ("Do Not" Rules)

| Rule | Rationale |
|------|-----------|
| No hardcoded absolute paths | Breaks reproducibility on other machines |
| No magic numbers in function bodies | Must be traceable to a config or CLI flag |
| No Jupyter-style global state | Scripts must be self-contained executables |
| No silent NaN/Inf | Fail immediately with a diagnostic message |
| Never modify `libs/` submodules | External dependencies pinned by git |
| Never commit `data/` or `output/` | See `.gitignore` |
| No backwards-compatibility shims | Remove or replace cleanly; don't wrap |

---

## 9. Testing Strategy

**Framework:** `pytest`

**`[unit-test]` flag** — delegate to the **test-generator** subagent. Pass the file(s) under test; the agent investigates and generates tests independently.

**Shadow testing:** For non-trivial implementations, run a quick smoke test in the background
(`run_in_background=True`) before reporting success.

**Integration test baseline:** A full pipeline run using a small synthetic dataset
(`n=20`, `epochs=2`) to confirm end-to-end execution without errors.

**Test file location:** `tests/` at project root (to be created).

---

## 10. Knowledge Transfer & Context Engineering

**`[take-notes]` flag** — delegate to the **note-taker** subagent. The agent reads the conversation and updates the relevant documentation files independently.


