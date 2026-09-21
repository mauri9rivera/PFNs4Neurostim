# CLAUDE.md — PFNs4Neurostim

## Session Start Protocol

@.claude/task_plan.md

The following files are loaded on demand:
- `.claude/roadmap.md` — load when ultrathink is specified
- `.claude/research_design.md` — load when the user explicitly requests it
- `.claude/research_design_log.md` — dated design decisions (formerly research_design.md §4); load alongside research_design.md when decision history matters

---

## 1. Project Overview

**PFNs4Neurostim** evaluates whether Tabular Prior-data Fitted Networks (TabPFNs) can replace
Gaussian Process Bayesian Optimization (GPBO) in neurostimulation applications, where GP's
O(n³) complexity hinders real-time and large-scale use.

**Evaluation axis:**
- **Bayesian optimization** — cumulative regret vs query budget (optimization
  task). R² of the surrogate's final prediction (after the BO loop) is reported
  as a secondary metric inside the same run; there is no standalone "fit task"
  (random-subset R² evaluation was removed on 2026-04-27).
- **Headline acquisition function is `ts_marginal`** (as of 2026-09-20): independent per-site
  Thompson draws from each model's predictive marginals, symmetric for GP and PFN. `ts_joint` (exact
  joint draw) is a GP-only reference. EI, UCB (`auto_dim` schedule and the fixed-kappa grid) and the
  rest are comparison rows of the formal acquisition experiment (`configs/experiment/hyp0_acq_table_*`),
  to be run later. The legacy name `acq_fn: ts` no longer exists; `kappa_*` keys apply to `ucb` only.

**ID/OOD reference policy:** the in-distribution reference is the TabPFN prior *bag*
(`prior_source: prior_bag`/`tabpfn_prior` — the GP+MLP mixture the network is pretrained
on), NOT the GP-only `gp_bag`. Noise is always retained as the OOD anchor.

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

### Preprocessing Contract
- **X:** MinMax to `[0, 1]^D` for every model (GP lengthscale priors assume O(1) inputs; `5d_rat` axes have different physical units).
- **y:** z-scored, scaler fitted on valid trials only (GP priors assume standardized targets; TabPFN standardizes internally).
- Regret is divided by the ground-truth range, so it is **independent of the y scaler**.
- Implemented natively in `data/preprocessing.py` (`NORMALIZATIONS`, default `pfn`); selected by `dataset.normalization`, recorded in `config.yaml` and the tidy `normalization` column. `data/legacy_io.py` is frozen and only used as a numerical reference in tests.

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

Everything is one installable package: `pip install -e .` then
`python -m pfns4neurostim <experiment> --config <yaml>`. The flat `src/` tree is gone
(task #1, migrated 2026-09-18).

```
PFNs4Neurostim/
├── CLAUDE.md  README.md  LICENSE
├── pyproject.toml               ← package metadata + optional extras for the Hyp 0 models
├── environment.yml              ← cross-platform conda env (Windows dev + Mila cluster)
├── configs/
│   ├── dataset/                 ← nhp · 5d_rat · spinal
│   ├── model/                   ← tabpfn_v2_5 · gp_mll · gp_naive · random
│   ├── acquisition/             ← ei · ucb · pi · ts_marginal · ts_joint · greedy · random
│   ├── experiment/              ← hyp_a_nhp · hyp0_acq_table_nhp · stress_k2_{nhp,5d_rat}
│   └── legacy/                  ← the 24 pre-restructure YAMLs
├── data/  output/  logs/        ← gitignored; symlinks into $SCRATCH on the cluster
├── libs/                        ← git submodules, READ-ONLY (PFNs, PFNs4BO, tabpfn-v1-prior)
├── scripts/                     ← mila_setup.sh · run_stress_sweep.sh · run_*_experiments.sh
├── tests/                       ← mirrors the package (+ integration/, shadow/)
└── src/pfns4neurostim/
    ├── __main__.py              ← CLI dispatch: bo_benchmark | stress_sweep
    ├── config.py                ← YAML group composition, validation, --set, resolved dump
    ├── seeding.py               ← set_seed, rng_for/seed_for (per-cell reproducibility)
    ├── data/
    │   ├── channels.py          ← ChannelData (the one object every experiment consumes)
    │   ├── splits.py            ← HELD_OUT / TRAIN / ALL subjects
    │   ├── stress.py            ← StressKnob ABC + registry; K2 live, K1/K5/K6/K7 declared
    │   ├── snr.py               ← achieved SNR (dB) — the canonical K2 x-axis
    │   ├── synthetic_neurostim.py ← Demo 1 generator (placeholder)
    │   ├── references/          ← prior bag + noise banks for the Hyp C placement analysis
    │   └── legacy_io.py         ← pre-restructure loader; being carved into loaders/preprocessing
    ├── models/
    │   ├── protocol.py          ← SurrogateModel + SurrogateAdapter (+ LegacySurrogateModel)
    │   ├── registry.py          ← name → constructor + version string (P0.1)
    │   ├── gp/                  ← exact_gp.py · surrogates.py (MLL, naive, deep-kernel)
    │   ├── pfn/                 ← tabpfn.py · bar_distribution.py · external.py · wrappers.py
    │   └── baselines/           ← random_search.py
    ├── acquisition/             ← base · registry · schedules · thompson (P0.2 schema)
    ├── evaluation/
    │   ├── bo_loop.py           ← model- and acquisition-agnostic loop
    │   ├── bo_runner.py         ← one instrumented repetition → tidy metrics
    │   ├── metrics.py           ← range-normalized regrets, R², top-k, calibration
    │   ├── robustness.py        ← breakdown point, degradation AUC, CVaR, relative robustness
    │   ├── stats.py             ← TOST, equivalence margins, bootstrap
    │   └── results.py           ← tidy schema + run-dir I/O
    ├── experiments/             ← bo_benchmark.py (Hyp 0/A) · stress_sweep.py (Hyp B)
    ├── analysis/                ← cka.py · id_ood.py · surface_geometry.py  (Hyp C)
    ├── visualization/           ← style.py (single source of style) · bo · stress · mechanism
    └── legacy_code/             ← superseded CLIs, finetuning/LoRA, old loop and plotting
```

### Dependency Graph (no cycles)

```
data/{channels,splits,stress,snr}
  └─▶ models/{protocol,registry,gp,pfn,baselines}
        └─▶ acquisition/{base,registry,schedules,thompson}
              └─▶ evaluation/{bo_loop,bo_runner,metrics,robustness,stats,results}
                    └─▶ experiments/{bo_benchmark,stress_sweep}   (CLI via __main__)
                          └─▶ visualization/{style,bo,stress}
analysis/*  ──▶ visualization/mechanism.py
legacy_code/*  — imports the package, never the reverse
```

`visualization/style.py` is imported by every figure module and imports nothing from the
package, so it can never introduce a cycle.

### Split Constants (`data/splits.py`)

```python
HELD_OUT_SUBJECTS = {'rat': (0, 5), 'nhp': (1,), 'spinal': (0, 2, 5, 9), '5d_rat': (1, 4, 5)}
TRAIN_SUBJECTS    = {'rat': (1, 2, 3, 4), 'nhp': (0, 3), ...}
ALL_SUBJECTS      = {'rat': (0, 1, 2, 3, 4, 5), 'nhp': (0, 1, 3), ...}
# NHP subject 2 excluded — pure noise signal
```

**Policy change (2026-09-20): there is no train/held-out split any more.** Every valid subject
(`ALL_SUBJECTS`: NHP without subject 2, `5d_rat` without the subject-6 placeholder) is evaluated and the
dataset configs list them all. Because choices made by looking at results (primary acquisition, knob
ranges, equivalence margins) are then made on the data that is reported, pre-register the decision rule
in the config header and always report the full comparison table next to the chosen setting. The
constants above are kept only as a regression reference.

---

## 6. Config Pattern

Experiment YAMLs **compose config groups** through a `defaults:` block and may override any
resolved key inline. Groups live in `configs/{dataset,model,acquisition}/`.

```yaml
defaults:
  dataset: nhp
  model: [tabpfn_v2_5, gp_mll, gp_naive]
  acquisition: ei          # or a list, to sweep types (bo_benchmark)

knob:                      # stress sweeps only
  type: k2_snr
  levels: [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

budget: 50                 # TOTAL queries including n_init (P0.3)
n_init: 5
n_reps: 5
equivalence_margin: 0.05   # pre-registered, in range-normalized regret units
```

**Acquisition schema (P0.2).** `acquisition: {type, params, schedules}`. `params` holds only
the parameters that type declares — an unknown key raises at load time, it is never ignored —
and `schedules` anneals a named parameter over the BO steps
(`kind ∈ constant | linear | cosine | auto_dim`). The registry in
`acquisition/registry.py` is the single definition; `config.py` delegates to it.

**Every run writes its resolved config** to `<run_dir>/config.yaml`, including `model_version`
(P0.1) and the verbatim acquisition block (P0.2), so a result is always traceable to the exact
settings that produced it.

```bash
pip install -e .
python -m pfns4neurostim stress_sweep --config configs/experiment/stress_k2_nhp.yaml
python -m pfns4neurostim bo_benchmark --config configs/experiment/hyp_a_nhp.yaml --set n_reps=2
python -m pfns4neurostim stress_sweep --config configs/experiment/stress_k2_nhp.yaml --replot
pytest tests -m "not slow and not gpu and not legacy" -q
```

`--set` takes dotted keys (`--set dataset.emgs=[0] budget=30`) and **refuses to invent keys**,
so a typo fails instead of silently running something else. `--replot` rebuilds every figure
and table from the run's `tidy.csv` alone — no experiment re-run.

**Do not hardcode hyperparameter values in function bodies.** All values must be reachable via
function arguments, CLI flags, or YAML config keys.

---

## 7. Plotting & Figures

**All figure style decisions live in one module: `src/pfns4neurostim/visualization/style.py`.**
Every figure-producing function — and every agent writing one — must import geometry, colours,
model labels and axis strings from it (`figure`, `save_figure`, `model_palette`, `plot_kwargs`,
`axis_label`, `KNOB_LABELS`) and must never hardcode a colour, figsize, font size, model name
or axis string. Settled with the user on 2026-09-18 for the JNE/IOP target:

- **Format:** SVG only by default (`save_figure(..., formats=("svg",))`); PNG at 600 dpi and PDF opt-in
- **Geometry:** single-column default (3.27 in / 8.3 cm); width tokens `single` / `onehalf` / `double`
- **Typography:** Arial → Helvetica → DejaVu Sans; base 9 pt, ticks 8 pt, panel letters 10 pt bold;
  TrueType/Type-42 (`pdf.fonttype=42`, `svg.fonttype="none"`)
- **Palette:** two hardcoded anchors, everything else derived (`style.ANCHOR_PFN` `#0072B2` = TabPFN-2.5,
  `style.ANCHOR_GP` `#D55E00` = GP-MLL; `NEUTRAL_GREY` for non-learning baselines). Other PFNs/GPs are shades of
  their family anchor via `derive_shade` (hue held, lightness walks a bounded band), and `acquisition_style(model, acq)`
  shades a model's colour by acquisition while linestyle carries the acquisition. Never add a hex literal for a model
- **Model names:** compact code-like — `TabPFN-2.5`, `GP-MLL`, `GP-fixed`, `GP-oracle`, `Random`
- **Stress vocabulary:** roadmap jargon is canonical in code, figures *and* text — `knob`, `level`,
  `K1`/`K2`/`K5`/`K6`, `Demo 1 (synthetic)` / `Demo 2 (in vivo)`, `breakdown point`; K2's x-axis is
  always **achieved SNR (dB)**, never the raw α
- **Axes:** always label with units, via `style.axis_label(<column name>)`
- **LaTeX rendering:** only when explicitly requested by the user (`plt.rc('text', usetex=True)`);
  mathtext is used by default

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
| No multi-line CLI commands | All flags on one line; never use `\` continuations — breaks copy-paste on Windows |

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

@.claude/skills/take-notes.md

**`[take-notes]` flag** — invokes the note-taker skill inline (no subagent spawned).
Claude reads the current conversation and updates the relevant documentation files directly.


