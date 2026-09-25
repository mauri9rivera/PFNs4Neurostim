# TabPFN v1 as a regression surrogate — adaptation spec

**Question asked (2026-09-20):** *can TabPFN v1's classification head be adapted, and if so how?*

**Answer: yes. Implemented 2026-09-25** (sections 4-5); sections 1-3 are the specification it follows.
What blocked v1 was packaging, not method: v1 is `tabpfn<2`, which cannot coexist with the pinned
`tabpfn==6.3.2` used for TabPFN v2.5 because both occupy the module name `tabpfn`. It now has its own
environment, `environment.v1.yml`.

This document also states what the adaptation costs, because every v1 number must be reported as a
**classification-head adaptation** rather than as a like-for-like regression result.

---

## 1. What v1 gives us

TabPFN v1 is a **classifier**: `TabPFNClassifier.fit(X, y_labels)` and `predict_proba(X) -> [N, C]`,
with a hard architectural ceiling of **10 classes** and 100 features. There is no regression head
and no bar distribution — those arrive in v2.

The acquisition layer needs `predict_marginals(X) -> (mean, std)` and, for Thompson sampling,
per-site draws. So the adaptation has to turn `[N, C]` class probabilities into a distribution over
the response axis.

## 2. The adaptation

Already implemented, model-agnostically, in
[`models/pfn/bar_distribution.py`](../src/pfns4neurostim/models/pfn/bar_distribution.py) and
[`BucketizedClassifierSurrogate`](../src/pfns4neurostim/models/pfn/external.py):

1. **Bin the observed responses.** `quantile_borders(y, K)` places `K + 1` borders at equally spaced
   quantiles of the *observed context* `y`, so bins carry roughly equal mass and resolution follows
   the data rather than an arbitrary uniform grid. Tied values are nudged apart so every bin has
   positive width.
2. **Fit the classifier on bin labels.** `bar.digitize(y)` maps each observation to its bin index;
   `TabPFNClassifier.fit(X, labels)`.
3. **Read class probabilities as a bar distribution.** `predict_proba(X)` gives `[N, C]` over the
   classes the classifier actually saw; `bar.expand(probs, classifier.classes_)` re-expands that to
   all `K` bins (a context that never contained bin 7 yields no column for it).
4. **Summaries.** `bar.mean` integrates the piecewise-uniform density; `bar.std` adds the
   **within-bin variance** `width² / 12`, so a confident single-bin prediction still carries the
   spread its resolution implies instead of a false zero. `bar.quantile` interpolates inside the
   containing bin; `bar.sample` draws a bin by its probability then uniformly within it, which is
   what `ts_marginal` uses.

All of this is tested exactly in `tests/models/test_pfn_wrappers.py::TestBarDistribution`.

## 3. The one v1-specific constraint: K ≤ 10

v1 supports at most 10 classes, so `n_bins` is capped at 10 — versus 32 for TabFlex. This is the
adaptation's dominant cost and it must be stated in every caption:

* **Resolution floor.** No prediction can be sharper than one bin. With `K = 10` equal-mass bins, the
  best achievable predictive SD on a standardized response is roughly `range / (10 · √12) ≈ 0.03·range`,
  which sets a floor on NLL and an upper bound on how confident the model can look.
* **Consequence for calibration metrics.** v1's ECE and coverage are partly a property of `K`, not of
  the model. Report them next to the bin count, and do not rank v1 against TabPFN v2.5 on calibration
  as if the comparison were like-for-like.
* **Consequence for BO.** EI and UCB only need a mean and an SD, both of which survive binning, so the
  *optimization* comparison is fairer than the calibration one. This is the comparison to lead with.

`TabPFNv1Surrogate` therefore defaults to `n_bins=10` rather than the 32 used elsewhere.

## 4. Implementation (done 2026-09-25)

`TabPFNv1Surrogate` in [`models/pfn/wrappers.py`](../src/pfns4neurostim/models/pfn/wrappers.py) is a
`BucketizedClassifierSurrogate` with `MAX_CLASSES = 10` and `n_bins=10` by default; binning,
expansion, moments and sampling are inherited. Two hooks carry everything v1-specific:

```python
def _make_classifier(self):
    from tabpfn import TabPFNClassifier        # the v1 API
    return TabPFNClassifier(
        device=self.device,
        N_ensemble_configurations=self.n_ensemble_configurations,
        seed=self.seed,
        **self.backend_kwargs,
    )

def _fit_classifier(self, classifier, X, labels):
    classifier.fit(X, labels, overwrite_warning=True)
```

`_fit_classifier` is a new hook on the bucketized base (the other bucketized model, TabFlex, takes the
default `classifier.fit(X, labels)`), so v1's extra `fit` argument did not need a special case in the
shared fit path.

Checks that come with it:

* `n_bins > MAX_CLASSES` raises **at construction, before the backend import** — a config asking for an
  impossible resolution is wrong in every environment, so it must not need the right one to say so.
* `BarDistribution.expand` now raises when the probability matrix and `classes_` disagree in width.
  v1 truncates `predict_proba` to the classes the context contained and reports them in `classes_`, so
  the two agree; a backend that padded to a fixed class count would otherwise be silently misaligned
  onto the wrong bins.
* `seed` is an explicit constructor argument: it drives v1's feature/class permutation ensemble, so
  leaving it to upstream's default would make a repetition irreproducible if that default ever changed.

Two upstream properties worth knowing:

* v1 keeps loaded checkpoints in a class-level `models_in_memory` cache keyed by `(model, device)`, so
  constructing one classifier per BO step re-reads nothing from disk after the first.
* v1's checkpoint ships inside the wheel (`tabpfn/models_diff/`). There is no weight host that can
  disappear, unlike TabFlex (microsoft/ticl #27).

## 5. Environment

v1 cannot share an environment with v2.5, so it has its own: **`environment.v1.yml`** → the conda env
`pfns4neurostim-v1`. It mirrors `environment.yml` exactly (Python 3.9, numpy 1.26.4, torch 2.5.1, the GP
stack) and differs only in `tabpfn<2` replacing `tabpfn==6.3.2`; `pfns4bo` is left out because it pins
`scikit-learn<1.2`. The comparison must differ in the model, not in numpy.

```bash
bash scripts/mila_setup.sh env v1     # create pfns4neurostim-v1 and install the package
```

`ExternalSpec.env` records `main` / `bench` / `v1` / `mitra` per model, and `external.conda_env_for(key)`
resolves it, so no script names an environment by hand. The availability check refuses to pretend v1 is
present when only v2.5 is installed (`external._backend_ok` compares the installed major version) and its
message now names the environment to activate, so a v1 run in the wrong environment fails in seconds
instead of silently benchmarking v2.5 twice.

v1 joins the existing `configs/experiment/hyp0_pfn_bench_{nhp,5d_rat}.yaml` rather than getting its own
config: that run is already assembled from cells computed in several environments, and cell identity
carries the model version, not the environment.

```bash
CONDA_ENV=pfns4neurostim-v1 LANES=2 sbatch scripts/run_bo_benchmark.sh configs/experiment/hyp0_pfn_bench_nhp.yaml "models=[tabpfn_v1]"
```

Portfolio units **E7** (NHP) and **E8** (5d_rat) in `scripts/portfolio.py`; both costs are unmeasured, so
run E7 first and read its wall time.

The rejected alternative — vendoring `automl/TabPFN` under `libs/` at its v1 tag — needs a rename patch
to avoid the same module-name collision, and `libs/` is read-only by project rule.
