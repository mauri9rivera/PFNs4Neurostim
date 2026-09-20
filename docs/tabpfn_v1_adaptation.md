# TabPFN v1 as a regression surrogate — adaptation spec

**Question asked (2026-09-20):** *can TabPFN v1's classification head be adapted, and if so how?*

**Answer: yes, and the adaptation is already implemented generically.** What blocks TabPFN v1 is
packaging, not method: v1 is `tabpfn<2`, which cannot coexist with the pinned `tabpfn==6.3.2` used
for TabPFN v2.5 because both occupy the module name `tabpfn`. It needs its own environment.

This document specifies the adaptation precisely enough to implement in one sitting once that
environment exists, and states what the adaptation costs, because every v1 number must be reported
as a **classification-head adaptation** rather than as a like-for-like regression result.

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

## 4. What remains to implement

Only `_make_classifier`:

```python
def _make_classifier(self):
    from tabpfn import TabPFNClassifier        # the v1 API
    return TabPFNClassifier(device=self.device, N_ensemble_configurations=3)
```

plus, in v1, `fit(X, y, overwrite_warning=True)` if the context exceeds its soft limits. Everything
else — binning, expansion, moments, sampling — is inherited.

## 5. Environment

v1 cannot share an environment with v2.5. Two options:

| Option | Command | Trade-off |
|---|---|---|
| **Separate env (recommended)** | `conda create -n pfns4neurostim-v1 python=3.9 && pip install 'tabpfn<2' && pip install -e .` | Clean; v1 rows are produced by a separate run and merged into the tidy CSV by `experiments/aggregate.py` |
| Vendored fork under `libs/` | submodule `automl/TabPFN` at its v1 tag, imported under a different top-level name | Avoids the env split but needs a rename patch, and `libs/` is read-only by project rule |

The availability check already refuses to pretend v1 is present when only v2.5 is installed
(`external._backend_ok` compares the installed major version), so a v1 run in the wrong environment
fails loudly instead of silently benchmarking v2.5 twice.
