"""
OOD Entropy Diagnostic Script
==============================
Investigates the bimodal sharp-peak pattern in the OOD noise baseline entropy
distribution. Tests two hypotheses:

  H1 — Noise type mixing: clustered / correlated / concentrated generators
       produce systematically different entropy levels, creating two sub-populations.

  H2 — Entropy computation bug: softmax(log_probs) ≠ exp(log_probs). The current
       implementation uses torch.softmax() on log-probability logits, which re-
       normalises an already-normalised distribution and distorts intermediate bins.

Usage (from repo root):
    conda activate pfns4neurostim
    python scripts/diagnose_ood_entropy.py
"""
import sys
import pickle
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Make src imports available
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis.synthetic_noise import generate_noise_dataset
from tabpfn import TabPFNRegressor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_DATASETS_PER_TYPE = 100   # datasets per noise type for the per-type histogram
N_CONTEXT           = 50    # must match what entropy_analysis() uses
N_FEATURES          = 2     # NHP: d=2
SEED                = 42
EXISTING_RESULTS    = (
    ROOT / "output/id_ood/id_ood_nhp_both_20260405/entropy/entropy_results.pkl"
)
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR             = ROOT / "output/id_ood/diagnostics/ood_entropy"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: entropy from logits using BOTH methods
# ---------------------------------------------------------------------------

def entropy_from_logits(logits: torch.Tensor, method: str) -> np.ndarray:
    """
    Args:
        logits: (n_test, n_bars) tensor — raw values from result['logits']
        method: 'softmax' (current code) or 'exp' (log-prob interpretation)
    Returns:
        (n_test,) numpy array of per-sample Shannon entropy (nats)
    """
    if method == "softmax":
        probs = torch.softmax(logits, dim=-1)
    elif method == "exp":
        probs = torch.exp(logits)          # correct if logits = log_softmax(raw)
    else:
        raise ValueError(f"Unknown method: {method}")

    log_probs = torch.log(probs + 1e-10)
    H = -(probs * log_probs).sum(dim=-1)
    return H.detach().cpu().numpy()


def get_raw_logits(model, X_ctx, y_ctx, X_tst):
    """Return raw logits tensor from TabPFN bar distribution."""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        model.fit(X_ctx, y_ctx)
        result = model.predict(X_tst, output_type="full")
    return result["logits"]  # (n_test, n_bars)


# ---------------------------------------------------------------------------
# Check 1 — Verify what 'logits' actually contains
# ---------------------------------------------------------------------------

def check_logits_format(model):
    """
    Determine whether result['logits'] contains raw logits or log-softmax
    outputs by checking whether exp(logits) sums to ~1 per row.

    If sum ≈ 1  → already softmax-normalised (log-probs): use exp().
    If sum ≫ 1  → raw logits: use softmax().
    """
    print("\n=== CHECK 1: Logit format ===")
    rng = np.random.RandomState(SEED)
    # Simple GP-like data (in-distribution reference)
    X = rng.rand(60, N_FEATURES).astype(np.float32)
    y = np.sin(3 * X[:, 0]).astype(np.float32)
    y = (y - y.mean()) / (y.std() + 1e-8)

    X_ctx, y_ctx = X[:N_CONTEXT], y[:N_CONTEXT]
    X_tst = X[N_CONTEXT:]

    logits = get_raw_logits(model, X_ctx, y_ctx, X_tst)  # (n_tst, n_bars)
    exp_sum  = logits.exp().sum(dim=-1)
    softmax_check = torch.softmax(logits, dim=-1).sum(dim=-1)

    print(f"  n_bars: {logits.shape[1]}")
    print(f"  exp(logits).sum(dim=-1) — mean: {exp_sum.mean():.6f}, std: {exp_sum.std():.6f}")
    print(f"  expected ≈ 1.0 if log-probs, ≫ 1 if raw logits")
    print(f"  softmax(logits).sum(dim=-1) — mean: {softmax_check.mean():.6f} (always 1)")
    print(f"  => logits are {'log-probabilities (use exp)' if abs(exp_sum.mean().item() - 1.0) < 0.01 else 'raw logits (use softmax)'}")

    max_theoretical_entropy = float(np.log(logits.shape[1]))
    print(f"  max theoretical entropy = log({logits.shape[1]}) = {max_theoretical_entropy:.4f} nats")

    return logits.shape[1]


# ---------------------------------------------------------------------------
# Check 2 — Per-noise-type entropy distributions
# ---------------------------------------------------------------------------

def _make_dataset(seed: int):
    """Generate one (X_ctx, y_ctx, X_tst) uniform noise dataset."""
    X, y = generate_noise_dataset(n_features=N_FEATURES, n_samples=100, seed=seed)
    X_ctx, y_ctx = X[:N_CONTEXT], y[:N_CONTEXT]
    X_tst = X[N_CONTEXT:]
    return X_ctx, y_ctx, X_tst


def check_per_type_entropy(model, n_bars: int):
    """
    Compute entropy for the uniform noise generator using both softmax and exp.
    Returns dict: {'uniform': {'softmax': np.ndarray, 'exp': np.ndarray}}
    """
    print("\n=== CHECK 2: Uniform noise entropy ===")
    max_H = np.log(n_bars)
    softmax_H, exp_H = [], []

    for i in range(N_DATASETS_PER_TYPE):
        X_ctx, y_ctx, X_tst = _make_dataset(seed=SEED + i)
        if len(X_tst) == 0:
            continue
        logits = get_raw_logits(model, X_ctx, y_ctx, X_tst)
        softmax_H.append(entropy_from_logits(logits, "softmax"))
        exp_H.append(entropy_from_logits(logits, "exp"))

    softmax_H = np.concatenate(softmax_H)
    exp_H     = np.concatenate(exp_H)
    results = {"uniform": {"softmax": softmax_H, "exp": exp_H}}
    print(f"  {'uniform':12s} | softmax: mean={softmax_H.mean():.3f}  std={softmax_H.std():.4f}  "
          f"frac@max={np.mean(softmax_H >= 0.99 * max_H):.2f} | "
          f"exp: mean={exp_H.mean():.3f}  std={exp_H.std():.4f}  "
          f"frac@max={np.mean(exp_H >= 0.99 * max_H):.2f}")
    return results


# ---------------------------------------------------------------------------
# Check 4 — Compare softmax vs exp entropy on existing results
# ---------------------------------------------------------------------------

def check_existing_results(n_bars: int):
    """
    Load the pre-computed entropy results and print noise summary statistics.
    This verifies whether the existing 'noise' entropy already shows bimodality.
    """
    print("\n=== CHECK 4: Existing entropy results (noise bank) ===")
    if not EXISTING_RESULTS.exists():
        print(f"  File not found: {EXISTING_RESULTS}")
        return

    with open(EXISTING_RESULTS, "rb") as f:
        data = pickle.load(f)

    max_H = np.log(n_bars)
    noise_H = data.get("noise", np.array([]))
    if len(noise_H) == 0:
        print("  'noise' key empty or missing.")
        return

    # Simple bimodality test: are there two modes?
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(noise_H, bw_method=0.1)
    x_grid = np.linspace(noise_H.min(), noise_H.max(), 500)
    density = kde(x_grid)
    # Count local maxima
    n_peaks = int(np.sum(
        (density[1:-1] > density[:-2]) & (density[1:-1] > density[2:])
    ))

    print(f"  noise entropy: n={len(noise_H)}  mean={noise_H.mean():.4f}  "
          f"std={noise_H.std():.4f}  min={noise_H.min():.4f}  max={noise_H.max():.4f}")
    print(f"  fraction at ≥99% of max entropy ({max_H:.3f}): "
          f"{np.mean(noise_H >= 0.99 * max_H):.3f}")
    print(f"  KDE local peaks detected: {n_peaks}  "
          f"{'=> bimodal (supports H1)' if n_peaks >= 2 else '=> unimodal'}")

    # Print the percentile range of each mode area (rough)
    if n_peaks >= 2:
        peak_locs = x_grid[1:-1][
            (density[1:-1] > density[:-2]) & (density[1:-1] > density[2:])
        ]
        for loc in peak_locs:
            mask = np.abs(noise_H - loc) < 0.1
            print(f"    peak near H={loc:.3f}: {mask.sum()} samples")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(per_type: dict, n_bars: int):
    max_H = np.log(n_bars)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    vals = per_type["uniform"]
    h_sm = vals["softmax"]
    h_ex = vals["exp"]

    # Left: entropy histogram
    axes[0].hist(h_sm, bins=30, color="#0072B2", alpha=0.8, density=True)
    axes[0].axvline(max_H, color="red", ls="--", lw=1.5, label=f"max H={max_H:.2f}")
    axes[0].set_title("Uniform noise entropy\n(softmax method)")
    axes[0].set_xlabel("Shannon entropy (nats)")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)

    # Right: softmax vs exp scatter
    axes[1].scatter(h_sm, h_ex, s=4, alpha=0.4, color="#0072B2")
    lim = max(h_sm.max(), h_ex.max()) * 1.05
    axes[1].plot([0, lim], [0, lim], "k--", lw=1, label="y=x")
    axes[1].set_xlabel("entropy via softmax(logits)  [current code]")
    axes[1].set_ylabel("entropy via exp(logits)  [if logits=log_probs]")
    axes[1].set_title("H(softmax) vs H(exp)\nlogit format diagnostic")
    axes[1].legend(fontsize=8)

    fig.suptitle("OOD Noise Entropy Diagnostic", fontsize=13, fontweight="bold")
    out_path = OUT_DIR / "ood_entropy_diagnostic.svg"
    fig.savefig(out_path, bbox_inches="tight")
    out_path_png = OUT_DIR / "ood_entropy_diagnostic.png"
    fig.savefig(out_path_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nDiagnostic plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    model = TabPFNRegressor(device=DEVICE)

    n_bars = check_logits_format(model)
    per_type = check_per_type_entropy(model, n_bars)
    check_existing_results(n_bars)
    plot_results(per_type, n_bars)

    print("\n=== SUMMARY ===")
    print("If softmax vs exp entropies differ substantially (non-diagonal scatter)")
    print("=> logit format bug confirmed (should use exp, not softmax).")
    print(f"\nAll outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
