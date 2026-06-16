"""Quick smoke test for TOST and gradient-share utilities."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from utils.stats import tost_equivalence, compute_equivalence_margin, run_tost_on_results
from utils.visualization import plot_gradient_share, plot_tost_forest

np.random.seed(42)

# --- TOST: equivalent case ---
a = np.random.normal(0.5, 0.05, 30)
b = np.random.normal(0.52, 0.05, 30)
r = tost_equivalence(a, b, margin=0.10)
assert r['equivalent'], f"Expected equiv, got tost_p={r['tost_p']:.4f}"
print(f"[PASS] equiv:   tost_p={r['tost_p']:.4f} CI=[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]")

# --- TOST: non-equivalent case ---
a2 = np.random.normal(0.5, 0.05, 30)
b2 = np.random.normal(0.8, 0.05, 30)
r2 = tost_equivalence(a2, b2, margin=0.10)
assert not r2['equivalent'], f"Expected non-equiv, got tost_p={r2['tost_p']:.4f}"
print(f"[PASS] non-equiv: tost_p={r2['tost_p']:.4f} CI=[{r2['ci_lo']:.4f},{r2['ci_hi']:.4f}]")

# --- compute_equivalence_margin ---
gp_results = [
    {
        'values': np.random.normal(0.5, 0.05, (30, 20)).tolist(),
        'y_test': np.linspace(0, 1, 20),
    }
    for _ in range(5)
]
margin = compute_equivalence_margin(gp_results, metric='final_regret')
assert margin > 0, f"Expected positive margin, got {margin}"
print(f"[PASS] auto-margin={margin:.4f}")

# --- run_tost_on_results ---
rng = np.random.default_rng(42)
def _make_res(subject, emg, offset=0.0):
    return {
        'subject': subject,
        'emg': emg,
        'values': (rng.random((30, 20)) * 0.6 + 0.2 + offset).tolist(),
        'y_test': np.linspace(0, 1, 20),
        'r2': (rng.random(30) * 0.4 + 0.4 + offset).tolist(),
    }

combined = {
    'GP': [_make_res(1, 0), _make_res(1, 1)],
    'TabPFN': [_make_res(1, 0, 0.01), _make_res(1, 1, 0.01)],
}
df = run_tost_on_results(combined, margin=0.15)
assert not df.empty, "Expected non-empty TOST DataFrame"
assert 'All' in df['label'].values, "Expected 'All' aggregate row"
print(f"[PASS] run_tost_on_results: {len(df)} rows, metrics={df['metric'].unique().tolist()}")

# --- plot_gradient_share (no save, just no crash) ---
diagnostics = [
    {
        'epoch': ep,
        'grad_norm': {
            'transformer_encoder': float(1.0 / (ep + 1)),
            'decoder_dict': float(3.0 / (ep + 1)),
            'other': float(0.2 / (ep + 1)),
        },
        'grad_weight_ratio': {
            'transformer_encoder': 0.1,
            'decoder_dict': 0.3,
            'other': 0.02,
        },
    }
    for ep in range(5)
]
plot_gradient_share(diagnostics, save=False)
print("[PASS] plot_gradient_share rendered without error")

# --- plot_tost_forest (no save, just no crash) ---
plot_tost_forest(df, margin=0.15, dataset='nhp', save=False)
print("[PASS] plot_tost_forest rendered without error")

print("\nAll smoke tests passed.")
