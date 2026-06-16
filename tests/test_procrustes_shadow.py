"""Shadow smoke test for Procrustes BO-trajectory analysis (B7).

Run with:
    conda activate pfns4neurostim
    cd PFNs4Neurostim
    python tests/test_procrustes_shadow.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from tabpfn import TabPFNRegressor

from analysis.id_ood import (
    compute_procrustes_disparity,
    _trajectory_disparities,
    embedding_trajectory_analysis,
)


def hr(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Test 1: disparity(Z, Z) == 0 ─────────────────────────────────────────────
hr("Test 1: compute_procrustes_disparity(Z, Z) should be 0.0")
rng = np.random.RandomState(0)
Z = rng.randn(30, 64).astype(np.float64)
d = compute_procrustes_disparity(Z, Z, n_subsample=30, seed=42)
print(f"  disparity(Z, Z) = {d:.8f}")
assert abs(d) < 1e-6, f"FAIL: expected ~0, got {d}"
print("  PASS")

# ── Test 2: disparity(Z1, Z2) > 0 for independent clouds ─────────────────────
hr("Test 2: disparity(Z1, Z2) > 0 for independent Gaussian clouds")
Z2 = rng.randn(30, 64).astype(np.float64)
d2 = compute_procrustes_disparity(Z, Z2, n_subsample=30, seed=42)
print(f"  disparity(Z1, Z2) = {d2:.6f}")
assert d2 > 0.01, f"FAIL: expected > 0, got {d2}"
print("  PASS")

# ── Test 3: disparity in [0, 1] ───────────────────────────────────────────────
hr("Test 3: disparity values are in [0, 1]")
assert 0.0 <= d2 <= 1.0, f"FAIL: out of range: {d2}"
print(f"  {d2:.6f} in [0, 1]  PASS")

# ── Test 4: _trajectory_disparities with small synthetic data ─────────────────
hr("Test 4: _trajectory_disparities returns correct-length list")
model = TabPFNRegressor(device='cpu')
model.n_estimators = 1

rng2 = np.random.RandomState(42)
X = rng2.rand(20, 2).astype(np.float32)
y = (np.sin(X[:, 0] * 3) + 0.1 * rng2.randn(20)).astype(np.float32)

budgets = [2, 5, 10, 15]
layer_name = 'transformer_encoder.layers.17'
rng_traj = np.random.RandomState(99)
traj = _trajectory_disparities(model, X, y, budgets, rng_traj, layer_name, n_subsample=300)
print(f"  budgets:     {budgets}")
print(f"  disparities: {traj}")
assert traj is not None, "FAIL: _trajectory_disparities returned None"
assert len(traj) == len(budgets), f"FAIL: expected {len(budgets)} values, got {len(traj)}"
print(f"  First disparity (vs baseline): {traj[0]:.8f}  (expected ~0.0)")
assert abs(traj[0]) < 1e-6, f"FAIL: first disparity should be 0, got {traj[0]}"
print("  All checks PASS")

# ── Test 5: All disparity values in [0, 1] ───────────────────────────────────
hr("Test 5: All trajectory disparities are in [0, 1]")
for i, (b, d) in enumerate(zip(budgets, traj)):
    ok = 0.0 <= d <= 1.0
    status = "PASS" if ok else "FAIL"
    print(f"  budget={b}: disparity={d:.6f}  {status}")
    assert ok, f"FAIL: disparity out of range at budget={b}: {d}"

# ── Test 6: end-to-end embedding_trajectory_analysis (tiny run) ───────────────
hr("Test 6: embedding_trajectory_analysis end-to-end (nhp, n_synthetic=2)")
results = embedding_trajectory_analysis(
    dataset_types=['nhp'],
    device='cpu',
    prior_source='tabpfn_prior',
    n_synthetic=2,
    budgets=[2, 10, 30],
    layer=17,
    n_subsample=50,
    seed=42,
)

print(f"  Top-level keys: {list(results.keys())}")
assert 'budgets' in results, "FAIL: 'budgets' key missing"
assert 'layer' in results, "FAIL: 'layer' key missing"
assert 'nhp' in results, "FAIL: 'nhp' key missing"
assert results['budgets'] == [2, 10, 30], f"FAIL: wrong budgets: {results['budgets']}"
assert results['layer'] == 17, f"FAIL: wrong layer: {results['layer']}"

print(f"  budgets: {results['budgets']}")
print(f"  layer:   {results['layer']}")

# Check NHP trajectories
nhp_data = results['nhp']
print(f"  NHP subjects: {list(nhp_data.keys())}")
assert len(nhp_data) > 0, "FAIL: no NHP subject data"

for subj_idx, subj_data in nhp_data.items():
    for emg_idx, t in list(subj_data.items())[:2]:
        print(f"    S{subj_idx} EMG{emg_idx}: {[f'{v:.4f}' for v in t]}")
        assert len(t) == 3, f"FAIL: expected 3 budget steps, got {len(t)}"
        assert abs(t[0]) < 1e-6, f"FAIL: first disparity should be 0, got {t[0]}"
        for v in t:
            assert 0.0 <= v <= 1.0, f"FAIL: disparity out of range: {v}"

# Check synthetic trajectories
print(f"  synthetic_prior: {len(results.get('synthetic_prior', []))} trajectories")
print(f"  synthetic_noise: {len(results.get('synthetic_noise', []))} trajectories")

print("\n" + "="*60)
print("  All tests PASSED")
print("="*60)
