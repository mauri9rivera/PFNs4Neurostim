"""Smoke test for the A7 scaling timing harness.

Uses tiny grid sizes (n=10,20,30) with n_reps=2 so the test runs in seconds
on CPU.  Verifies shape, non-negative timings, and that plot_scaling_timing
renders without errors.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from scaling_timing import run_scaling_benchmark, _make_synthetic_data, _time_query
from models.regressors import GPSurrogate, TabPFNSurrogate
from tabpfn import TabPFNRegressor
from utils.visualization import plot_scaling_timing

GRID_SIZES = [10, 20, 30]
N_REPS = 2

# --- run_scaling_benchmark ---
df = run_scaling_benchmark(
    grid_sizes=GRID_SIZES,
    n_context_frac=0.3,
    n_reps=N_REPS,
    d=2,
    tabpfn_device='cpu',
    gp_n_opt_steps=5,
    seed=42,
)

expected_rows = len(GRID_SIZES) * N_REPS * 2  # GP + TabPFN
assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"
assert set(df.columns) == {'model', 'n', 'rep', 'time_s'}, f"Unexpected columns: {df.columns.tolist()}"
assert (df['time_s'] >= 0).all(), "Negative timings detected"
assert set(df['model'].unique()) == {'GP', 'TabPFN'}, f"Unexpected models: {df['model'].unique()}"
assert set(df['n'].unique()) == set(GRID_SIZES), f"Unexpected n values: {df['n'].unique()}"
print(f"[PASS] run_scaling_benchmark: {len(df)} rows, models={df['model'].unique().tolist()}")

# --- plot_scaling_timing (no save) ---
plot_scaling_timing(df, save=False, anchor_n=20)
print("[PASS] plot_scaling_timing rendered without error")

# --- plot_scaling_timing with anchor outside grid range (no save) ---
plot_scaling_timing(df, save=False, anchor_n=50, anchor_label='outside range')
print("[PASS] plot_scaling_timing with out-of-range anchor_n — no crash")

# --- _make_synthetic_data shapes ---
rng = np.random.default_rng(0)
X_ctx, y_ctx, X_pool = _make_synthetic_data(100, 10, 2, rng)
assert X_ctx.shape == (10, 2), f"X_ctx shape: {X_ctx.shape}"
assert y_ctx.shape == (10,), f"y_ctx shape: {y_ctx.shape}"
assert X_pool.shape == (90, 2), f"X_pool shape: {X_pool.shape}"
print("[PASS] _make_synthetic_data shapes correct")

print("\nAll smoke tests passed.")
