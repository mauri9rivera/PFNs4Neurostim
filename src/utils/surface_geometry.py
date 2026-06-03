"""
Surface geometry metrics for comparing predicted vs true response surfaces.

Provides gradient-field similarity (GFS) and structural similarity (SSIM)
to test whether TabPFN's predicted surfaces faithfully reproduce the spatial
derivative structure of ground-truth neurostimulation response maps.

Scientific rationale: GP posterior means are globally smooth (kernel = spatial
low-pass filter). R² is insensitive to *where* explained variance sits on the
grid; a GP can achieve R²=0.7 while mislocating sharp motor-point boundaries.
GFS tests whether predicted gradient magnitude is high at the same grid cells
as the true gradient magnitude — the decisive geometric test for non-smoothness.

Reference: A5 in roadmap.md — Prediction Surface Geometry analysis.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, sobel
from scipy.stats import pearsonr

# Default smoothing scales (grid-cell units) for the GFS multi-scale profile.
GFS_SIGMAS_DEFAULT: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)

# Single-scale summary sigma for box plots and paper text.
GFS_SUMMARY_SIGMA: float = 1.0


def _to_grid(
    values: np.ndarray,
    ch2xy: np.ndarray,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """Scatter electrode values into a NaN-padded 2D display grid.

    Args:
        values: [N] per-electrode values.
        ch2xy: [N, 2] integer (row, col) grid positions.
        grid_shape: (H, W) grid dimensions.

    Returns:
        [H, W] float64 array; NaN where no electrode is present.
    """
    grid = np.full(grid_shape, np.nan, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).ravel()
    for i, (r, c) in enumerate(ch2xy):
        grid[int(r), int(c)] = values[i]
    return grid


def _gradient_magnitude_grid(surface: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude of a 2D surface.

    NaN positions are temporarily filled with 0 for the convolution and
    then re-masked so they do not enter the downstream Pearson correlation.

    Args:
        surface: [H, W] float array, possibly containing NaN.

    Returns:
        [H, W] gradient magnitude array; NaN where *surface* was NaN.
    """
    nan_mask = np.isnan(surface)
    filled = np.where(nan_mask, 0.0, surface)
    gx = sobel(filled, axis=0)  # [H, W]
    gy = sobel(filled, axis=1)  # [H, W]
    mag = np.sqrt(gx ** 2 + gy ** 2)  # [H, W]
    mag[nan_mask] = np.nan
    return mag


def compute_gfs(
    f_pred: np.ndarray,
    f_true: np.ndarray,
    ch2xy: np.ndarray,
    grid_shape: tuple[int, int],
    sigmas: tuple[float, ...] = GFS_SIGMAS_DEFAULT,
) -> dict[float, float]:
    """Gradient-Field Similarity between predicted and true response surfaces.

    For each smoothing scale σ:
    1. Reconstruct sparse electrode map onto 2D grid (NaN for empty positions).
    2. Gaussian-smooth: f̃ = G_σ * f  (NaN positions filled with 0 before convolution).
    3. Compute Sobel gradient magnitude map: M = ‖∇f̃‖₂.
    4. GFS(σ) = Pearson(M_pred, M_true) on valid electrode positions only.

    GFS ≈ 1 means predicted gradients match ground-truth (faithful spatial
    geometry). GFS ≈ 0 means gradient patterns are uncorrelated (smoothed-over
    boundaries or mislocated hotspots). GP posterior means are expected to have
    depressed GFS at fine scales (σ ≤ 1) because the kernel acts as a spatial
    low-pass filter. TabPFN, being non-smooth, should preserve fine-scale
    gradient structure that GP loses.

    Args:
        f_pred: [N] predicted response values; N = number of valid electrodes.
        f_true: [N] ground-truth response values (same sites as f_pred).
        ch2xy: [N, 2] integer (row, col) electrode grid positions.
        grid_shape: (H, W) full grid dimensions.
        sigmas: Gaussian smoothing scales in grid-cell units.

    Returns:
        Dict[sigma → GFS scalar ∈ [-1, 1]].
        NaN for a scale if gradient magnitudes have zero variance (degenerate).
    """
    pred_grid = _to_grid(f_pred, ch2xy, grid_shape)   # [H, W]
    true_grid = _to_grid(f_true, ch2xy, grid_shape)   # [H, W]

    # Combined validity mask: only positions where BOTH grids have electrodes.
    valid = ~np.isnan(pred_grid) & ~np.isnan(true_grid)  # [H, W]

    gfs: dict[float, float] = {}
    for sigma in sigmas:
        pred_filled = np.where(np.isnan(pred_grid), 0.0, pred_grid)
        true_filled = np.where(np.isnan(true_grid), 0.0, true_grid)

        pred_smooth = gaussian_filter(pred_filled, sigma=float(sigma))  # [H, W]
        true_smooth = gaussian_filter(true_filled, sigma=float(sigma))  # [H, W]

        # Re-apply validity mask before Sobel so out-of-grid zeros don't
        # contribute spurious edge artifacts to the gradient map.
        pred_masked = np.where(valid, pred_smooth, np.nan)  # [H, W]
        true_masked = np.where(valid, true_smooth, np.nan)  # [H, W]

        pred_mag = _gradient_magnitude_grid(pred_masked)  # [H, W]
        true_mag = _gradient_magnitude_grid(true_masked)  # [H, W]

        # Flatten to 1D — keep only jointly valid, non-NaN cells.
        finite = valid & ~np.isnan(pred_mag) & ~np.isnan(true_mag)
        v_pred = pred_mag[finite]
        v_true = true_mag[finite]

        if len(v_pred) < 3 or float(v_pred.std()) == 0.0 or float(v_true.std()) == 0.0:
            gfs[sigma] = float('nan')
            continue

        r, _ = pearsonr(v_pred, v_true)
        gfs[sigma] = float(r)

    return gfs


def compute_ssim_score(
    f_pred: np.ndarray,
    f_true: np.ndarray,
    ch2xy: np.ndarray,
    grid_shape: tuple[int, int],
    win_size: int = 3,
) -> float:
    """Structural similarity between predicted and true response surfaces.

    Uses ``skimage.metrics.structural_similarity`` on the 2D grid
    reconstruction.  Only applicable where ``min(H, W) >= win_size + 2``.

    Excluded for Rat (8×4) — narrow dimension too small for default window.
    Must be pre-validated with B9 Cohen's d > 0.8 before use as paper evidence.

    Args:
        f_pred: [N] predicted responses.
        f_true: [N] ground-truth responses.
        ch2xy: [N, 2] electrode grid positions.
        grid_shape: (H, W) grid dimensions.
        win_size: SSIM local window size (must be odd, ≥ 3).

    Returns:
        SSIM score in [-1, 1], or NaN if data range is 0.

    Raises:
        ImportError: If ``scikit-image`` is not installed.
        ValueError: If ``min(grid_shape) < win_size + 2``.
    """
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise ImportError(
            "compute_ssim_score requires scikit-image: pip install scikit-image"
        ) from exc

    H, W = grid_shape
    if min(H, W) < win_size + 2:
        raise ValueError(
            f"Grid {grid_shape} too small for win_size={win_size}: "
            f"need min(H, W) >= {win_size + 2}."
        )

    pred_grid = _to_grid(f_pred, ch2xy, grid_shape)
    true_grid = _to_grid(f_true, ch2xy, grid_shape)

    data_range = float(np.nanmax(true_grid) - np.nanmin(true_grid))
    if data_range == 0.0:
        return float('nan')

    pred_filled = np.where(np.isnan(pred_grid), 0.0, pred_grid)
    true_filled = np.where(np.isnan(true_grid), 0.0, true_grid)

    score = structural_similarity(
        pred_filled, true_filled,
        win_size=win_size,
        data_range=data_range,
    )
    return float(score)
