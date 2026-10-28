"""Schematic of how RQMC and measures A / B / C place nodes on TabPFN's predictive (illustration, not data).

Tail sizes follow nhp-s1-e0, t = 40: outer buckets 61 SD wide holding 3e-6 / 7e-5 of the mass, half-normal scale 90 SD.

    python scripts/rqmc_measures_cartoon.py
"""
from __future__ import annotations

import argparse

import numpy as np

from pfns4neurostim.visualization import style

Z_RIGHT, Z_LEFT, WIDTH, SCALE = 3.5, -1.8, 61.0, 90.0
P_RIGHT, P_LEFT = 6.8e-5, 3.4e-6
N_STRATA = 32
N_SCRAMBLES, N_PER_SCRAMBLE = 8, 8
SEED = 7


def body_density(z: np.ndarray) -> np.ndarray:
    """Schematic skewed body density per SD, shape [len(z)]."""
    return np.where(z >= -0.3, 0.95 / 0.6 * np.exp(-(z + 0.3) / 0.6), 0.04 / 1.5)


def body_quantile(u: np.ndarray) -> np.ndarray:
    """Quantile function of the schematic body, shape [len(u)]."""
    v = np.clip((u - 0.04) / 0.96, 0.0, 1.0)
    return np.where(u < 0.04, Z_LEFT + 0.1 + u / 0.04 * 1.4, -0.3 - 0.6 * np.log(1.0 - v * 0.9995))


def half_normal_pdf(x: np.ndarray, scale: float) -> np.ndarray:
    """Half-normal density, shape [len(x)]."""
    return 2.0 / (scale * np.sqrt(2.0 * np.pi)) * np.exp(-(x ** 2) / (2.0 * scale ** 2))


def main() -> None:
    """Draw the figure."""
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    rng = np.random.default_rng(SEED)
    fig, axes = style.figure("double", nrows=5, ncols=1, panel_aspect=0.16, sharex=True, layout=style.LAYOUT_ENGINE,
                             gridspec_kw={"height_ratios": [2.2, 1, 1, 1, 1]})
    ax = axes[0]
    zb = np.linspace(Z_LEFT, Z_RIGHT, 400)
    ax.fill_between(zb, 1e-9, body_density(zb), color=style.ANCHOR_PFN, alpha=0.4, lw=0)
    ax.plot(zb, body_density(zb), color=style.ANCHOR_PFN, lw=style.LINE_WIDTH, label="bars (body)")
    for sign, z0, p in ((1, Z_RIGHT, P_RIGHT), (-1, Z_LEFT, P_LEFT)):
        ax.plot([z0, z0 + sign * WIDTH], [p / WIDTH] * 2, color=style.NEUTRAL_GREY, lw=2.5,
                label="uniform on the bucket (B)" if sign == 1 else None)
        x = np.linspace(0, 330, 300)
        ax.plot(z0 + sign * x, p * half_normal_pdf(x, SCALE), color=style.ANCHOR_GP, lw=style.LINE_WIDTH,
                label="half-normal tail (A, official)" if sign == 1 else None)
        ax.axvline(z0 + sign * WIDTH, color=style.NEUTRAL_GREY, lw=0.5, ls=":")
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 20)
    ax.set_ylabel("Density per SD")
    ax.legend(loc="upper left", fontsize=style.FONT_SIZES["legend"] - 1)
    ax.set_title("Predictive of $y'$ at the anchor (log density)", fontsize=style.FONT_SIZES["legend"])

    strata = body_quantile((np.arange(N_STRATA) + 0.5) / N_STRATA)
    mean_hn, sd_hn = np.sqrt(2 / np.pi) * SCALE, SCALE * np.sqrt(1 - 2 / np.pi)
    centre, sd_u = WIDTH / 2, WIDTH / np.sqrt(12.0)
    u8 = np.concatenate([(np.arange(1, N_PER_SCRAMBLE + 1) / (N_PER_SCRAMBLE + 1) + rng.random()) % 1.0 for _ in range(N_SCRAMBLES)])
    rows = [
        ("RQMC 8x8: 64 equal-weight nodes, all in the body", body_quantile(np.clip(u8, 0.002, 0.9995)), None, None),
        ("A: 32 strata + 2 nodes per half-normal tail (tail nodes: weight 3e-5 each, ~71% of the stated variance)", strata,
         np.array([Z_LEFT - mean_hn - sd_hn, Z_LEFT - mean_hn + sd_hn, Z_RIGHT + mean_hn - sd_hn, Z_RIGHT + mean_hn + sd_hn]), style.ANCHOR_GP),
        ("B: 32 strata + 2 nodes per uniform bucket", strata,
         np.array([Z_LEFT - centre - sd_u, Z_LEFT - centre + sd_u, Z_RIGHT + centre - sd_u, Z_RIGHT + centre + sd_u]), style.NEUTRAL_GREY),
        ("C: 32 strata, outer buckets dropped", strata, None, None),
    ]
    for ax, (label, body, tails, colour) in zip(axes[1:], rows):
        ax.scatter(body, np.zeros_like(body), s=14, color=style.ANCHOR_PFN, alpha=0.7, lw=0)
        if tails is not None:
            ax.scatter(tails, np.zeros_like(tails), s=60, color=colour, lw=0, zorder=5)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-1, 1)
        ax.set_title(label, fontsize=style.FONT_SIZES["legend"], loc="left")
    axes[-1].set_xscale("symlog", linthresh=3.0)
    axes[-1].set_xlim(-160, 160)
    axes[-1].set_xticks([-100, -10, -1, 0, 1, 10, 100])
    axes[-1].set_xticklabels(["-100", "-10", "-1", "0", "1", "10", "100"])
    axes[-1].set_xlabel(style.axis_label("z_sd"))
    print("\n".join(style.save_figure(fig, "output/diagnostics/rqmc", "rqmc_measures_cartoon", formats=("svg", "png"))))


if __name__ == "__main__":
    main()
