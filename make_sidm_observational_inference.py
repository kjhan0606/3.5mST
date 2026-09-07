#!/usr/bin/env python3
"""Draw the observational SIDM inference used in Appendix A.

The table reconstructs the plotted posterior medians and 68 per cent intervals
from the vector paths in the author-supplied arXiv figure of Kaplinghat, Tulin,
and Yu (2016). Their article does not provide the individual fit values as a
machine-readable table. No simulation particles or published figure pixels are
used in the output.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "sidm_observational_inference.csv"
OUTPUT = ROOT / "sidm_observational_inference.png"

STYLE = {
    "Dwarf": ("#e64b35", "o"),
    "LSB": ("#2f73c8", "s"),
    "Cluster": ("#159a74", "^"),
}


def main() -> None:
    data = np.genfromtxt(DATA, delimiter=",", names=True, dtype=None, encoding="utf-8")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.labelsize": 15,
            "axes.titlesize": 17,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )
    fig, ax = plt.subplots(figsize=(10.4, 6.6), facecolor="white")
    ax.set_facecolor("#fbfcfe")

    velocity_grid = np.geomspace(18.0, 3200.0, 400)
    for sigma, alpha in [(0.1, 0.30), (1.0, 0.44), (10.0, 0.30)]:
        rate = sigma * velocity_grid
        ax.plot(velocity_grid, rate, color="#5b6470", lw=1.2, alpha=alpha, zorder=1)
        x_label = 2600.0 if sigma < 10 else 32.0
        y_label = sigma * x_label
        ax.text(
            x_label,
            y_label,
            rf"$\sigma/m={sigma:g}\ \mathrm{{cm^2\,g^{{-1}}}}$",
            color="#46505c",
            fontsize=10.5,
            rotation=24,
            ha="right" if sigma < 10 else "left",
            va="bottom",
            bbox={"facecolor": "#fbfcfe", "edgecolor": "none", "pad": 1.0, "alpha": 0.82},
        )

    for group, (color, marker) in STYLE.items():
        rows = data[data["class"] == group]
        x = rows["velocity_km_s"]
        y = rows["rate_cm2_g_km_s"]
        xerr = np.vstack((x - rows["velocity_lo_km_s"], rows["velocity_hi_km_s"] - x))
        yerr = np.vstack((y - rows["rate_lo_cm2_g_km_s"], rows["rate_hi_cm2_g_km_s"] - y))
        label = "Low-surface-brightness galaxy" if group == "LSB" else group
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt=marker,
            ms=7.5,
            mfc=color,
            mec="white",
            mew=0.8,
            ecolor=color,
            elinewidth=1.25,
            capsize=2.5,
            alpha=0.96,
            label=label,
            zorder=4,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(18, 3200)
    ax.set_ylim(2.5, 6000)
    ax.set_xlabel(r"Mean dark-matter collision velocity, $\langle v\rangle$ [km s$^{-1}$]")
    ax.set_ylabel(r"Velocity-weighted interaction, $\langle\sigma v\rangle/m$ [cm$^2$ g$^{-1}$ km s$^{-1}$]")
    ax.set_title("Observed halo fits probe velocity-dependent dark-matter scattering", pad=12)
    ax.grid(which="major", color="#aeb7c2", alpha=0.28, lw=0.8)
    ax.grid(which="minor", color="#c8ced6", alpha=0.13, lw=0.55)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#c4cad1", framealpha=0.96)
    ax.text(
        0.018,
        0.025,
        "Posterior medians and 68% intervals from halo-profile fits",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#4d5865",
    )

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
