#!/usr/bin/env python3
"""Generate the publication view of the ETC interface and its live plots."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import slitless_etc as etc

HR = 3600.0
EXPOSURES = (0.75, 3.0, 12.0)
COLORS = ("#348bd4", "#35b9a0", "#d66d4b")


def control_group(ax, y, title, rows, height):
    box = FancyBboxPatch((0.025, y - height), 0.95, height,
                         boxstyle="round,pad=0.008", facecolor="#f5f6f7",
                         edgecolor="#9aa0a6", linewidth=0.8)
    ax.add_patch(box)
    ax.text(0.045, y - 0.025, title, fontsize=9, weight="bold", va="top")
    yy = y - 0.07
    for label, value in rows:
        ax.text(0.055, yy, label, fontsize=7.5, va="center", color="#30343b")
        ax.text(0.56, yy, value, fontsize=7.5, va="center", color="#111827",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#b7bcc2", lw=0.5))
        yy -= 0.037
    return y - height - 0.018


def make_figure(output="etc_gui_capture.png"):
    cfg = etc.realistic_cfg(R=5000)
    filters = etc.STANDARD_FILTERS
    names = list(filters)
    x = np.arange(len(names), dtype=float)

    fig = plt.figure(figsize=(12, 9.5), facecolor="#e6e8eb")
    grid = fig.add_gridspec(2, 2, width_ratios=(0.34, 0.66), hspace=0.34,
                            wspace=0.18, left=0.025, right=0.985,
                            bottom=0.07, top=0.90)
    ui = fig.add_subplot(grid[:, 0])
    imaging = fig.add_subplot(grid[0, 1])
    spectrum = fig.add_subplot(grid[1, 1])

    ui.set_xlim(0, 1)
    ui.set_ylim(0, 1)
    ui.axis("off")
    ui.text(0.025, 1.025, "Space ETC", fontsize=14, weight="bold", va="bottom")
    ui.text(0.975, 1.025, "imaging + slitless spectroscopy", fontsize=8,
            ha="right", va="bottom", color="#4b5563")

    y = 0.99
    y = control_group(ui, y, "TELESCOPE", [
        ("preset", "3.5mST"), ("spectral element", "R5000 NIR")], 0.13)
    y = control_group(ui, y, "MODE AND CALCULATION", [
        ("mode", "Both"), ("calculation", "Limiting depth (5sigma)")], 0.13)
    y = control_group(ui, y, "TELESCOPE AND DETECTOR", [
        ("aperture", "350 cm"), ("pixel scale", "0.11 arcsec"),
        ("peak throughput", "0.30"), ("read noise", "8 e-"),
        ("dark current", "0.010 e-/s/pix"), ("exposures", "3")], 0.28)
    y = control_group(ui, y, "OBSERVATION", [
        ("imaging filter", "SDSS g"), ("spectral wavelength", "1.60 um"),
        ("spectral band", "0.40 um"), ("resolving power", "5000"),
        ("exposure", "0.75 hr")], 0.245)

    selected = filters["SDSS g"]
    selected_depth = etc.imaging_maglimit(cfg, selected[0] * 1e4,
                                           selected[1] * 1e4, 0.75 * HR)
    line_depth = etc.f_limit(cfg, 1.6e4, 0.75 * HR, source_fwhm=0.3)
    ui.text(0.045, y - 0.005,
            "OUTPUT\n"
            f"SDSS g   5sigma AB = {selected_depth:.2f}\n"
            f"R5000 NIR   F5sigma = {line_depth:.2e} cgs\n\n"
            "Available imaging systems\n"
            "Johnson-Cousins  UBVRI\n"
            "SDSS  ugriz\n"
            "2MASS  JHKs\n\n"
            "Available resolving powers\n"
            "R = 1000, 5000",
            family="monospace", fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#9aa0a6", lw=0.8))

    for lo, hi, colour in [(0, 4, "#f3ead8"), (5, 9, "#e4f0f8"),
                            (10, 12, "#ebe6f4")]:
        imaging.axvspan(lo - 0.48, hi + 0.48, color=colour, alpha=0.78, zorder=0)
    for hour, colour, dx in zip(EXPOSURES, COLORS, (-0.18, 0.0, 0.18)):
        depths = [etc.imaging_maglimit(cfg, lam * 1e4, width * 1e4, hour * HR)
                  for lam, width in filters.values()]
        imaging.plot(x + dx, depths, "o-", color=colour, ms=5, lw=1.4,
                     label=f"{hour:g} hr")
    for xpos, label in [(2, "Johnson-Cousins"), (7, "SDSS"), (11, "2MASS")]:
        imaging.text(xpos, 1.025, label, ha="center", va="bottom", fontsize=8,
                     transform=imaging.get_xaxis_transform())
    imaging.set_xticks(x, [name.split()[-1] for name in names])
    imaging.set_xlim(-0.55, len(names) - 0.45)
    imaging.set_ylabel(r"$5\sigma$ point-source depth [AB mag]")
    imaging.set_xlabel("standard photometric filter")
    imaging.set_title("3.5mST imaging depth by passband", pad=25)
    imaging.invert_yaxis()
    imaging.grid(alpha=0.22)
    imaging.legend(loc="upper right", fontsize=8, ncol=3)

    lam = np.linspace(cfg.band_min_A + 200, cfg.band_max_A - 200, 220)
    for hour, colour in zip(EXPOSURES, COLORS):
        depth = [etc.f_limit(cfg, wavelength, hour * HR, source_fwhm=0.3)
                 for wavelength in lam]
        spectrum.plot(lam / 1e4, depth, color=colour, lw=1.8,
                      label=f"{hour:g} hr")
    spectrum.axhline(1e-16, color="#222222", ls=":", lw=1)
    spectrum.set_yscale("log")
    spectrum.set_xlabel(r"observed wavelength [$\mu$m]")
    spectrum.set_ylabel(r"$F_{5\sigma}$ [erg s$^{-1}$ cm$^{-2}$]")
    spectrum.set_title(r"Slitless depth at $R=5000$")
    spectrum.grid(alpha=0.22, which="both")
    spectrum.legend(loc="upper left", fontsize=8)

    fig.savefig(output, dpi=150, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    make_figure()
