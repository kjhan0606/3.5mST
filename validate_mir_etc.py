#!/usr/bin/env python3
"""Validate the cooled MIR imaging mode and write proposal-ready products."""

from dataclasses import replace
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

import slitless_etc as etc


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "appendixC_assets"
CSV_PATH = ROOT / "mir_etc_validation.csv"
FIGURE_PATH = ASSET_DIR / "mir_etc_validation.png"
EXPOSURES_S = (145.0, 580.0, 2320.0)
LEVELS = ("low", "nominal", "high")
PUBLISHED_NESI5_UJY = {"NC1": (65.0, 120.0), "NC2": (110.0, 280.0)}


def band_A(channel):
    return tuple(
        wavelength * 1e4
        for wavelength in etc.MIR_CHANNELS[channel]["band_um"])


def limit_ujy(observatory, channel, level, exposure_s):
    cfg = etc.mir_imaging_cfg(
        channel, level, observatory=observatory)
    return 1e6 * etc.imaging_flux_limit_jy(
        cfg, band_A(channel), exposure_s)


def thermal_fraction(channel, level="low"):
    cfg = etc.mir_imaging_cfg(channel, level)
    no_thermal = replace(cfg, include_thermal=False)
    total = etc.background_per_pixel(cfg, band_A(channel), imaging=True)
    zodiacal = etc.background_per_pixel(
        no_thermal, band_A(channel), imaging=True)
    return max(0.0, (total - zodiacal) / total)


def write_csv():
    rows = []
    for observatory in ("NEO Surveyor", "3.5mST"):
        for channel in ("NC1", "NC2"):
            for level in LEVELS:
                for exposure_s in EXPOSURES_S:
                    rows.append({
                        "observatory": observatory,
                        "channel": channel,
                        "background": level,
                        "exposure_s": exposure_s,
                        "five_sigma_ujy": limit_ujy(
                            observatory, channel, level, exposure_s),
                    })
    with CSV_PATH.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_figure():
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "text.color": "#151B23",
        "axes.labelcolor": "#151B23",
        "axes.edgecolor": "#5C6875",
        "xtick.color": "#151B23",
        "ytick.color": "#151B23",
    })
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)
    colors = {"low": "#007C77", "nominal": "#B86B00", "high": "#B55220"}

    ax = axes[0]
    x = np.arange(2)
    for offset, level in zip((-0.24, 0.0, 0.24), LEVELS):
        values = [
            limit_ujy("NEO Surveyor", channel, level, 145.0)
            for channel in ("NC1", "NC2")
        ]
        ax.bar(
            x + offset, values, width=0.22, color=colors[level],
            label=f"{level} zodiacal field")
    for index, channel in enumerate(("NC1", "NC2")):
        lower, upper = PUBLISHED_NESI5_UJY[channel]
        ax.fill_between(
            [index - 0.42, index + 0.42], lower, upper,
            color="#245AA6", alpha=0.16)
        ax.hlines(
            np.sqrt(lower * upper), index - 0.42, index + 0.42,
            color="#245AA6", lw=2.0)
    ax.set_xticks(x, ("NC1  4.0-5.2 um", "NC2  6.0-10.0 um"))
    ax.set_ylabel("NESI5 [uJy]")
    ax.set_title("Literature validation with the 50 cm configuration")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(fontsize=9)
    ax.text(
        0.03, 0.78,
        "Blue bands: Mainzer et al. (2023) requirements",
        transform=ax.transAxes,
        color="#5C6875",
        fontsize=9,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )

    ax = axes[1]
    for channel, linestyle in (("NC1", "-"), ("NC2", "--")):
        for level in LEVELS:
            values = [
                limit_ujy("3.5mST", channel, level, exposure)
                for exposure in EXPOSURES_S
            ]
            ax.plot(
                EXPOSURES_S, values, marker="o", lw=2.0, ls=linestyle,
                color=colors[level],
                label=f"{channel}, {level}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("On-source integration [s]")
    ax.set_ylabel("5-sigma point-source limit [uJy]")
    ax.set_title("3.5ST cooled MIR reference performance")
    ax.grid(alpha=0.18)
    ax.legend(ncol=2, fontsize=8)

    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_color("#5C6875")
    fig.savefig(FIGURE_PATH, dpi=220, facecolor="white")
    plt.close(fig)


def validate():
    failures = []
    for channel in ("NC1", "NC2"):
        cfg = etc.mir_imaging_cfg(channel)
        pivot_um = np.sqrt(np.prod(etc.MIR_CHANNELS[channel]["band_um"]))
        expected_scale = etc.diffraction_nyquist_pixel_scale_arcsec(
            pivot_um, cfg.diameter_cm)
        if not np.isclose(cfg.pix_scale, expected_scale, rtol=0.0, atol=1e-12):
            failures.append(
                f"{channel} pixel scale is not lambda/(2D): "
                f"{cfg.pix_scale:.6f} versus {expected_scale:.6f} arcsec")

        model = limit_ujy("NEO Surveyor", channel, "nominal", 145.0)
        published = np.sqrt(np.prod(PUBLISHED_NESI5_UJY[channel]))
        residual = model / published - 1.0
        if abs(residual) > 0.08:
            failures.append(
                f"{channel} validation residual {residual:+.1%}")

        limits = np.array([
            limit_ujy("3.5mST", channel, "nominal", exposure)
            for exposure in EXPOSURES_S
        ])
        slopes = np.diff(np.log(limits)) / np.diff(np.log(EXPOSURES_S))
        if np.any((slopes > -0.42) | (slopes < -0.58)):
            failures.append(
                f"{channel} exposure slopes outside background limit: {slopes}")

        fraction = thermal_fraction(channel)
        if fraction > 0.10:
            failures.append(
                f"{channel} 55 K thermal fraction is {fraction:.1%}")

    if failures:
        raise RuntimeError("\n".join(failures))


def main():
    ASSET_DIR.mkdir(exist_ok=True)
    validate()
    write_csv()
    make_figure()
    print("MIR ETC validation passed")
    for channel in ("NC1", "NC2"):
        low = limit_ujy("3.5mST", channel, "low", 145.0)
        nominal = limit_ujy("3.5mST", channel, "nominal", 145.0)
        high = limit_ujy("3.5mST", channel, "high", 145.0)
        print(
            f"{channel} pixel scale: "
            f"{etc.mir_imaging_cfg(channel).pix_scale:.4f} arcsec; "
            f"145 s NESI5: {low:.2f}, {nominal:.2f}, "
            f"{high:.2f} uJy; 55 K thermal fraction "
            f"{thermal_fraction(channel):.3%}")
    print(CSV_PATH)
    print(FIGURE_PATH)


if __name__ == "__main__":
    main()
