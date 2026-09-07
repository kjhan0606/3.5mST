#!/usr/bin/env python3
"""Reproduce how Nube emerges as the surface-brightness limit deepens.

The source frame is the public HiPERCAM r-band image distributed with Montes
et al. (2024) through CDS. Shallower panels are generated in calibrated flux
units by adding the noise required to reproduce the quoted 3-sigma limits in
10 by 10 arcsec boxes. The image content itself is never synthesized.
"""

from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


ROOT = Path(__file__).resolve().parent
URL = "https://cdsarc.cds.unistra.fr/ftp/J/A+A/681/A15/fits/DGS82-r-v0.0.5-dgs82-4-gb8cc884.fits"
CACHE = Path("/tmp/DGS82-r-v0.0.5-dgs82-4-gb8cc884.fits")
OUTPUT = ROOT / "nube_surface_brightness_depth.png"

RA_DEG = 20.8640417
DEC_DEG = -0.6243972
LIMITS = (26.7, 28.6, 30.5)
BIN = 8
CUTOUT_NATIVE_PIXELS = 1200


def block_mean(array: np.ndarray, factor: int) -> np.ndarray:
    ny = array.shape[0] // factor * factor
    nx = array.shape[1] // factor * factor
    trimmed = array[:ny, :nx]
    return trimmed.reshape(ny // factor, factor, nx // factor, factor).mean(axis=(1, 3))


def robust_sigma(array: np.ndarray) -> float:
    median = np.nanmedian(array)
    mad = np.nanmedian(np.abs(array - median))
    return 1.4826 * mad


def pixel_noise_for_limit(mu_limit: float, zero_point: float, pixel_scale: float) -> float:
    area = 10.0 * 10.0
    aperture_pixels = area / pixel_scale**2
    integrated_magnitude = mu_limit - 2.5 * np.log10(area)
    limiting_flux = 10.0 ** (-0.4 * (integrated_magnitude - zero_point))
    return limiting_flux / (3.0 * np.sqrt(aperture_pixels))


def main() -> None:
    if not CACHE.exists():
        print(f"Downloading {URL}")
        urlretrieve(URL, CACHE)

    with fits.open(CACHE, memmap=True) as hdul:
        data = np.asarray(hdul[1].data, dtype=np.float32)
        header = hdul[1].header
        x_center, y_center = WCS(header).world_to_pixel_values(RA_DEG, DEC_DEG)
        x_center = float(np.asarray(x_center).reshape(-1)[0])
        y_center = float(np.asarray(y_center).reshape(-1)[0])
        half = CUTOUT_NATIVE_PIXELS // 2
        x0 = int(round(x_center)) - half
        y0 = int(round(y_center)) - half
        cutout = data[y0 : y0 + 2 * half, x0 : x0 + 2 * half].astype(float)
        zero_point = float(header["ZP"])
        pixel_scale = abs(float(header["CDELT1"])) * 3600.0

    cutout -= np.nanmedian(cutout)
    deep_sigma = pixel_noise_for_limit(30.5, zero_point, pixel_scale)
    rng = np.random.default_rng(20240818)

    images = []
    for limit in LIMITS:
        target_sigma = pixel_noise_for_limit(limit, zero_point, pixel_scale)
        added_sigma = np.sqrt(max(target_sigma**2 - deep_sigma**2, 0.0))
        degraded = cutout + rng.normal(0.0, added_sigma, cutout.shape)
        rebinned = block_mean(degraded, BIN)
        rebinned -= np.nanmedian(rebinned)
        images.append((rebinned, robust_sigma(rebinned)))

    binned_scale = pixel_scale * BIN
    extent = np.array([-half, half, -half, half]) * pixel_scale
    center_binned = np.array(images[0][0].shape[::-1]) / 2.0
    re_pixels = 13.7 / binned_scale

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 15,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.65), facecolor="white")
    for ax, limit, (image, sigma) in zip(axes, LIMITS, images):
        ax.imshow(
            image / sigma,
            origin="lower",
            cmap="gray_r",
            vmin=-1.4,
            vmax=4.2,
            extent=extent,
            interpolation="nearest",
        )
        radius_arcsec = re_pixels * binned_scale
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                radius_arcsec,
                fill=False,
                edgecolor="#f28e2b",
                linewidth=1.7,
                linestyle=(0, (4, 3)),
            )
        )
        ax.plot(0.0, 0.0, marker="+", color="#f28e2b", ms=8, mew=1.5)
        ax.set_title(rf"$\mu_{{r,3\sigma}}={limit:.1f}$ mag arcsec$^{{-2}}$")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel(r"Offset in R.A. [$''$]")
        ax.tick_params(direction="in", top=True, right=True, color="white")
    axes[0].set_ylabel(r"Offset in Dec. [$''$]")
    for ax in axes[1:]:
        ax.set_yticklabels([])

    axes[0].plot([-41, -21], [-42, -42], color="#f28e2b", lw=3.0, solid_capstyle="butt")
    axes[0].text(-31, -38.5, r"$20''$", color="#f28e2b", ha="center", va="bottom", fontsize=11)
    axes[2].text(
        0.97,
        0.04,
        r"Dashed circle: $R_e=13.7''$",
        transform=axes[2].transAxes,
        color="#f28e2b",
        ha="right",
        va="bottom",
        fontsize=10.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.5},
    )
    fig.suptitle("Nube emerges only in ultra-deep surface-brightness imaging", fontsize=18, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=0.25)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
