#!/usr/bin/env python3
"""Physical-optics PSF of the 3.5 m 19-segment concept primary (POPPY
Fraunhofer/FFT propagation of the real segmented pupil, segmented_psf.py)
versus the analytic obscured-Airy PSF used by default in slitless_etc.py.

Shows the segmented pupil's diffraction spikes/gap structure (absent from a
smooth annulus) both perfectly phased and with an illustrative 30 nm RMS
per-segment piston + tip/tilt residual, and the resulting encircled-energy
loss relative to the analytic model that slitless_etc.f_limit()/line_sn()
use for the fast, wavelength-continuous spectroscopic depth calculation.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import slitless_etc as etc
import segmented_psf as sps

LAM_UM = 1.6
PISTON_RMS_NM = 30.0
TILT_RMS_URAD = 0.05

psf_phased, _, _ = sps.monochromatic_psf(LAM_UM, piston_rms_nm=0.0, tilt_rms_urad=0.0)
psf_errored, _, _ = sps.monochromatic_psf(LAM_UM, piston_rms_nm=PISTON_RMS_NM,
                                          tilt_rms_urad=TILT_RMS_URAD)

r_grid = np.linspace(0.02, 1.0, 60)
ee_phased = sps.ee_curve(psf_phased, r_grid)
ee_errored = sps.ee_curve(psf_errored, r_grid)

cfg_analytic = etc.InstrumentConfig(diameter_cm=350.0, obstruction=0.15, psf_floor=0.0)
ee_analytic = np.array([etc.encircled_energy(cfg_analytic, LAM_UM * 1e4, r) for r in r_grid])

fwhm_phased = sps.fwhm_arcsec(psf_phased)
fwhm_errored = sps.fwhm_arcsec(psf_errored)
fwhm_analytic = cfg_analytic.psf_fwhm(LAM_UM * 1e4)

print(f"At {LAM_UM} um:")
print(f"  analytic obscured-Airy   : FWHM={fwhm_analytic:.4f}\"  "
      f"EE(0.3\")={np.interp(0.3, r_grid, ee_analytic):.3f}  "
      f"EE(0.5\")={np.interp(0.5, r_grid, ee_analytic):.3f}")
print(f"  segmented, well-phased   : FWHM={fwhm_phased:.4f}\"  "
      f"EE(0.3\")={np.interp(0.3, r_grid, ee_phased):.3f}  "
      f"EE(0.5\")={np.interp(0.5, r_grid, ee_phased):.3f}")
print(f"  segmented, {PISTON_RMS_NM:.0f} nm RMS piston : FWHM={fwhm_errored:.4f}\"  "
      f"EE(0.3\")={np.interp(0.3, r_grid, ee_errored):.3f}  "
      f"EE(0.5\")={np.interp(0.5, r_grid, ee_errored):.3f}")

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), dpi=150)

for ax, psf, title in [(axes[0], psf_phased, "segmented, well-phased"),
                       (axes[1], psf_errored, f"segmented, {PISTON_RMS_NM:.0f} nm RMS piston")]:
    d = psf[0].data
    pxscale = psf[0].header["PIXELSCL"]
    half = d.shape[0] * pxscale / 2.0
    ax.imshow(np.log10(d / d.max() + 1e-6), extent=(-half, half, -half, half),
              cmap="inferno", vmin=-5, vmax=0, origin="lower")
    ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel("arcsec"); ax.set_title(title, fontsize=10)
axes[0].set_ylabel("arcsec")

ax = axes[2]
ax.plot(r_grid, ee_analytic, color="#333", lw=2, ls="--", label="analytic obscured-Airy")
ax.plot(r_grid, ee_phased, color="#3898ec", lw=2, label="segmented, well-phased")
ax.plot(r_grid, ee_errored, color="#d97757", lw=2, label=f"segmented, {PISTON_RMS_NM:.0f} nm RMS piston")
ax.set_xlabel("aperture radius [arcsec]"); ax.set_ylabel("encircled energy")
ax.set_title(f"{LAM_UM} $\\mu$m", fontsize=10)
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.2)

fig.suptitle("3.5 m, 19-segment concept primary: physical-optics PSF (POPPY) vs the "
             "analytic model", fontsize=11, y=1.03)
fig.tight_layout()
fig.savefig("segmented_psf_demo.png", bbox_inches="tight")
print("wrote segmented_psf_demo.png")
