#!/usr/bin/env python3
"""Wide-field (10 arcsec radius) view of the 3.5 m 19-segment concept
primary's physical-optics PSF (segmented_psf.py, POPPY Fraunhofer/FFT),
displayed at the actual 3.5mST imaging-CCD pixel scale (0.11"/pix,
InstrumentConfig.pix_scale in slitless_etc.py) so the pixelated image shows
what the detector itself would actually record -- not an arbitrarily fine
display grid. Shows the segment-gap/support-strut diffraction structure well
outside the core, versus the analytic obscured-Airy model used by default in
slitless_etc.py.
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
FOV_ARCSEC = 20.0                 # +/-10 arcsec
DET_PIXSCALE = 0.11                # 3.5mST imaging-CCD plate scale (InstrumentConfig.pix_scale)
OVERSAMPLE = 4                      # internal accuracy oversampling; binned back to DET_PIXSCALE for display

psf_phased, _, _ = sps.monochromatic_psf(LAM_UM, piston_rms_nm=0.0, tilt_rms_urad=0.0,
                                         fov_arcsec=FOV_ARCSEC, pixelscale_arcsec=DET_PIXSCALE,
                                         oversample=OVERSAMPLE)
psf_errored, _, _ = sps.monochromatic_psf(LAM_UM, piston_rms_nm=PISTON_RMS_NM,
                                          tilt_rms_urad=TILT_RMS_URAD,
                                          fov_arcsec=FOV_ARCSEC, pixelscale_arcsec=DET_PIXSCALE,
                                          oversample=OVERSAMPLE)

# radial EE from the finely-sampled (oversampled) array -- accurate, not pixelated
r_grid = np.concatenate([np.linspace(0.02, 1.0, 40), np.linspace(1.2, 10.0, 40)])
ee_phased = sps.ee_curve(psf_phased, r_grid)
ee_errored = sps.ee_curve(psf_errored, r_grid)

cfg_analytic = etc.InstrumentConfig(diameter_cm=350.0, obstruction=0.15, psf_floor=0.0)
ee_analytic = np.array([etc.encircled_energy(cfg_analytic, LAM_UM * 1e4, r) for r in r_grid])

print(f"At {LAM_UM} um, out to r=10\" (EE from the finely-sampled array):")
for r in (0.3, 0.5, 1.0, 3.0, 5.0, 10.0):
    print(f"  r={r:5.1f}\"  analytic={np.interp(r, r_grid, ee_analytic):.4f}  "
          f"segmented(phased)={np.interp(r, r_grid, ee_phased):.4f}  "
          f"segmented(30nm)={np.interp(r, r_grid, ee_errored):.4f}")

# pixelated, detector-sampled images (0.11"/pix, flux-conserving block sum)
binned_phased, pix_scale = sps.bin_to_detector(psf_phased)
binned_errored, _ = sps.bin_to_detector(psf_errored)
print(f"detector-sampled image pixel scale: {pix_scale:.3f}\"/pix "
      f"({binned_phased.shape[0]}x{binned_phased.shape[0]} pixels over +/-10\")")

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), dpi=150)

for ax, img, title in [(axes[0], binned_phased, "segmented, well-phased"),
                       (axes[1], binned_errored, f"segmented, {PISTON_RMS_NM:.0f} nm RMS piston")]:
    half = img.shape[0] * pix_scale / 2.0
    ax.imshow(np.log10(img / img.max() + 1e-30), extent=(-half, half, -half, half),
              cmap="inferno", vmin=-6, vmax=0, origin="lower", interpolation="nearest")
    ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
    ax.set_xlabel("arcsec"); ax.set_title(title, fontsize=10)
axes[0].set_ylabel("arcsec")
fig.text(0.35, 0.965, f"pixel scale = {pix_scale:.2f}\" (3.5mST imaging CCD)",
         ha="center", fontsize=8, color="#555")

ax = axes[2]
ax.plot(r_grid, ee_analytic, color="#333", lw=2, ls="--", label="analytic obscured-Airy")
ax.plot(r_grid, ee_phased, color="#3898ec", lw=2, label="segmented, well-phased")
ax.plot(r_grid, ee_errored, color="#d97757", lw=2, label=f"segmented, {PISTON_RMS_NM:.0f} nm RMS piston")
ax.set_xlabel("aperture radius [arcsec]"); ax.set_ylabel("encircled energy")
ax.set_title(f"{LAM_UM} $\\mu$m, out to r=10\"", fontsize=10)
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.2)

fig.suptitle("3.5 m, 19-segment concept primary: physical-optics PSF (POPPY) out to "
             "r=10\", at the true detector pixel scale", fontsize=11, y=1.05)
fig.tight_layout()
fig.savefig("segmented_psf_wide.png", bbox_inches="tight")
print("wrote segmented_psf_wide.png")
