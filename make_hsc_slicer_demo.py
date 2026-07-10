#!/usr/bin/env python3
"""Image-slicer IFU concept test, as an alternative to full-field slitless
dispersion (make_slitless_scene_sim.py / make_hsc_slitless_blend.py).

Instead of dispersing the whole 2D field in one direction, where any two
sources anywhere along the trace can overlap, an image slicer first cuts the
field of view into narrow horizontal slices; each slice is then dispersed
independently as its own mini long-slit spectrum. Sources in different
slices can never blend; only sources sharing (or adjacent to) the same slice
can still overlap in wavelength. This uses the same real HSC-SSP-detected
UDS/SXDS source scene (position, size, ellipticity, PA, flux) as
make_hsc_slitless_blend.py, with the same simulated FSPS-template spectra,
diced this way at three instrument roll angles, one row per angle: left, the
direct image with the slicer boundaries marked; right, each slice's own
dispersed spectrum (wavelength increasing upward), stacked at the same
vertical position as its slice in the direct image.

The number of slices (6) and the per-slice spatial resolution (the source's
elliptical Gaussian profile projected onto the slit's spatial axis) are
simplifications of a real image slicer's optics, chosen for a clear,
tractable illustration rather than a full instrument-level slicer design.
"""
import glob
import os

import numpy as np
from scipy import ndimage as ndi
from scipy.special import erf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch
from PIL import Image

RNG = np.random.default_rng(35)

PIX = 0.11                       # arcsec/pixel, concept instrument
LAM1, LAM2 = 1.00e4, 1.75e4
LAMC = 0.5 * (LAM1 + LAM2)
R = 1000.0
DLAM = LAMC / R / 2.0
NSTEP = int(round((LAM2 - LAM1) / DLAM))          # full trace resolution
NDISP = 300                       # downsampled per-slice display resolution
CROP = 800                       # displayed field, 800 px = 88 arcsec
PSF_FWHM = 0.12                  # arcsec, diffraction at 1.4 um for 3.5 m
NSLICE = 5

HSC_FILE = "hsc_uds_hires.png"
HSC_PIXSCALE = 0.3
TILE_ARCMIN = 30.0
TILE_PX = int(round(TILE_ARCMIN * 60.0 / HSC_PIXSCALE))
RENDER_CENTER_RC = (2183, 3383)    # same crowded sub-field as make_hsc_slitless_blend.py
DET_SIGMA = 3.0
MIN_AREA = 4
STAR_FWHM_MAX = 0.5
STAR_ELLIP_MAX = 0.2
BUFFER = int(CROP * 0.8)          # keep sources within this radius pre-rotation

# ---- detect real sources over the full 30'x30' tile (same recipe as
# ---- make_hsc_slitless_blend.py) --------------------------------------------
img = np.asarray(Image.open(HSC_FILE).convert("RGB")).astype(float)
lum = img[:TILE_PX, :TILE_PX, :].mean(axis=2)
bg = np.median(lum)
mad = np.median(np.abs(lum - bg)) * 1.4826
mask = lum > (bg + DET_SIGMA * mad)
labels, nlab = ndi.label(mask, structure=np.ones((3, 3)))
flux_img = np.clip(lum - bg, 0.0, None)
slices_obj = ndi.find_objects(labels)

rows, cols, tot_flux, fwhm_maj, ellip_l, pa_l = [], [], [], [], [], []
for k, sl in enumerate(slices_obj, start=1):
    if sl is None:
        continue
    sub_lab = labels[sl]
    sub_flux = flux_img[sl]
    m = sub_lab == k
    if m.sum() < MIN_AREA:
        continue
    w = sub_flux[m]
    f = w.sum()
    if f <= 0:
        continue
    yy, xx = np.mgrid[sl[0], sl[1]]
    yy, xx = yy[m], xx[m]
    cy = (w * yy).sum() / f
    cx = (w * xx).sum() / f
    ixx = (w * (xx - cx) ** 2).sum() / f
    iyy = (w * (yy - cy) ** 2).sum() / f
    ixy = (w * (xx - cx) * (yy - cy)).sum() / f
    trace, det = ixx + iyy, ixx * iyy - ixy ** 2
    disc = max(trace ** 2 / 4.0 - det, 0.0)
    lam1 = trace / 2.0 + np.sqrt(disc)
    lam2 = max(trace / 2.0 - np.sqrt(disc), 1e-6)
    pa = 0.5 * np.arctan2(2 * ixy, ixx - iyy)
    rows.append(cy); cols.append(cx); tot_flux.append(f)
    fwhm_maj.append(2.355 * np.sqrt(lam1) * HSC_PIXSCALE)
    ellip_l.append(1.0 - np.sqrt(lam2 / lam1))
    pa_l.append(pa)

rows, cols = np.array(rows), np.array(cols)
tot_flux = np.array(tot_flux)
fwhm_arcsec = np.clip(np.array(fwhm_maj), 0.05, None)
ellip = np.clip(np.array(ellip_l), 0.0, 0.9)
pa = np.array(pa_l)

scale = HSC_PIXSCALE / PIX
xC = (cols - RENDER_CENTER_RC[1]) * scale
yC = (rows - RENDER_CENTER_RC[0]) * scale
keep = np.hypot(xC, yC) < BUFFER * 1.5      # margin so 45 deg rotation still fills the crop
xC, yC = xC[keep], yC[keep]
tot_flux, fwhm_arcsec = tot_flux[keep], fwhm_arcsec[keep]
ellip, pa = ellip[keep], pa[keep]
NSRC = len(tot_flux)
print(f"{NSRC} real sources loaded around the render centre")

flux = tot_flux / tot_flux.min()
is_star = (fwhm_arcsec < STAR_FWHM_MAX) & (ellip < STAR_ELLIP_MAX)
fwhm = np.where(is_star, 0.0, fwhm_arcsec)
zred = RNG.uniform(0.4, 2.2, NSRC)

tnames = sorted(glob.glob("galaxy_templates/*.dat"))
tnames = [t for t in tnames if "index" not in t]
templates = [np.loadtxt(t) for t in tnames]
weights = np.array([2.0 if any(k in os.path.basename(n) for k in
                               ("Starburst", "Sc", "Sd", "Irregular")) else 1.0
                    for n in tnames])
tidx = RNG.choice(len(templates), NSRC, p=weights / weights.sum())

lam_full = LAM1 + DLAM * np.arange(NSTEP)
disp_idx = np.linspace(0, NSTEP - 1, NDISP).astype(int)
lam_disp = lam_full[disp_idx]


def source_spectrum(i):
    if is_star[i]:
        s = np.ones(NDISP)
    else:
        wl, fl = templates[tidx[i]][:, 0], templates[tidx[i]][:, 1]
        s = np.interp(lam_disp / (1.0 + zred[i]), wl, fl)
        s = np.clip(s, 0.0, None)
    m = s.mean()
    return s / m if m > 0 else np.ones(NDISP)


SPECTRA = np.array([source_spectrum(i) for i in range(NSRC)])     # (NSRC, NDISP)

sig_psf = PSF_FWHM / 2.355 / PIX
sig_maj0 = np.hypot(fwhm / 2.355 / PIX, sig_psf)
sig_min0 = np.hypot(fwhm * (1 - ellip) / 2.355 / PIX, sig_psf)


def kernel2d(i, pa_i):
    sig_maj, sig_min = sig_maj0[i], sig_min0[i]
    n = max(3, int(np.ceil(3.5 * sig_maj)))
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    c, s = np.cos(pa_i), np.sin(pa_i)
    xr, yr = c * xx + s * yy, -s * xx + c * yy
    k = np.exp(-0.5 * ((xr / sig_maj) ** 2 + (yr / sig_min) ** 2))
    return k / k.sum()


def splat2d(canvas, k, xc, yc, amp):
    n = k.shape[0] // 2
    ix, iy = int(round(xc)), int(round(yc))
    x1, x2 = ix - n, ix + n + 1
    y1, y2 = iy - n, iy + n + 1
    if x2 <= 0 or y2 <= 0 or x1 >= canvas.shape[1] or y1 >= canvas.shape[0]:
        return
    kx1, ky1 = max(0, -x1), max(0, -y1)
    kx2 = k.shape[1] - max(0, x2 - canvas.shape[1])
    ky2 = k.shape[0] - max(0, y2 - canvas.shape[0])
    canvas[max(0, y1):min(canvas.shape[0], y2), max(0, x1):min(canvas.shape[1], x2)] += \
        amp * k[ky1:ky2, kx1:kx2]


half = CROP / 2.0
slice_h = CROP / NSLICE
slice_h_px = CROP // NSLICE
xx_full = np.arange(CROP)
# the right column is NSLICE times as wide, on screen, as the left column, so
# each reformatted slice segment renders at exactly the width it was cut from
# in the direct image (both columns show the same CROP pixel width per slice)
fig = plt.figure(figsize=(5.2 * (1 + NSLICE), 15.0), dpi=100)
gs = GridSpec(3, 2, figure=fig, width_ratios=[1, NSLICE], wspace=0.035, hspace=0.22,
             left=0.012, right=0.998, top=0.98, bottom=0.012)

for row_i, angle in enumerate([0, 45, 90]):
    th = np.radians(angle)
    c, s = np.cos(th), np.sin(th)
    xr = xC * c - yC * s
    yr = xC * s + yC * c
    par = pa - th
    inbox = (np.abs(xr) < half) & (np.abs(yr) < half)
    xi, yi = xr[inbox] + half, yr[inbox] + half        # crop pixel coords, 0..CROP
    fi = flux[inbox]
    pai = par[inbox]
    smaj_i, smin_i = sig_maj0[inbox], sig_min0[inbox]
    spec_i = SPECTRA[inbox]
    idxs = np.where(inbox)[0]

    # ---- left: direct image with the slicer grid ----
    axL = fig.add_subplot(gs[row_i, 0])
    dimg = np.zeros((CROP, CROP))
    for j in range(len(xi)):
        k = kernel2d(idxs[j], pai[j])
        splat2d(dimg, k, xi[j], yi[j], fi[j])
    stretch = np.arcsinh(dimg / (1.6 * np.std(dimg[dimg < np.percentile(dimg, 90)]) + 1e-9))
    axL.imshow(-stretch, cmap="gray", origin="lower", extent=(0, CROP, 0, CROP),
              vmin=-np.percentile(stretch, 99.8), vmax=0.6)
    for si in range(1, NSLICE):
        axL.axhline(si * slice_h, color="#c0392b", lw=2.0, ls="--")
    for si in range(NSLICE):
        axL.text(10, si * slice_h + slice_h / 2, f"{si + 1}", color="#c0392b",
                 fontsize=20, va="center", ha="left", fontweight="bold")
    axL.set_xticks([]); axL.set_yticks([])
    axL.set_title(f"roll {angle}$^\\circ$: direct image, {NSLICE} slices", fontsize=26)

    # ---- right: the slices reformatted into ONE long horizontal pseudo-slit
    # ---- (bottom), dispersed together into a single spectrum (top, wavelength
    # ---- increasing upward) -- the real image-slicer detector layout, where
    # ---- each slice occupies its own segment of the combined frame. The
    # ---- pseudo-slit band shows the ACTUAL sliced sub-images of the direct
    # ---- image (dimg), not a re-synthesised profile.
    combined_spec = np.zeros((NDISP, NSLICE * CROP))
    slit_img = np.zeros((slice_h_px, NSLICE * CROP))
    for si in range(NSLICE):
        ylo, yhi = si * slice_h, (si + 1) * slice_h
        seg = slice(si * CROP, (si + 1) * CROP)
        row0 = int(round(ylo))
        slit_img[:, seg] = dimg[row0:row0 + slice_h_px, :]
        # an extended source is not assigned wholesale to whichever slice its
        # centre falls in: the FRACTION of its elliptical profile that actually
        # falls within [ylo,yhi) is what a real slice intercepts, so a source
        # straddling a boundary contributes (partially) to both neighbouring
        # slices, matching the real sliced sub-image already shown in slit_img
        for j in range(len(xi)):
            sy2 = (smaj_i[j] ** 2 * np.sin(pai[j]) ** 2
                  + smin_i[j] ** 2 * np.cos(pai[j]) ** 2)
            sy = max(np.sqrt(sy2), 0.6)
            frac = 0.5 * (erf((yhi - yi[j]) / (sy * np.sqrt(2)))
                         - erf((ylo - yi[j]) / (sy * np.sqrt(2))))
            if frac < 1e-3:
                continue
            sx2 = (smaj_i[j] ** 2 * np.cos(pai[j]) ** 2
                  + smin_i[j] ** 2 * np.sin(pai[j]) ** 2)
            sx = max(np.sqrt(sx2), 0.6)
            profile = np.exp(-0.5 * ((xx_full - xi[j]) / sx) ** 2)
            profile /= profile.sum() + 1e-12
            # spec_i is unit-mean over NDISP samples; dividing by NDISP conserves
            # total flux across the dispersed trace (same convention as dividing
            # by NSTEP in make_hsc_slitless_blend.py), instead of depositing the
            # full flux into every wavelength bin
            combined_spec[:, seg] += np.outer(spec_i[j] / NDISP, profile) * fi[j] * frac

    # axSlit gets exactly 1/NSLICE of the row height, the correct proportion
    # for the sliced image's true (unmagnified) pixel scale. It uses
    # aspect="auto" (like axSpec) rather than "equal" so it fills its box
    # exactly and its x-axis stays pixel-for-pixel aligned with axSpec above
    # -- aspect="equal" can letterbox within the box and desync the two.
    gsR = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[row_i, 1],
                                 height_ratios=[NSLICE - 1, 1], hspace=0.03)
    axSpec = fig.add_subplot(gsR[0, 0])
    axSlit = fig.add_subplot(gsR[1, 0], sharex=axSpec)
    # same contrast convention as the dispersed panels of make_hsc_slitless_blend.py
    # (dispersed sky=0.15 plus its noise, then arcsinh of the median-subtracted,
    # sub-90th-percentile-scaled residual), so the two figures are comparable
    spec_sky = 0.15
    noisy_spec = combined_spec + spec_sky + RNG.normal(0.0, np.sqrt(spec_sky) * 0.028,
                                                       combined_spec.shape)
    v_spec = noisy_spec - np.median(noisy_spec)
    st = np.arcsinh(v_spec / (1.6 * np.std(v_spec[v_spec < np.percentile(v_spec, 90)])))
    axSpec.imshow(-st, cmap="gray", origin="lower", aspect="auto",
                 extent=(0, NSLICE * CROP, 0, NDISP),
                 vmin=-np.percentile(st, 99.8), vmax=0.6)
    slit_stretch = np.arcsinh(slit_img / (1.6 * np.std(dimg[dimg < np.percentile(dimg, 90)]) + 1e-9))
    axSlit.imshow(-slit_stretch, cmap="gray", origin="lower", aspect="auto",
                 extent=(0, NSLICE * CROP, 0, slice_h_px),
                 vmin=-np.percentile(stretch, 99.8), vmax=0.6)
    for si in range(1, NSLICE):
        axSpec.axvline(si * CROP, color="#c0392b", lw=1.6, ls="--")
        axSlit.axvline(si * CROP, color="#c0392b", lw=1.6, ls="--")
    for si in range(NSLICE):
        axSlit.text((si + 0.5) * CROP, slice_h_px / 2, f"{si + 1}", color="#c0392b",
                   fontsize=20, ha="center", va="center", fontweight="bold")
    axSpec.set_xticks([]); axSpec.set_yticks([])
    axSlit.set_xticks([]); axSlit.set_yticks([])
    # wavelength-direction arrow + label, placed in FIGURE (not axes) fraction
    # coordinates at a small fixed gap from axSpec's own left edge -- axSpec is
    # NSLICE times wider than it is tall, so an axes-fraction offset would land
    # far from the axis; anchoring to its real bounding box keeps the gap fixed.
    # Drawn unrotated so the arrow actually points up (a rotated arrow glyph
    # inside a rotated ylabel does not point up on screen).
    posS = axSpec.get_position()
    ax_x = posS.x0 - 0.007
    y_lo, y_hi = posS.y0 + 0.08 * posS.height, posS.y1 - 0.08 * posS.height
    fig.add_artist(FancyArrowPatch((ax_x, y_lo), (ax_x, y_hi), transform=fig.transFigure,
                                  arrowstyle="-|>", mutation_scale=36, lw=3.2, color="black"))
    fig.text(ax_x - 0.016, 0.5 * (y_lo + y_hi), "$\\lambda$", fontsize=28,
            ha="center", va="center")
    axSlit.set_xlabel("pseudo-slit position (slices laid end to end)", fontsize=20)
    axSpec.set_title(f"roll {angle}$^\\circ$: slices reformatted into one "
                     "pseudo-slit, dispersed together", fontsize=26)

fig.savefig("hsc_slicer_demo.png", bbox_inches="tight")
fig.savefig("hsc_slicer_demo.png", bbox_inches="tight")
print("wrote hsc_slicer_demo.png")
