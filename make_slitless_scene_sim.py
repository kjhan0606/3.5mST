#!/usr/bin/env python3
"""Simulated crowded-field slitless exposure for the proposed instrument
(Figure fig:crowded).

A random extragalactic field is rendered as the direct image and as slitless
exposures at the three survey roll angles. The instrument numbers are the
proposal baseline, 0.11 arcsec pixels and the 1.0-1.75 micron band-limited
near-infrared setting at R = 1000 defined at the setting centre, which gives
constant dispersion dlam = lam_c/(2R) = 6.9 A per pixel and first-order traces
of 1091 pixels. Source spectra are the repository's FSPS galaxy templates
(with nebular emission lines) redshifted to 0.4 < z < 2.2, so emission lines
appear as compact knots on the long continuum traces. Magnitudes follow
dN/dm ~ 10^{0.3m} over 18.5 < m < 24.5 and the source density is 40 per
arcmin^2, representative of deep near-infrared counts at these depths. The
displayed field is 88 x 88 arcsec; traces from sources outside the crop enter
it, as on a real detector.
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(35)

PIX = 0.11                       # arcsec / pixel
LAM1, LAM2 = 1.00e4, 1.75e4      # band-limited NIR setting [A]
LAMC = 0.5 * (LAM1 + LAM2)
R = 1000.0
DLAM = LAMC / R / 2.0            # constant dispersion, 2 pix per res element
NSTEP = int(round((LAM2 - LAM1) / DLAM))          # 1091 trace pixels
CROP = 800                       # displayed field, 800 px = 88 arcsec
CANVAS = CROP + 2 * NSTEP + 200  # sources beyond the crop still leave traces
DENSITY = 40.0                   # sources / arcmin^2
M1, M2 = 18.5, 24.5              # AB magnitude range
SLOPE = 0.3                      # dN/dm ~ 10^{SLOPE m}
PSF_FWHM = 0.12                  # arcsec, diffraction at 1.4 um for 3.5 m
STAR_FRAC = 0.05

area_arcmin2 = (CANVAS * PIX / 60.0) ** 2
NSRC = RNG.poisson(DENSITY * area_arcmin2)

# ---- source population ------------------------------------------------------
u = RNG.random(NSRC)             # inverse-CDF sample of dN/dm ~ 10^{am}
a = SLOPE * np.log(10.0)
mags = np.log(np.exp(a * M1) + u * (np.exp(a * M2) - np.exp(a * M1))) / a
flux = 10.0 ** (-0.4 * (mags - M2))               # relative flux, faintest = 1
x0 = RNG.uniform(0, CANVAS, NSRC)
y0 = RNG.uniform(0, CANVAS, NSRC)
is_star = RNG.random(NSRC) < STAR_FRAC
fwhm = np.where(is_star, 0.0, RNG.uniform(0.15, 0.9, NSRC))   # arcsec
ellip = RNG.uniform(0.0, 0.6, NSRC)
pa = RNG.uniform(0, np.pi, NSRC)
zred = RNG.uniform(0.4, 2.2, NSRC)

tnames = sorted(glob.glob("galaxy_templates/*.dat"))
tnames = [t for t in tnames if "index" not in t]
templates = [np.loadtxt(t) for t in tnames]
weights = np.array([2.0 if any(k in os.path.basename(n) for k in
                               ("Starburst", "Sc", "Sd", "Irregular")) else 1.0
                    for n in tnames])
tidx = RNG.choice(len(templates), NSRC, p=weights / weights.sum())

lam_obs = LAM1 + DLAM * np.arange(NSTEP)


def source_spectrum(i):
    """Relative f_lambda of source i on the observed trace grid, unit mean."""
    if is_star[i]:
        s = np.ones(NSTEP)
    else:
        wl, fl = templates[tidx[i]][:, 0], templates[tidx[i]][:, 1]
        s = np.interp(lam_obs / (1.0 + zred[i]), wl, fl)
        s = np.clip(s, 0.0, None)
    m = s.mean()
    return s / m if m > 0 else np.ones(NSTEP)


def kernel(i):
    """Source profile convolved with the PSF, normalised to unit sum."""
    sig_psf = PSF_FWHM / 2.355 / PIX
    sig_maj = np.hypot(fwhm[i] / 2.355 / PIX, sig_psf)
    sig_min = np.hypot(fwhm[i] * (1 - ellip[i]) / 2.355 / PIX, sig_psf)
    n = max(3, int(np.ceil(3.5 * sig_maj)))
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    c, s = np.cos(pa[i]), np.sin(pa[i])
    xr, yr = c * xx + s * yy, -s * xx + c * yy
    k = np.exp(-0.5 * ((xr / sig_maj) ** 2 + (yr / sig_min) ** 2))
    return k / k.sum()


def splat(img, k, xc, yc, amp):
    n = k.shape[0] // 2
    ix, iy = int(round(xc)), int(round(yc))
    x1, x2 = ix - n, ix + n + 1
    y1, y2 = iy - n, iy + n + 1
    if x2 <= 0 or y2 <= 0 or x1 >= img.shape[1] or y1 >= img.shape[0]:
        return
    kx1, ky1 = max(0, -x1), max(0, -y1)
    kx2 = k.shape[1] - max(0, x2 - img.shape[1])
    ky2 = k.shape[0] - max(0, y2 - img.shape[0])
    img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)] += \
        amp * k[ky1:ky2, kx1:kx2]


def render(angle_deg):
    """Slitless exposure with dispersion along angle_deg (None = direct)."""
    img = np.zeros((CANVAS, CANVAS))
    if angle_deg is None:
        for i in range(NSRC):
            splat(img, kernel(i), x0[i], y0[i], flux[i])
        return img
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    for i in range(NSRC):
        k = kernel(i)
        spec = flux[i] * source_spectrum(i) / NSTEP
        # skip traces that never touch the displayed crop
        lo, hi = (CANVAS - CROP) / 2, (CANVAS + CROP) / 2
        xs, ys = x0[i] + c * np.array([0, NSTEP]), y0[i] + s * np.array([0, NSTEP])
        if (max(xs) < lo - 60 or min(xs) > hi + 60
                or max(ys) < lo - 60 or min(ys) > hi + 60):
            continue
        step = 2                                     # 2-px steps for speed
        for j in range(0, NSTEP, step):
            splat(img, k, x0[i] + c * j, y0[i] + s * j,
                  spec[j:j + step].sum())
    return img


def crop(img):
    o = (CANVAS - CROP) // 2
    return img[o:o + CROP, o:o + CROP]


panels, titles = [], []
for ang, ttl in [(None, "direct image"), (0, r"grism, roll $0^\circ$"),
                 (45, r"grism, roll $45^\circ$"), (90, r"grism, roll $90^\circ$")]:
    print("rendering", ttl)
    img = crop(render(ang))
    sky = 0.15 if ang is not None else 0.03          # dispersed sky is brighter
    img = img + sky + RNG.normal(0.0, np.sqrt(sky) * 0.028, img.shape)
    panels.append(img)
    titles.append(ttl)

fig, axes = plt.subplots(1, 4, figsize=(19.6, 5.35), dpi=170)
for ax, img, ttl in zip(axes.ravel(), panels, titles):
    v = img - np.median(img)
    stretch = np.arcsinh(v / (1.6 * np.std(v[v < np.percentile(v, 90)])))
    ax.imshow(-stretch, cmap="gray", origin="lower",
              vmin=-np.percentile(stretch, 99.8), vmax=0.6)
    ax.set_title(ttl, fontsize=12)
    ax.set_xticks([]), ax.set_yticks([])
for ax, ang in zip(axes.ravel()[1:], (0, 45, 90)):
    c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    ax.annotate("", xy=(60 + 90 * c, 60 + 90 * s), xytext=(60, 60),
                arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.8))
    ax.text(46, 24, "dispersion", color="#c0392b", fontsize=9)
axes[0].plot([CROP - 91 - 20, CROP - 20], [30, 30], color="k", lw=2)
axes[0].text(CROP - 65 - 20, 42, r"$10''$", fontsize=10, ha="center")
fig.tight_layout(rect=(0, 0, 1, 0.965))
fig.savefig("slitless_scene_sim.png", bbox_inches="tight")
print(f"wrote slitless_scene_sim.png  ({NSRC} sources over "
      f"{area_arcmin2:.1f} arcmin^2, {NSTEP}-pixel traces)")
