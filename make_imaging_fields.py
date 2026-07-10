#!/usr/bin/env python3
"""The four dedicated imaging fields on the Planck 857 GHz thermal-dust map,
in the gnomonic (TAN) WCS style of the ELG survey zoom.

Top row: the two ultra-deep fields (3x3 tiles, 1.5 deg, 2.25 deg^2), the
minima of the mean 857 GHz intensity over a 7-deg aperture across all
|b| > 48 deg sky. Bottom row: the two imaging-only wide-extension fields
(35x35 tiles, 17.5 deg, ~306 deg^2), the minima with a matching 9.9-deg
aperture over |b| > 55 deg; the northern extension sits beside the Lockman
Hole and the southern near the south Galactic pole. Each panel annotates the
footprint-mean intensity. The map's native beam is 5 arcmin, so the deeper
zooms of the top row appear smoother.

The SN Ia time-domain cadence fields are shown instead in the spectroscopic
wide-tier zoom, make_skymap.py's elg_deepfields_zoom.png, since they are
placed beside that footprint rather than at a separate low-cirrus minimum.
"""
import os
import urllib.parse
import urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.visualization.wcsaxes            # registers the WCS projection

HIPS2FITS = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits?"
FOVTILE = 0.5                                   # 30' tile
# (label, ra, dec, half-size [deg], view half-size [deg], stat radius [deg], colour)
PANELS = [
    ("N ultra-deep field", 210.0, 38.0, 0.75, 8.0, 1.0, "#00e5ff"),
    ("S ultra-deep field", 344.0, -48.0, 0.75, 8.0, 1.0, "#00e5ff"),
    ("N wide extension (Lockman Hole)", 162.0, 52.0, 8.75, 26.0, 9.87, "#ff5cf0"),
    ("S wide extension (south polar cap)", 3.4, -38.0, 8.75, 26.0, 9.87, "#ff5cf0"),
]


def fetch_hips(fname, **params):
    os.makedirs("hips_cache", exist_ok=True)
    path = os.path.join("hips_cache", fname)
    if not os.path.exists(path):
        url = HIPS2FITS + urllib.parse.urlencode({"format": "fits", **params})
        print("fetching", fname)
        urllib.request.urlretrieve(url, path)
    h = fits.open(path)[0]
    return h.data.astype(float), h.header


def cap_wcs(ra0, dec0):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra0, dec0]
    w.wcs.crpix = [1.0, 1.0]
    w.wcs.cdelt = [-1.0, 1.0]
    return w


ALLSKY = fits.open("hips_cache/planck857_allsky_car.fits")[0]


def field_stats(ra0, dec0, radius):
    """Mean Planck 857 GHz intensity within `radius` deg, and all-sky median."""
    h = ALLSKY
    d = h.data.astype(float)
    ny, nx = d.shape
    ra = h.header["CRVAL1"] + h.header["CDELT1"] * (np.arange(nx) + 1 - h.header["CRPIX1"])
    dec = h.header["CRVAL2"] + h.header["CDELT2"] * (np.arange(ny) + 1 - h.header["CRPIX2"])
    RA, DEC = np.meshgrid(ra, dec)
    c = SkyCoord(RA.ravel() * u.deg, DEC.ravel() * u.deg)
    sep = c.separation(SkyCoord(ra0 * u.deg, dec0 * u.deg)).deg.reshape(d.shape)
    w = np.cos(np.radians(DEC))
    m = (sep < radius) & np.isfinite(d)
    return np.sum(d[m] * w[m]) / np.sum(w[m]), np.nanmedian(d)


cuts = []
for i, (lab, ra0, dec0, half, view, rstat, col) in enumerate(PANELS):
    fov = 2.2 * view
    data, hdr = fetch_hips(f"planck857_p{i}_{ra0:.0f}_{dec0:+.0f}_v{view:g}.fits",
                           hips="CDS/P/PLANCK/R3/HFI857", width=900, height=900,
                           fov=fov, projection="TAN", coordsys="icrs",
                           ra=ra0, dec=dec0)
    cuts.append((lab, ra0, dec0, half, view, rstat, col, data, hdr))
allpix = np.concatenate([c[7].ravel() for c in cuts])
norm = LogNorm(vmin=np.nanpercentile(allpix, 2.0), vmax=np.nanpercentile(allpix, 99.8))

fig = plt.figure(figsize=(11.6, 11.4), dpi=150)
for i, (lab, ra0, dec0, half, view, rstat, col, data, hdr) in enumerate(cuts):
    w = cap_wcs(ra0, dec0)
    ax = fig.add_subplot(2, 2, i + 1, projection=w)
    n = data.shape[1]
    hw = abs(hdr["CDELT1"]) * n / 2.0
    im = ax.imshow(data, origin="lower", extent=(-hw, hw, -hw, hw),
                   cmap="gray", norm=norm, interpolation="bicubic", zorder=0)
    grid = np.arange(-half + FOVTILE / 2, half, FOVTILE)
    for cx in grid:
        for cy in grid:
            ax.add_patch(Rectangle((cx - FOVTILE / 2, cy - FOVTILE / 2),
                                   FOVTILE, FOVTILE, facecolor=col,
                                   edgecolor="white", lw=0.05, alpha=0.18, zorder=2))
    ax.add_patch(Rectangle((-half, -half), 2 * half, 2 * half, fill=False,
                           edgecolor=col, lw=2.2, zorder=3))
    mean, med = field_stats(ra0, dec0, rstat)
    gal = SkyCoord(ra0 * u.deg, dec0 * u.deg).galactic
    ax.text(0.03, 0.97,
            f"$\\langle I_{{857}}\\rangle$ = {mean:.2f} MJy sr$^{{-1}}$\n"
            f"all-sky median {med:.1f} MJy sr$^{{-1}}$",
            transform=ax.transAxes, color="white", fontsize=9, va="top", zorder=5)
    area = (2 * half) ** 2
    ax.text(0, -half - 0.06 * view, f"{area:.4g} deg$^2$", color=col,
            fontsize=8.5, ha="center", va="top", zorder=5)
    ax.set_xlim(-view, view); ax.set_ylim(-view, view)
    ax.coords.grid(color="0.8", alpha=0.55, ls=":")
    ax.coords[0].set_axislabel("R.A. (TAN)"); ax.coords[1].set_axislabel("Dec. (TAN)")
    ax.coords[0].set_major_formatter("d"); ax.coords[1].set_major_formatter("d")
    sp = 3 if view < 10 else 10
    ax.coords[0].set_ticks(spacing=sp * u.deg); ax.coords[1].set_ticks(spacing=sp * u.deg)
    for cc, pos in ((0, "b"), (1, "l")):
        ax.coords[cc].set_ticks_position(pos)
        ax.coords[cc].set_ticklabel_position(pos)
        ax.coords[cc].set_axislabel_position(pos)
    ax.set_title(f"{lab}   $\\alpha={ra0:.0f}^\\circ$, $\\delta={dec0:+.0f}^\\circ$"
                 f"  ($b={gal.b.deg:+.0f}^\\circ$)", fontsize=10, pad=8)
    print(f"{lab}: mean I857({rstat:g} deg) = {mean:.3f}")
fig.subplots_adjust(left=0.07, right=0.86, top=0.93, bottom=0.05,
                    wspace=0.18, hspace=0.22)
cax = fig.add_axes([0.885, 0.05, 0.02, 0.88])
cb = fig.colorbar(im, cax=cax)
cb.set_label(r"Planck 857 GHz intensity [MJy sr$^{-1}$]", fontsize=9)
fig.savefig("imaging_fields_zoom.png", bbox_inches="tight")
print("wrote imaging_fields_zoom.png")
