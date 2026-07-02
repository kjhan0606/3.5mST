#!/usr/bin/env python3
"""Low-cirrus imaging-survey fields: zoomed confirmation on the Planck 857 GHz
thermal-dust map, oversampled with smooth interpolation (the map's native
beam is 5 arcmin; the 15 arcsec WISE 12um WSSA atlas shows mostly
instrumental residuals in fields this clean), in the same TAN WCS style
as the ELG survey zoom.

The two fields are the minima of the mean Planck 857 GHz intensity over a
survey-footprint-sized aperture, searched over all |b| > 48 deg sky on the
2000x1000 all-sky rendering (see make_skymap.py). Each panel is a TAN zoom
with the RA/Dec graticule, the 1.5-deg ultra-deep field paved with its
30'x30' tiles, and the measured mean intensity annotated. In the north the
overlapping ELG wide tier on the Bootes anchor is outlined for context.
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
FIELDS = {"N": (210.0, 38.0), "S": (344.0, -48.0)}   # cirrus minima per cap
ELG_N = (218.0, 34.0)                                # Bootes ELG anchor
HALF = 0.75                                          # 3x3-tile deep field half-size [deg]
FOVTILE = 0.5                                        # 30' tile
VIEW = 8.0                                           # half-size of the view [deg]


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


def field_stats(ra0, dec0):
    """Mean Planck 857 GHz intensity within 1 deg (the field), and the all-sky median."""
    h = fits.open("hips_cache/planck857_allsky_car.fits")[0]
    d = h.data.astype(float)
    ny, nx = d.shape
    ra = h.header["CRVAL1"] + h.header["CDELT1"] * (np.arange(nx) + 1 - h.header["CRPIX1"])
    dec = h.header["CRVAL2"] + h.header["CDELT2"] * (np.arange(ny) + 1 - h.header["CRPIX2"])
    RA, DEC = np.meshgrid(ra, dec)
    c = SkyCoord(RA.ravel() * u.deg, DEC.ravel() * u.deg)
    sep = c.separation(SkyCoord(ra0 * u.deg, dec0 * u.deg)).deg.reshape(d.shape)
    w = np.cos(np.radians(DEC))
    m = (sep < 1.0) & np.isfinite(d)
    return np.sum(d[m] * w[m]) / np.sum(w[m]), np.nanmedian(d)


# shared display normalisation across both fields
cuts = []
for cap, (ra0, dec0) in FIELDS.items():
    data, hdr = fetch_hips(f"planck857_zoom{cap}.fits", hips="CDS/P/PLANCK/R3/HFI857",
                           width=1100, height=1100, fov=20.0, projection="TAN",
                           coordsys="icrs", ra=ra0, dec=dec0)
    cuts.append((cap, ra0, dec0, data, hdr))
allpix = np.concatenate([c[3].ravel() for c in cuts])
norm = LogNorm(vmin=np.nanpercentile(allpix, 2.0), vmax=np.nanpercentile(allpix, 99.8))

fig = plt.figure(figsize=(11.6, 5.9), dpi=150)
axes = []
for i, (cap, ra0, dec0, data, hdr) in enumerate(cuts):
    w = cap_wcs(ra0, dec0)
    ax = fig.add_subplot(1, 2, i + 1, projection=w)
    axes.append(ax)
    n = data.shape[1]
    hw = abs(hdr["CDELT1"]) * n / 2.0
    im = ax.imshow(data, origin="lower", extent=(-hw, hw, -hw, hw),
                   cmap="gray", norm=norm, interpolation="bicubic", zorder=0)
    # pave the imaging footprint with its 30' tiles, semi-transparent
    grid = np.arange(-HALF + FOVTILE / 2, HALF, FOVTILE)
    for cx in grid:
        for cy in grid:
            ax.add_patch(Rectangle((cx - FOVTILE / 2, cy - FOVTILE / 2),
                                   FOVTILE, FOVTILE, facecolor="#00e5ff",
                                   edgecolor="white", lw=0.1, alpha=0.22, zorder=2))
    ax.add_patch(Rectangle((-HALF, -HALF), 2 * HALF, 2 * HALF, fill=False,
                           edgecolor="#00e5ff", lw=2.2, zorder=3))
    ax.text(0, -HALF - 0.55, "ultra-deep imaging field (2.25 deg$^2$)",
            color="#00e5ff", fontsize=8.5, ha="center", va="top", zorder=5)
    if cap == "N":                                 # neighbouring ELG wide tier
        welg = cap_wcs(*ELG_N)
        EH = 6.25                                  # ELG wide-tier half-size
        t = np.linspace(-EH, EH, 160)
        per = ([(v, -EH) for v in t] + [(EH, v) for v in t]
               + [(v, EH) for v in t[::-1]] + [(-EH, v) for v in t[::-1]])
        ra_o, dec_o = welg.pixel_to_world_values([q[0] for q in per],
                                                 [q[1] for q in per])
        px, py = w.world_to_pixel_values(np.asarray(ra_o), np.asarray(dec_o))
        ax.plot(px, py, color="#f5b041", lw=2.0, zorder=4)
        ax.text(4.2, -6.4, "ELG wide tier\n(Boötes anchor)", color="#f5b041",
                fontsize=8.5, ha="center", zorder=5)
    mean, med = field_stats(ra0, dec0)
    gal = SkyCoord(ra0 * u.deg, dec0 * u.deg).galactic
    ax.text(0.03, 0.97,
            f"$\\langle I_{{857}}\\rangle$ = {mean:.2f} MJy sr$^{{-1}}$\n"
            f"all-sky median {med:.1f} MJy sr$^{{-1}}$",
            transform=ax.transAxes, color="white", fontsize=9, va="top", zorder=5)
    ax.set_xlim(-VIEW, VIEW); ax.set_ylim(-VIEW, VIEW)
    ax.coords.grid(color="0.8", alpha=0.55, ls=":")
    ax.coords[0].set_axislabel("R.A. (TAN)"); ax.coords[1].set_axislabel("Dec. (TAN)")
    ax.coords[0].set_major_formatter("d"); ax.coords[1].set_major_formatter("d")
    ax.coords[0].set_ticks(spacing=3 * u.deg); ax.coords[1].set_ticks(spacing=3 * u.deg)
    for cc, pos in ((0, "b"), (1, "l")):
        ax.coords[cc].set_ticks_position(pos)
        ax.coords[cc].set_ticklabel_position(pos)
        ax.coords[cc].set_axislabel_position(pos)
    ax.set_title(f"{cap} imaging field   $\\alpha={ra0:.0f}^\\circ$, "
                 f"$\\delta={dec0:+.0f}^\\circ$  ($b={gal.b.deg:+.0f}^\\circ$)",
                 fontsize=10.5, pad=8)
    print(f"{cap}: mean I857(7deg) = {mean:.3f}, median = {med:.2f}")
fig.subplots_adjust(left=0.06, right=0.86, top=0.86, bottom=0.09, wspace=0.16)
cax = fig.add_axes([0.885, 0.09, 0.02, 0.77])
cb = fig.colorbar(im, cax=cax)
cb.set_label(r"Planck 857 GHz intensity [MJy sr$^{-1}$]", fontsize=9)
fig.suptitle("Dedicated ultra-deep imaging fields (2.25 deg$^2$ each) at the Planck 857 GHz "
             "minima of the two caps (map native beam 5')", fontsize=11.5)
fig.savefig("imaging_fields_zoom.png", bbox_inches="tight")
print("wrote imaging_fields_zoom.png")
