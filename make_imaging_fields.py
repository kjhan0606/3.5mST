#!/usr/bin/env python3
"""Low-cirrus imaging-survey fields: zoomed confirmation on the Planck 857 GHz
thermal-dust map.

The two fields are the minima of the mean Planck 857 GHz intensity over a
survey-footprint-sized aperture, searched over all |b| > 48 deg sky on the
2000x1000 all-sky rendering (see make_skymap.py). Each panel is a gnomonic
(TAN) zoom on one field with the 12.5-deg wide-tier tiling boundary overlaid
and the measured mean intensity annotated, so the absence of cirrus structure
inside the footprint is verified directly on the data.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
import urllib.parse
import urllib.request

HIPS2FITS = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits?"


def fetch_hips(fname, **params):
    os.makedirs("hips_cache", exist_ok=True)
    path = os.path.join("hips_cache", fname)
    if not os.path.exists(path):
        url = HIPS2FITS + urllib.parse.urlencode({"format": "fits", **params})
        print("fetching", fname)
        urllib.request.urlretrieve(url, path)
    h = fits.open(path)[0]
    return h.data.astype(float), h.header


FIELDS = {"N": (210.0, 38.0), "S": (344.0, -48.0)}   # cirrus minima per cap
HALF = 6.25                                          # wide tier half-size [deg]
FOV = 40.0                                           # zoom field of view [deg]


def field_stats(ra0, dec0):
    """Mean Planck 857 GHz intensity within 7 deg of the centre, and the
    all-sky median, from the cached all-sky CAR rendering."""
    h = fits.open("hips_cache/planck857_allsky_car.fits")[0]
    d = h.data.astype(float)
    ny, nx = d.shape
    ra = h.header["CRVAL1"] + h.header["CDELT1"] * (np.arange(nx) + 1 - h.header["CRPIX1"])
    dec = h.header["CRVAL2"] + h.header["CDELT2"] * (np.arange(ny) + 1 - h.header["CRPIX2"])
    RA, DEC = np.meshgrid(ra, dec)
    c = SkyCoord(RA.ravel() * u.deg, DEC.ravel() * u.deg)
    sep = c.separation(SkyCoord(ra0 * u.deg, dec0 * u.deg)).deg.reshape(d.shape)
    w = np.cos(np.radians(DEC))
    m = (sep < 7.0) & np.isfinite(d)
    return np.sum(d[m] * w[m]) / np.sum(w[m]), np.nanmedian(d)


fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6), dpi=150)
fig.subplots_adjust(left=0.06, right=0.88, top=0.86, bottom=0.09, wspace=0.18)
ims = []
for ax, (cap, (ra0, dec0)) in zip(axes, FIELDS.items()):
    data, hdr = fetch_hips(f"planck857_img{cap}.fits", hips="CDS/P/PLANCK/R3/HFI857",
                           width=520, height=520, fov=FOV, projection="TAN",
                           coordsys="icrs", ra=ra0, dec=dec0)
    n = data.shape[1]
    hw = abs(hdr["CDELT1"]) * n / 2.0
    ims.append((ax, data, hw))
lo = np.nanpercentile(np.concatenate([d.ravel() for _, d, _ in ims]), 2.0)
hi = np.nanpercentile(np.concatenate([d.ravel() for _, d, _ in ims]), 99.8)
norm = LogNorm(vmin=lo, vmax=hi)
for (ax, data, hw), (cap, (ra0, dec0)) in zip(ims, FIELDS.items()):
    im = ax.imshow(data, origin="lower", extent=(-hw, hw, -hw, hw),
                   cmap="gray", norm=norm, interpolation="bilinear")
    ax.add_patch(Rectangle((-HALF, -HALF), 2 * HALF, 2 * HALF, fill=False,
                           edgecolor="#f5b041", lw=2.2))
    for t in np.arange(-HALF + 0.5, HALF, 0.5):     # 30' tiling grid, faint
        ax.plot([t, t], [-HALF, HALF], color="#f5b041", lw=0.15, alpha=0.5)
        ax.plot([-HALF, HALF], [t, t], color="#f5b041", lw=0.15, alpha=0.5)
    mean, med = field_stats(ra0, dec0)
    gal = SkyCoord(ra0 * u.deg, dec0 * u.deg).galactic
    ax.set_title(f"{cap} imaging field   $\\alpha={ra0:.0f}^\\circ$, "
                 f"$\\delta={dec0:+.0f}^\\circ$  ($b={gal.b.deg:+.0f}^\\circ$)",
                 fontsize=10.5)
    ax.text(0.03, 0.965,
            f"$\\langle I_{{857}}\\rangle$ = {mean:.2f} MJy sr$^{{-1}}$\n"
            f"all-sky median {med:.1f} MJy sr$^{{-1}}$",
            transform=ax.transAxes, color="white", fontsize=9, va="top")
    ax.set_xlabel(r"$\Delta$ tangent-plane [deg]")
    ax.set_ylabel(r"$\Delta$ tangent-plane [deg]")
    ax.set_xlim(-hw, hw); ax.set_ylim(-hw, hw)
    print(f"{cap}: mean I857(7deg) = {mean:.3f}, all-sky median = {med:.2f}")
cax = fig.add_axes([0.905, 0.09, 0.02, 0.77])
cb = fig.colorbar(im, cax=cax)
cb.set_label(r"Planck 857 GHz intensity [MJy sr$^{-1}$]", fontsize=9)
fig.suptitle("Low-cirrus imaging-survey fields, the Planck 857 GHz minima of the "
             "two caps, with the $12.5^\\circ$ wide-tier tiling overlaid", fontsize=11.5)
fig.savefig("imaging_fields_zoom.png", bbox_inches="tight")
print("wrote imaging_fields_zoom.png")
