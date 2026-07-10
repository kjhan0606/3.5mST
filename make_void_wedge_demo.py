#!/usr/bin/env python3
"""Sky-projection (Mollweide) view of the one void of the independent Bayesian
catalog of Malandrino, Lavaux, Wandelt, McAlpine & Jasche (2026, A&A 705,
A160; SDSS/2MRS-based) that falls inside the flagship Bo\"otes wide
spectroscopic tier (Section 3.2.7), the field where the mission's own
slitless spectroscopy, not just its imaging, actually reaches. Real SDSS-I/II
Main Galaxy Sample spectroscopy (survey='sdss', not BOSS/eBOSS) is queried
live from SkyServer DR17 over the void's own radial extent (its comoving
distance +/- its catalogued radius) and classified by true three-dimensional
comoving distance from the void center: inside the void's sphere (green)
versus outside it (blue). A same-shell, angle-only cut (the void's projected
disk on the sky) is NOT the same selection -- most of that disk's galaxies
sit in front of or behind the sphere along the line of sight, so the
classification here uses the full 3D separation. The field's own Planck 857
GHz cirrus map is reprojected through the same Mollweide transform as the
background, and the whole figure is mirrored to match the view looking up at
the sky (RA increasing to the left) rather than looking down at a map.

All three external data products (SDSS query, the void catalog, and the
Planck cirrus cutout) are cached locally after the first run; delete the
cache files to re-query.
"""
import os
import subprocess
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates
from mpl_toolkits.axes_grid1 import make_axes_locatable

CACHE_SDSS = "void_wedge_sdss_cache.csv"
CACHE_VOIDS = "void_wedge_catalog_cache.csv"
CIRRUS_FITS = "hips_cache/planck857_pvoid_218_+34_v25.fits"
HIPS2FITS = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits?"

# Bo\"otes wide-tier spectroscopic footprint, 12.5 deg on a side centered on
# (218, 34) (make_skymap.py), not the imaging-only Lockman Hole extension
RA_FIELD_LO, RA_FIELD_HI = 211.75, 224.25
DEC_FIELD_LO, DEC_FIELD_HI = 27.75, 40.25
RA_QUERY_LO, RA_QUERY_HI = 201.75, 234.25
DEC_QUERY_LO, DEC_QUERY_HI = 17.75, 50.25

# the single Malandrino et al. (2026) void whose center falls inside the field
RA0, DEC0, Z0 = 222.797488, 39.402638, 0.046735


def fetch_voids():
    if not os.path.exists(CACHE_VOIDS):
        url = "https://raw.githubusercontent.com/RosaMalandrino/LocalVoids/master/voids_catalog.csv"
        subprocess.run(["curl", "-s", "--max-time", "30", url, "-o", CACHE_VOIDS], check=True)
    df = pd.read_csv(CACHE_VOIDS)
    df.columns = [c.strip() for c in df.columns]
    return df


RADIUS_SCALE = 0.50    # restrict to the void's innermost interior, away from the wall population

voids = fetch_voids()
row = voids[np.isclose(voids["center RA [deg]"], RA0) & np.isclose(voids["center Dec [deg]"], DEC0)].iloc[0]
R_VOID = row["mean radius (Mpc/h)"] / Planck18.h * RADIUS_SCALE      # comoving Mpc
D0 = Planck18.comoving_distance(Z0).to(u.Mpc).value
ANG_RAD_DEG = np.degrees(R_VOID / D0)


def fetch_sdss():
    if not os.path.exists(CACHE_SDSS):
        d_near, d_far = D0 - R_VOID, D0 + R_VOID
        from astropy.cosmology import z_at_value
        z_near = z_at_value(Planck18.comoving_distance, d_near * u.Mpc)
        z_far = z_at_value(Planck18.comoving_distance, d_far * u.Mpc)
        q = (f"SELECT ra,dec,z FROM SpecObj WHERE class='GALAXY' AND survey='sdss' "
             f"AND ra BETWEEN {RA_QUERY_LO} AND {RA_QUERY_HI} "
             f"AND dec BETWEEN {DEC_QUERY_LO} AND {DEC_QUERY_HI} "
             f"AND z BETWEEN {z_near.value:.6f} AND {z_far.value:.6f}")
        url = ("https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch?cmd="
               + urllib.parse.quote(q) + "&format=csv")
        subprocess.run(["curl", "-s", "--max-time", "90", url, "-o", CACHE_SDSS], check=True)
    return pd.read_csv(CACHE_SDSS, skiprows=1)


def mollweide_xy(ra_deg, dec_deg, ra0_deg):
    """Standard Mollweide forward transform, centered on ra0_deg."""
    lam = np.radians(np.remainder(np.asarray(ra_deg) - ra0_deg + 180.0, 360.0) - 180.0)
    phi = np.radians(np.asarray(dec_deg))
    theta = phi.copy()
    for _ in range(100):
        f = 2 * theta + np.sin(2 * theta) - np.pi * np.sin(phi)
        fp = 2 + 2 * np.cos(2 * theta)
        theta = theta - f / fp
    x = (2 * np.sqrt(2) / np.pi) * lam * np.cos(theta)
    y = np.sqrt(2) * np.sin(theta)
    return x, y


df = fetch_sdss()

ra0r, dec0r = np.radians(RA0), np.radians(DEC0)
x0 = D0 * np.cos(dec0r) * np.cos(ra0r)
y0 = D0 * np.cos(dec0r) * np.sin(ra0r)
z0c = D0 * np.sin(dec0r)

Dg = Planck18.comoving_distance(df["z"].values).to(u.Mpc).value
rag, decg = np.radians(df["ra"].values), np.radians(df["dec"].values)
xg = Dg * np.cos(decg) * np.cos(rag)
yg = Dg * np.cos(decg) * np.sin(rag)
zg = Dg * np.sin(decg)
d3d = np.sqrt((xg - x0) ** 2 + (yg - y0) ** 2 + (zg - z0c) ** 2)
member = d3d < R_VOID

if not os.path.exists(CIRRUS_FITS):
    os.makedirs("hips_cache", exist_ok=True)
    url = HIPS2FITS + urllib.parse.urlencode({
        "format": "fits", "hips": "CDS/P/PLANCK/R3/HFI857", "width": 900, "height": 900,
        "fov": 55.0, "projection": "TAN", "coordsys": "icrs", "ra": 218.0, "dec": 34.0})
    urllib.request.urlretrieve(url, CIRRUS_FITS)
hdu = fits.open(CIRRUS_FITS)[0]
src_data = hdu.data.astype(float)
src_wcs = WCS(hdu.header)
NGRID = 400
ra_grid = np.linspace(RA_QUERY_LO - 5, RA_QUERY_HI + 5, NGRID)
dec_grid = np.linspace(DEC_QUERY_LO - 2, DEC_QUERY_HI + 2, NGRID)
RA2, DEC2 = np.meshgrid(ra_grid, dec_grid)
xarr, yarr = src_wcs.world_to_pixel_values(RA2, DEC2)
cirrus = map_coordinates(src_data, [yarr, xarr], order=1, mode="constant", cval=np.nan)
MX, MY = mollweide_xy(RA2, DEC2, RA0)

mx_out, my_out = mollweide_xy(df["ra"].values[~member], df["dec"].values[~member], RA0)
mx_in, my_in = mollweide_xy(df["ra"].values[member], df["dec"].values[member], RA0)

fig, ax = plt.subplots(figsize=(9, 7), dpi=170)
pcm = ax.pcolormesh(MX, MY, cirrus, cmap="gray", shading="auto",
                     vmin=np.nanpercentile(cirrus, 2), vmax=np.nanpercentile(cirrus, 98), zorder=0)

ax.scatter(mx_out, my_out, s=8, c="tab:blue", alpha=0.75, zorder=2,
           label=f"outside void, 3D (N={(~member).sum()})")
ax.scatter(mx_in, my_in, s=8, c="tab:green", alpha=0.85, zorder=3,
           label=f"inside void, 3D (N={member.sum()})")

fbox_ra = np.concatenate([np.linspace(RA_FIELD_LO, RA_FIELD_HI, 40), np.full(40, RA_FIELD_HI),
                          np.linspace(RA_FIELD_HI, RA_FIELD_LO, 40), np.full(40, RA_FIELD_LO)])
fbox_dec = np.concatenate([np.full(40, DEC_FIELD_LO), np.linspace(DEC_FIELD_LO, DEC_FIELD_HI, 40),
                           np.full(40, DEC_FIELD_HI), np.linspace(DEC_FIELD_HI, DEC_FIELD_LO, 40)])
mx_f, my_f = mollweide_xy(fbox_ra, fbox_dec, RA0)
ax.plot(mx_f, my_f, color="crimson", lw=2, zorder=4, label="spectroscopic wide-tier boundary")

theta_c = np.linspace(0, 2 * np.pi, 300)
cosd0 = np.cos(np.radians(DEC0))
void_ra = RA0 + (ANG_RAD_DEG / cosd0) * np.cos(theta_c)
void_dec = DEC0 + ANG_RAD_DEG * np.sin(theta_c)
mx_v, my_v = mollweide_xy(void_ra, void_dec, RA0)
ax.plot(mx_v, my_v, color="darkorange", lw=2.2, zorder=4, label="void angular boundary")
mx_c, my_c = mollweide_xy([RA0], [DEC0], RA0)
ax.plot(mx_c, my_c, "+", color="darkorange", ms=12, zorder=5)

# RA/Dec graticule in place of the raw (unitless) Mollweide x/y ticks
RA_TICKS = np.arange(210, 241, 10)
DEC_TICKS = np.arange(20, 51, 10)
dec_line = np.linspace(DEC_QUERY_LO - 2, DEC_QUERY_HI + 2, 100)
ra_line = np.linspace(RA_QUERY_LO - 5, RA_QUERY_HI + 5, 100)
for ra_t in RA_TICKS:
    mx_t, my_t = mollweide_xy(np.full_like(dec_line, ra_t), dec_line, RA0)
    ax.plot(mx_t, my_t, color="0.5", lw=0.6, ls=":", zorder=1)
for dec_t in DEC_TICKS:
    mx_t, my_t = mollweide_xy(ra_line, np.full_like(ra_line, dec_t), RA0)
    ax.plot(mx_t, my_t, color="0.5", lw=0.6, ls=":", zorder=1)

ax.set_aspect("equal")
ax.invert_xaxis()
x0lim, x1lim = ax.get_xlim()
y0lim, y1lim = ax.get_ylim()

# tick labels for the graticule, placed at the bottom/left frame edges
ax.set_xticks([])
ax.set_yticks([])
dec_bottom = DEC_QUERY_LO - 2 + 0.5
for ra_t in RA_TICKS:
    mx_t, my_t = mollweide_xy(ra_t, dec_bottom, RA0)
    ax.annotate(f"{ra_t:.0f}°", (mx_t, y0lim), xytext=(0, -14),
                textcoords="offset points", ha="center", va="top", fontsize=8, annotation_clip=False)
ra_left = RA_QUERY_HI + 5 - 1.0
for dec_t in DEC_TICKS:
    mx_t, my_t = mollweide_xy(ra_left, dec_t, RA0)
    ax.annotate(f"{dec_t:.0f}°", (x1lim, my_t), xytext=(-14, 0),
                textcoords="offset points", ha="right", va="center", fontsize=8, annotation_clip=False)
ax.set_xlabel("Right Ascension")
ax.set_ylabel("Declination")
ax.legend(fontsize=8, loc="lower right")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4%", pad=0.55)
cb0 = fig.colorbar(pcm, cax=cax)
cb0.set_label(r"Planck 857 GHz cirrus [MJy sr$^{-1}$]")

fig.savefig("void_wedge_demo.png", bbox_inches="tight")
print(f"wrote void_wedge_demo.png: {member.sum()} inside / {(~member).sum()} outside "
      f"(R_void={R_VOID:.1f} Mpc, ang_radius={ANG_RAD_DEG:.2f} deg)")
