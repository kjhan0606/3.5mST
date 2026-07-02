#!/usr/bin/env python3
"""All-sky map (equatorial Mollweide) of the proposed ELG survey fields: two
high-Galactic-latitude caps that avoid the Milky Way and overlap existing
wide surveys (DESI, Euclid, Rubin/LSST, Roman) for cross-survey synergy."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u


def wrap(ra_deg):
    r = np.remainder(ra_deg + 180.0, 360.0) - 180.0
    return np.radians(r)


# Legacy HST/Subaru deep fields. Footprint width x height [deg] are the real
# survey dimensions (angular size on the sky), not derived from the area:
#   COSMOS     1.4x1.4 deg  (Scoville+ 2007)
#   GOODS-N/-S 16.4'x10.1'  (Giavalisco+ 2004)  -> 0.273 x 0.168 deg
#   EGS/AEGIS  70.5'x10.1'  (Davis+ 2007, strip) -> 1.175 x 0.168 deg
#   Subaru DF  34'x27'      (Kashikawa+ 2004)    -> 0.567 x 0.450 deg
#   ELAIS-N1   HSC-Deep ~1 deg^2                 -> 1.0 x 1.0 deg
#   UDS/SXDS   SXDS 1.22 deg^2 (Furusawa+ 2008)  -> 1.10 x 1.10 deg
#   XMM-LSS    ~3.5 deg^2 deep tier              -> 1.87 x 1.87 deg
# (name, RA, Dec [deg], w [deg], h [deg], facility, cap)
DEEPFIELDS = [
    ("COSMOS",    150.12,   2.21, 1.400, 1.400, "multi",  "N"),
    ("GOODS-N",   189.23,  62.24, 0.273, 0.168, "HST",    "N"),
    ("EGS/AEGIS", 214.80,  52.80, 1.175, 0.168, "HST",    "N"),
    ("Subaru DF", 201.20,  27.40, 0.567, 0.450, "Subaru", "N"),
    ("ELAIS-N1",  242.80,  54.00, 1.000, 1.000, "Subaru", "N"),
    ("GOODS-S",    53.12, -27.80, 0.273, 0.168, "HST",    "S"),
    ("UDS/SXDS",   34.40,  -5.20, 1.100, 1.100, "multi",  "S"),
    ("XMM-LSS",    35.70,  -4.75, 1.870, 1.870, "Subaru", "S"),
]
FCOL = {"HST": "#e67e22", "Subaru": "#2980b9", "multi": "#27ae60"}


fig = plt.figure(figsize=(9.2, 5.4), dpi=150)
ax = fig.add_subplot(111, projection="mollweide")
ax.grid(alpha=0.3)

# sky grid, split by Galactic latitude and ecliptic latitude
ra = np.linspace(-179.9, 179.9, 360)
dec = np.linspace(-89.5, 89.5, 180)
RA, DEC = np.meshgrid(ra, dec)
c = SkyCoord(RA * u.deg, DEC * u.deg, frame="icrs")
b = c.galactic.b.deg
elat = c.barycentricmeanecliptic.lat.deg

mw = np.abs(b) < 20.0                                   # Milky Way avoidance
ax.scatter(wrap(RA[mw]), np.radians(DEC[mw]), s=2, color="#c0392b", alpha=0.10)

# proposed caps: high |b|, split into North and South Galactic caps
ngc = b > 45.0     # North Galactic cap
sgc = b < -45.0    # South Galactic cap
for m, lab, col in [(ngc, "proposed N cap", "#2ecc71"), (sgc, "proposed S cap", "#3498db")]:
    ax.scatter(wrap(RA[m]), np.radians(DEC[m]), s=3, color=col, alpha=0.35, label=lab)

# Galactic plane and |b|=20 boundaries as guide curves
lgrid = np.linspace(0, 360, 400)
for bb, ls in [(0, "-"), (20, "--"), (-20, "--")]:
    g = SkyCoord(lgrid * u.deg, np.full_like(lgrid, bb) * u.deg, frame="galactic").icrs
    order = np.argsort(g.ra.wrap_at(180 * u.deg).deg)
    ax.plot(wrap(g.ra.deg[order]), np.radians(g.dec.deg[order]), color="#7f2d1a",
            lw=1.2 if bb == 0 else 0.8, ls=ls, alpha=0.7)

ax.set_title("Proposed ELG survey fields: high-latitude caps away from the Milky Way", pad=14)
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
ax.text(wrap(-110), np.radians(42), "Milky Way\n|b|<20° (avoid)", color="#7f2d1a",
        fontsize=7, ha="center")
# synergy annotations placed on their caps
ax.text(wrap(158), np.radians(33), "N cap:\nDESI NGC + Euclid N\n+ Rubin edge",
        fontsize=6.5, ha="center", color="#145a32")
ax.text(wrap(6), np.radians(-25), "S cap:\nDESI SGC + Euclid S\n+ Rubin/LSST + Roman",
        fontsize=6.5, ha="center", color="#154360")
# legacy deep fields as gold stars within the caps
for name, ra, dec, w, h, fac, cap in DEEPFIELDS:
    ax.plot(wrap(ra), np.radians(dec), marker="*", ms=10, color="#f1c40f",
            mec="k", mew=0.5, zorder=6)
ax.plot([], [], marker="*", ls="", color="#f1c40f", mec="k", label="HST/Subaru deep fields")
fig.tight_layout()
fig.savefig("elg_skymap.png", bbox_inches="tight")
print("wrote elg_skymap.png")
print(f"N-cap points: {ngc.sum()}, S-cap points: {sgc.sum()}, MW-avoid points: {mw.sum()}")


# ---- zoom-in view in a gnomonic (TAN) tangent-plane projection ----
# TAN is the standard projection for imaging fields in astronomy; astropy's
# WCSAxes draws the curved RA/Dec graticule automatically. Each cap is a
# separate tangent plane. We overlay the proposed ELG survey rectangle (a
# high-|b|, Milky-Way-avoiding region enclosing the legacy fields) and the
# legacy deep fields at their true footprint size.
from matplotlib.patches import Rectangle
from astropy.wcs import WCS
import astropy.visualization.wcsaxes            # registers the WCS projection
from astropy.visualization.wcsaxes import Quadrangle

# Proposed ELG survey rectangles (RA range, Dec range, tangent point). Both stay
# at |b|>27 deg (Milky Way clear) and enclose that cap's legacy deep fields.
ELG_BOXES = {
    "N": dict(ra=(140.0, 250.0), dec=(-2.0, 64.0), cen=(196.0, 34.0)),
    "S": dict(ra=(26.0, 62.0),  dec=(-32.0, 2.0),  cen=(44.0, -15.0)),
}


def cap_wcs(ra0, dec0):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra0, dec0]
    w.wcs.crpix = [1.0, 1.0]
    w.wcs.cdelt = [-1.0, 1.0]                    # 1 deg per pixel
    return w


fig3 = plt.figure(figsize=(11.4, 5.2), dpi=150)
for i, (cap, title) in enumerate([("N", "North Galactic cap"), ("S", "South Galactic cap")]):
    box = ELG_BOXES[cap]
    w = cap_wcs(*box["cen"])
    axx = fig3.add_subplot(1, 2, i + 1, projection=w)
    tw = axx.get_transform("world")
    # proposed ELG survey region
    (r0, r1), (d0, d1) = box["ra"], box["dec"]
    axx.add_patch(Quadrangle((r0 * u.deg, d0 * u.deg), (r1 - r0) * u.deg, (d1 - d0) * u.deg,
                             transform=tw, facecolor="#f4d03f", alpha=0.15,
                             edgecolor="#b7950b", lw=1.6, ls="--", zorder=1))
    # legacy deep fields at true footprint size
    off = {"GOODS-N": (10, 9), "EGS/AEGIS": (-34, -20), "COSMOS": (10, 8),
           "ELAIS-N1": (11, 5), "Subaru DF": (9, -16),
           "UDS/SXDS": (11, -16), "XMM-LSS": (10, 11), "GOODS-S": (6, -8)}
    for name, ra, dec, ww, hh, fac, c in DEEPFIELDS:
        if c != cap:
            continue
        w_ra = ww / np.cos(np.radians(dec))                # true RA extent [deg]
        axx.add_patch(Quadrangle(((ra - w_ra / 2) * u.deg, (dec - hh / 2) * u.deg),
                                 w_ra * u.deg, hh * u.deg, transform=tw,
                                 facecolor=FCOL[fac], alpha=0.75, edgecolor="k",
                                 lw=0.8, zorder=4))
        axx.annotate(f"{name}\n({ww * hh:.2g} deg$^2$)",
                     w.world_to_pixel_values(ra, dec), textcoords="offset points",
                     xytext=off.get(name, (9, 8)), fontsize=7.2, zorder=5)
    # frame the plot on the ELG box with a little margin
    cr = [r0, r1, r1, r0]; cd = [d0, d0, d1, d1]
    px, py = w.world_to_pixel_values(cr, cd)
    m = 6.0
    axx.set_xlim(px.min() - m, px.max() + m); axx.set_ylim(py.min() - m, py.max() + m)
    axx.coords.grid(color="0.6", alpha=0.5, ls=":")
    axx.coords[0].set_axislabel("R.A."); axx.coords[1].set_axislabel("Dec.")
    axx.coords[0].set_major_formatter("d"); axx.coords[1].set_major_formatter("d")
    axx.coords[0].set_ticks(spacing=20 * u.deg); axx.coords[1].set_ticks(spacing=20 * u.deg)
    for c, pos in ((0, "b"), (1, "l")):                    # RA on bottom, Dec on left
        axx.coords[c].set_ticks_position(pos)
        axx.coords[c].set_ticklabel_position(pos)
        axx.coords[c].set_axislabel_position(pos)
    axx.set_title(title, pad=10)

handles = [Rectangle((0, 0), 1, 1, facecolor="#f4d03f", edgecolor="#b7950b", ls="--",
                     alpha=0.4, label="proposed ELG survey region")]
handles += [Rectangle((0, 0), 1, 1, facecolor=FCOL[k], edgecolor="k", alpha=0.75, label=v)
            for k, v in [("HST", "HST (+JWST)"), ("Subaru", "Subaru/HSC"),
                         ("multi", "HST+Subaru(+JWST)")]]
fig3.legend(handles=handles, fontsize=7.8, loc="lower center", ncol=4,
            bbox_to_anchor=(0.5, -0.02))
fig3.suptitle("Proposed ELG survey regions and legacy deep fields "
              "(gnomonic TAN projection, true footprint sizes)", y=0.99)
fig3.tight_layout(rect=(0, 0.04, 1, 1))
fig3.savefig("elg_deepfields_zoom.png", bbox_inches="tight")
print("wrote elg_deepfields_zoom.png")
