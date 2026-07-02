#!/usr/bin/env python3
"""All-sky map (equatorial Mollweide) of the proposed ELG survey fields: two
high-Galactic-latitude caps that avoid the Milky Way and overlap existing
wide surveys (DESI, Euclid, Rubin/LSST, Roman) for cross-survey synergy.

The backgrounds are real sky maps rendered through the CDS hips2fits service:
the COBE/DIRBE 2.2 um near-infrared all-sky map (CADE HEALPix rendering) for
the Mollweide view, and the Planck 857 GHz map (5' resolution, thermal dust /
cirrus) for the high-resolution tangent-plane zooms. Both carry their native
intensity units (MJy/sr), shown on the colour bars.
"""
import os
import urllib.parse
import urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

HIPS2FITS = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits?"


def fetch_hips(fname, **params):
    """Download (once, cached) a FITS rendering of a HiPS map with the given
    WCS parameters through the CDS hips2fits service."""
    os.makedirs("hips_cache", exist_ok=True)
    path = os.path.join("hips_cache", fname)
    if not os.path.exists(path):
        url = HIPS2FITS + urllib.parse.urlencode({"format": "fits", **params})
        print("fetching", fname)
        urllib.request.urlretrieve(url, path)
    h = fits.open(path)[0]
    return h.data.astype(float), h.header


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
#   NDWFS/HETDEX Bootes 9.3 deg^2 (Jannuzi & Dey 1999) -> 3.5 x 2.65 deg
# (name, RA, Dec [deg], w [deg], h [deg], facility, cap)
DEEPFIELDS = [
    ("COSMOS",       150.12,   2.21, 1.400, 1.400, "multi",  "N"),
    ("GOODS-N",      189.23,  62.24, 0.273, 0.168, "HST",    "N"),
    ("EGS/AEGIS",    214.80,  52.80, 1.175, 0.168, "HST",    "N"),
    ("Subaru DF",    201.20,  27.40, 0.567, 0.450, "Subaru", "N"),
    ("ELAIS-N1",     242.80,  54.00, 1.000, 1.000, "Subaru", "N"),
    ("NDWFS Bootes", 218.00,  34.00, 3.500, 2.650, "wide",   "N"),
    ("GOODS-S",       53.12, -27.80, 0.273, 0.168, "HST",    "S"),
    ("UDS/SXDS",      34.40,  -5.20, 1.100, 1.100, "multi",  "S"),
    ("XMM-LSS",       35.70,  -4.75, 1.870, 1.870, "Subaru", "S"),
]
FCOL = {"HST": "#e67e22", "Subaru": "#2980b9", "multi": "#27ae60", "wide": "#8e44ad"}


fig = plt.figure(figsize=(9.2, 5.8), dpi=150)
ax = fig.add_subplot(111, projection="mollweide")

# real all-sky background at high resolution: Planck 857 GHz thermal dust /
# cirrus (~5' native), the dominant diffuse foreground; MJy/sr
nir, nhdr = fetch_hips("planck857_allsky_car.fits", hips="CDS/P/PLANCK/R3/HFI857",
                       width=2000, height=1000, fov=360, projection="CAR",
                       coordsys="icrs", ra=0, dec=0)
ny, nx = nir.shape
ra_e = nhdr["CRVAL1"] + nhdr["CDELT1"] * (np.arange(nx + 1) + 0.5 - nhdr["CRPIX1"])
dec_e = nhdr["CRVAL2"] + nhdr["CDELT2"] * (np.arange(ny + 1) + 0.5 - nhdr["CRPIX2"])
dec_e = np.clip(dec_e, -90.0, 90.0)
lo, hi = np.nanpercentile(nir, [2.0, 99.5])
pcm = ax.pcolormesh(np.radians(ra_e), np.radians(dec_e), nir,
                    norm=LogNorm(vmin=max(lo, 0.05), vmax=hi), cmap="gray",
                    rasterized=True, zorder=0)
cb = fig.colorbar(pcm, ax=ax, orientation="horizontal", fraction=0.05,
                  pad=0.05, aspect=45)
cb.set_label(r"Planck 857 GHz intensity [MJy sr$^{-1}$]", fontsize=8)
cb.ax.tick_params(labelsize=7)
ax.grid(alpha=0.4, color="0.75", lw=0.5)

# sky grid, split by Galactic latitude and ecliptic latitude
ra = np.linspace(-179.9, 179.9, 360)
dec = np.linspace(-89.5, 89.5, 180)
RA, DEC = np.meshgrid(ra, dec)
c = SkyCoord(RA * u.deg, DEC * u.deg, frame="icrs")
b = c.galactic.b.deg
elat = c.barycentricmeanecliptic.lat.deg

mw = np.abs(b) < 20.0                                   # Milky Way avoidance
ngc = b > 45.0     # North Galactic cap
sgc = b < -45.0    # South Galactic cap

def gal_curve(bb, **kw):
    """Curve of constant galactic latitude, split at the RA wrap."""
    lg = np.linspace(0, 360, 720)
    g = SkyCoord(lg * u.deg, np.full_like(lg, bb) * u.deg, frame="galactic").icrs
    x = wrap(g.ra.deg); y = np.radians(g.dec.deg)
    jump = np.where(np.abs(np.diff(x)) > np.pi)[0]
    x = np.insert(x, jump + 1, np.nan); y = np.insert(y, jump + 1, np.nan)
    ax.plot(x, y, **kw)

gal_curve(45.0, color="#2ecc71", lw=2.2, alpha=0.95, label="proposed N cap (b>45°)")
gal_curve(-45.0, color="#3498db", lw=2.2, alpha=0.95, label="proposed S cap (b<−45°)")

# Galactic plane and |b|=20 boundaries as guide curves
for bb, ls in [(0, "-"), (20, "--"), (-20, "--")]:
    gal_curve(bb, color="#ff8a80", lw=1.2 if bb == 0 else 0.8, ls=ls, alpha=0.75)

ax.set_title("Proposed ELG survey fields: high-latitude caps away from the Milky Way", pad=14)
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
ax.text(wrap(-110), np.radians(42), "Milky Way\n|b|<20° (avoid)", color="#ff8a80",
        fontsize=7, ha="center")
# synergy annotations placed on their caps
ax.text(wrap(158), np.radians(33), "N cap:\nDESI NGC + Euclid N\n+ Rubin edge",
        fontsize=6.5, ha="center", color="#a9dfbf")
ax.text(wrap(6), np.radians(-25), "S cap:\nDESI SGC + Euclid S\n+ Rubin/LSST + Roman",
        fontsize=6.5, ha="center", color="#aed6f1")
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
# separate tangent plane. We overlay one proposed ELG survey block at its true
# 100-300 deg^2 scale (goal is 300 deg^2 total, so ~150 deg^2 per cap), placed
# on the premier legacy-field concentration; the other legacy fields lie
# elsewhere in the same cap and anchor the deep and calibration tiers.
from matplotlib.patches import Rectangle
from astropy.wcs import WCS
import astropy.visualization.wcsaxes            # registers the WCS projection
from astropy.visualization.wcsaxes import Quadrangle

# Anchor of the nested wedding-cake tiers per cap. North is anchored on the
# high-Galactic-latitude NDWFS/HETDEX Bootes field ($b=+67^\circ$), South on the
# SXDS/UDS/XMM-LSS deep field ($b=-59^\circ$); the projection is centred on the
# anchor so the 30'x30' tiling is locally undistorted.
ANCHOR = {
    "N": dict(cen=(218.0, 34.0), where="wide survey on Bootes/HETDEX ($b\\!=\\!+67^\\circ$)"),
    "S": dict(cen=(34.7, -5.05), where="wide survey on SXDS/UDS/XMM-LSS ($b\\!=\\!-59^\\circ$)"),
}
# Real footprints of the SXDS/UDS/XMM-LSS deep field for the South zoom. SXDS is
# five Subaru Suprime-Cam pointings (34'x27' each) in a cross (Furusawa+ 2008);
# UDS is one UKIDSS/WFCAM tile (~0.77 deg^2, roughly square). These are drawn
# instead of a single rectangle. (RA, Dec, w, h) in deg.
SC = 0.567, 0.450                                 # Suprime-Cam field 34'x27'
SXDS_POINTINGS = [(34.53, -5.02), (34.53, -4.52), (34.53, -5.52),
                  (35.05, -5.02), (34.01, -5.02)]     # C, N, S, E, W cross
UDS_TILE = (34.40, -5.10, 0.877, 0.877)           # UKIDSS/WFCAM ~0.77 deg^2
FOV = 0.5                                         # 30 arcmin = 0.5 deg square survey tile
# wedding-cake tiers (area deg^2, tile fill colour, label). Each tier is a SQUARE
# region paved with an integer number of 30'x30' tiles, so its boundary is a
# clean rectangle (easier for the statistical/boundary treatment) rather than a
# circle. L_* is the half-extent of the tile CENTRES [deg]; with a tile centred
# on the anchor the tier is (2 L/FOV + 1) tiles per side.
TIERS = [
    (156.0, "#f7dc6f", "wide tier ($\\sim$150 deg$^2$)"),     # 25x25 tiles, 12.5 deg side
    (30.0,  "#e59866", "medium tier (30 deg$^2$)"),           # 11x11 tiles, 5.5 deg side
    (2.2,   "#cd6155", "deep tier (1--3 deg$^2$)"),           # 3x3 tiles, 1.5 deg side
]
L_WIDE, L_MED, L_DEEP = 6.0, 2.5, 0.5             # square half-extent of tile centres [deg]


def cap_wcs(ra0, dec0):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra0, dec0]
    w.wcs.crpix = [1.0, 1.0]
    w.wcs.cdelt = [-1.0, 1.0]                    # 1 deg per pixel (pixel = tangent-plane deg)
    return w


def tile_tier(cx, cy):
    """Tier index (0 wide, 1 medium, 2 deep) for a tile centred at (cx, cy) [deg]
    within nested square regions, else None."""
    m = max(abs(cx), abs(cy))
    if m <= L_DEEP + 1e-9:
        return 2
    if m <= L_MED + 1e-9:
        return 1
    if m <= L_WIDE + 1e-9:
        return 0
    return None


# high-resolution background for the zooms: Planck 857 GHz (thermal dust /
# cirrus, ~5' resolution), rendered on the same TAN tangent plane per cap.
PLANCK = {}
for cap in ("N", "S"):
    ra0, dec0 = ANCHOR[cap]["cen"]
    PLANCK[cap] = dict(zip(("main", "inset"), (
        fetch_hips(f"planck857_{cap}_main.fits", hips="CDS/P/PLANCK/R3/HFI857",
                   width=640, height=640, fov=64, projection="TAN",
                   coordsys="icrs", ra=ra0, dec=dec0),
        fetch_hips(f"planck857_{cap}_inset.fits", hips="CDS/P/PLANCK/R3/HFI857",
                   width=400, height=400, fov=8.2, projection="TAN",
                   coordsys="icrs", ra=ra0, dec=dec0))))
_pl_all = np.concatenate([PLANCK[c]["main"][0].ravel() for c in ("N", "S")])
PL_NORM = LogNorm(vmin=np.nanpercentile(_pl_all, 2.0),
                  vmax=np.nanpercentile(_pl_all, 99.8))


def show_planck(axes, data, hdr, zorder=0):
    """Draw a hips2fits TAN cutout in the tangent-plane (deg) frame of the
    plotting WCS (same CRVAL, cdelt = 1 deg/pix), preserving orientation."""
    n = data.shape[1]
    hw = abs(hdr["CDELT1"]) * n / 2.0                # tangent-plane half-width
    return axes.imshow(data, origin="lower", extent=(-hw, hw, -hw, hw),
                       cmap="gray", norm=PL_NORM, zorder=zorder,
                       interpolation="bilinear")


fig3 = plt.figure(figsize=(11.6, 5.9), dpi=150)
main_axes = []
for i, (cap, ttl) in enumerate([("N", "North Galactic cap"), ("S", "South Galactic cap")]):
    a = ANCHOR[cap]
    ra0, dec0 = a["cen"]
    w = cap_wcs(ra0, dec0)                         # projection centred on the anchor
    axx = fig3.add_subplot(1, 2, i + 1, projection=w)
    main_axes.append(axx)
    tw = axx.get_transform("world")
    im_pl = show_planck(axx, *PLANCK[cap]["main"])
    # pave the nested tiers with 30'x30' tiles on a 0.5 deg tangent-plane grid
    # (pixel = deg here); tiles whose centre is inside a tier radius are kept, so
    # each tier boundary comes out jagged rather than a clean rectangle.
    ncell = int(np.ceil(L_WIDE / FOV)) + 1
    grid = np.arange(-ncell, ncell + 1) * FOV
    for cx in grid:
        for cy in grid:
            k = tile_tier(cx, cy)
            if k is None:
                continue
            axx.add_patch(Rectangle((cx - FOV / 2, cy - FOV / 2), FOV, FOV,
                                     facecolor=TIERS[k][1], edgecolor="white",
                                     lw=0.15, alpha=0.45, zorder=1 + k))
    # neighbouring legacy deep fields in view (the anchor field is shown in the
    # inset instead, so it is drawn here without a label to avoid clutter).
    off = {"EGS/AEGIS": (-58, 2), "Subaru DF": (10, -15),
           "ELAIS-N1": (-12, 9), "GOODS-S": (10, -8)}
    half = 32.0                                   # wide window so the TAN sky curvature shows
    zoom_skip = {"UDS/SXDS", "XMM-LSS"}           # shown as their real footprints in the inset
    for name, ra, dec, ww, hh, fac, c in DEEPFIELDS:
        if c != cap or name in zoom_skip:
            continue
        px, py = w.world_to_pixel_values(ra, dec)
        if abs(px) > half or abs(py) > half:      # outside this zoom
            continue
        w_ra = ww / np.cos(np.radians(dec))                # true RA extent [deg]
        # the NDWFS field is drawn as an outline so the 30' survey tiling inside
        # it stays visible; the compact legacy fields are filled.
        outline = fac == "wide"
        axx.add_patch(Quadrangle(((ra - w_ra / 2) * u.deg, (dec - hh / 2) * u.deg),
                                 w_ra * u.deg, hh * u.deg, transform=tw,
                                 facecolor="none" if outline else FCOL[fac],
                                 alpha=0.95, edgecolor=FCOL[fac] if outline else "k",
                                 lw=1.8 if outline else 0.9, zorder=6))
        if name != "NDWFS Bootes":                # anchor field is labelled in the inset
            axx.annotate(f"{name}\n({ww * hh:.2g} deg$^2$)", (float(px), float(py)),
                         textcoords="offset points", xytext=off.get(name, (9, 8)),
                         fontsize=7.4, color="white", zorder=7)
    axx.set_xlim(-half, half); axx.set_ylim(-half, half)
    axx.coords.grid(color="0.8", alpha=0.55, ls=":")
    axx.coords[0].set_axislabel("R.A. (TAN)"); axx.coords[1].set_axislabel("Dec. (TAN)")
    axx.coords[0].set_major_formatter("d"); axx.coords[1].set_major_formatter("d")
    axx.coords[0].set_ticks(spacing=10 * u.deg); axx.coords[1].set_ticks(spacing=10 * u.deg)
    for cc, pos in ((0, "b"), (1, "l")):                   # RA on bottom, Dec on left
        axx.coords[cc].set_ticks_position(pos)
        axx.coords[cc].set_ticklabel_position(pos)
        axx.coords[cc].set_axislabel_position(pos)
    # inset: reveal the 30' tiling (jagged) and the true footprint shapes, which
    # are invisible at the wide scale needed to show the sky curvature.
    ih = 4.0
    axins = axx.inset_axes([0.61, 0.57, 0.37, 0.40])
    show_planck(axins, *PLANCK[cap]["inset"])
    for cx in grid:
        for cy in grid:
            if abs(cx) > ih or abs(cy) > ih:
                continue
            k = tile_tier(cx, cy)
            if k is not None:
                axins.add_patch(Rectangle((cx - FOV / 2, cy - FOV / 2), FOV, FOV,
                                          facecolor=TIERS[k][1], edgecolor="white",
                                          lw=0.3, alpha=0.45))
    if cap == "N":
        bpx, bpy = w.world_to_pixel_values(218.0, 34.0)
        # outline only, so the 30' tiles paving the NDWFS region remain visible
        axins.add_patch(Rectangle((float(bpx) - 3.5 / 2, float(bpy) - 2.65 / 2), 3.5, 2.65,
                                  facecolor="none", edgecolor=FCOL["wide"], lw=2.0))
        axins.text(0, 3.0, "NDWFS Bootes (9.3 deg$^2$)", ha="center", fontsize=6.4, color="white")
    else:
        for pra, pdec in SXDS_POINTINGS:
            spx, spy = w.world_to_pixel_values(pra, pdec)
            axins.add_patch(Rectangle((float(spx) - SC[0] / 2, float(spy) - SC[1] / 2),
                                      SC[0], SC[1], facecolor=FCOL["Subaru"], alpha=0.85,
                                      edgecolor="k", lw=0.6))
        upx, upy = w.world_to_pixel_values(UDS_TILE[0], UDS_TILE[1])
        axins.add_patch(Rectangle((float(upx) - 0.877 / 2, float(upy) - 0.877 / 2), 0.877, 0.877,
                                  facecolor="none", edgecolor="#58d68d", lw=1.4, ls="--"))
        axins.text(0, 3.0, "SXDS cross + UDS tile", ha="center", fontsize=6.4, color="white")
    axins.set_xlim(-ih, ih); axins.set_ylim(-ih, ih)       # RA increases to the left
    axins.set_xticks([]); axins.set_yticks([])
    for s in axins.spines.values():
        s.set_edgecolor("0.3")
    axins.set_title(f"deep-tier zoom (±{ih:.0f}°, 30$'$ tiles)", fontsize=6.6, pad=2, color="white")
    axx.indicate_inset_zoom(axins, edgecolor="0.85", alpha=0.7, lw=0.8)
    axx.set_title(f"{ttl}\n{a['where']}", fontsize=9.5, pad=8)

handles = [Rectangle((0, 0), 1, 1, facecolor=fc, edgecolor="0.4", lw=0.4, alpha=0.9, label=lab)
           for (area, fc, lab) in TIERS]
handles += [Rectangle((0, 0), 1, 1, facecolor=FCOL[k], edgecolor="k", alpha=0.95, label=v)
            for k, v in [("HST", "HST (+JWST)"), ("Subaru", "Subaru/HSC"),
                         ("multi", "HST+Subaru(+JWST)")]]
handles += [Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=FCOL["wide"], lw=2.0,
                      label="NDWFS/HETDEX (outline)"),
            Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="#58d68d", lw=1.6, ls="--",
                      label="UDS (UKIDSS/WFCAM tile)")]
fig3.legend(handles=handles, fontsize=7.4, loc="lower center", ncol=4,
            bbox_to_anchor=(0.5, -0.12))
fig3.suptitle("Proposed ELG survey tiers (nested square regions paved with "
              "30$'\\!\\times\\!$30$'$ tiles) and the legacy deep fields they cover,\n"
              "on the Planck 857 GHz thermal-dust (cirrus) map", y=1.00)
fig3.tight_layout(rect=(0, 0.11, 1, 0.97))
cb3 = fig3.colorbar(im_pl, ax=main_axes, fraction=0.03, pad=0.015)
cb3.set_label(r"Planck 857 GHz intensity [MJy sr$^{-1}$]", fontsize=8)
cb3.ax.tick_params(labelsize=7)
fig3.savefig("elg_deepfields_zoom.png", bbox_inches="tight")
print("wrote elg_deepfields_zoom.png")
