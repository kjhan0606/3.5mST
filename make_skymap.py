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


# Legacy HST/Subaru deep fields (name, RA, Dec [deg], area [deg^2], facility, cap)
DEEPFIELDS = [
    ("COSMOS",    150.12,   2.21, 2.00, "multi",  "N"),
    ("GOODS-N",   189.23,  62.24, 0.04, "HST",    "N"),
    ("EGS/AEGIS", 214.80,  52.80, 0.06, "HST",    "N"),
    ("Subaru DF", 201.20,  27.40, 0.25, "Subaru", "N"),
    ("ELAIS-N1",  242.80,  54.00, 1.00, "Subaru", "N"),
    ("GOODS-S",    53.12, -27.80, 0.04, "HST",    "S"),
    ("UDS/SXDS",   34.40,  -5.20, 0.80, "multi",  "S"),
    ("XMM-LSS",    35.70,  -4.75, 3.50, "Subaru", "S"),
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
for name, ra, dec, area, fac, cap in DEEPFIELDS:
    ax.plot(wrap(ra), np.radians(dec), marker="*", ms=10, color="#f1c40f",
            mec="k", mew=0.5, zorder=6)
ax.plot([], [], marker="*", ls="", color="#f1c40f", mec="k", label="HST/Subaru deep fields")
fig.tight_layout()
fig.savefig("elg_skymap.png", bbox_inches="tight")
print("wrote elg_skymap.png")
print(f"N-cap points: {ngc.sum()}, S-cap points: {sgc.sum()}, MW-avoid points: {mw.sum()}")


# ---- zoom-in view of the legacy deep fields inside each cap ----
from matplotlib.lines import Line2D
fig3, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), dpi=150)
for axx, cap, title in [(axes[0], "N", "North Galactic cap"), (axes[1], "S", "South Galactic cap")]:
    off = {"GOODS-N": (8, -18), "UDS/SXDS": (8, -20), "XMM-LSS": (8, 8),
           "COSMOS": (-10, 10), "ELAIS-N1": (8, -18)}
    for name, ra, dec, area, fac, c in DEEPFIELDS:
        if c != cap:
            continue
        axx.scatter(ra, dec, s=90*np.sqrt(area)+40, color=FCOL[fac], alpha=0.65,
                    edgecolor="k", linewidth=0.5, zorder=3)
        axx.annotate(f"{name}\n({area:g} deg$^2$)", (ra, dec), textcoords="offset points",
                     xytext=off.get(name, (8, 7)), fontsize=7.5)
    axx.set_xlabel("RA [deg]"); axx.set_ylabel("Dec [deg]"); axx.set_title(title, pad=12)
    axx.grid(alpha=0.3); axx.invert_xaxis(); axx.margins(0.28)
handles = [Line2D([0], [0], marker="o", ls="", markerfacecolor=FCOL[k], markeredgecolor="k",
                  markersize=9, label=v) for k, v in
           [("HST", "HST (+JWST)"), ("Subaru", "Subaru/HSC"), ("multi", "HST+Subaru(+JWST)")]]
axes[0].legend(handles=handles, fontsize=7.5, loc="best")
fig3.suptitle("Legacy deep fields within the proposed ELG caps (marker size $\\propto\\sqrt{\\rm area}$)", y=1.02)
fig3.tight_layout()
fig3.savefig("elg_deepfields_zoom.png", bbox_inches="tight")
print("wrote elg_deepfields_zoom.png")
