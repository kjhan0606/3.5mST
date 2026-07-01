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
fig.tight_layout()
fig.savefig("elg_skymap.png", bbox_inches="tight")
print("wrote elg_skymap.png")
print(f"N-cap points: {ngc.sum()}, S-cap points: {sgc.sum()}, MW-avoid points: {mw.sum()}")
