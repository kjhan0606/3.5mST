"""Observed ELG absolute-magnitude proxy versus redshift from the DESI DR1
LSS ELG catalog (Figure fig:elgmagdepth).

Points are the secure DESI DR1 ELG_LOPnotqso sample (ZWARN=0, GALAXY,
[O II] S/N>5, m_r<=24.3) written to elg_absmag_depth_catalog_sample.dat,
with M_r - 5log10(h) computed from FlatLambdaCDM(H0=70, Om0=0.30) and no
K-correction. Depth curves translate observed r-band limits into the
faintest reachable proxy magnitude: the effective eBOSS limit r=22.44, the
DESI DR1 99th-percentile depth of the plotted sample r=24.08, and the
proposed wide/medium/deep continuum limits 24.3/25.1/25.8.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

DATA = Path("elg_absmag_depth_catalog_sample.dat")
OUT = Path("elg_absmag_depth_catalog.png")

COSMO = FlatLambdaCDM(H0=70.0, Om0=0.30)
HLITTLE = 0.70

# (label, observed r limit, colour, linestyle, linewidth)
LIMITS = [
    (r"eBOSS effective: $r\simeq22.44$", 22.44, "black", "-.", 2.6),
    (r"DESI DR1 99\%: $r\simeq24.08$".replace("\\%", "%"), 24.08, "#8b46e0", (0, (7, 3)), 2.2),
    (r"proposed wide 0.75 hr: $m_{AB}=24.3$", 24.3, "#1f7a4d", "-", 2.2),
    (r"proposed medium 3 hr: $m_{AB}=25.1$", 25.1, "#2a6cb5", (0, (6, 3)), 2.2),
    (r"proposed deep 12 hr: $m_{AB}=25.8$", 25.8, "#d68f1f", ":", 2.4),
]


def mu_proxy(z):
    dl = COSMO.luminosity_distance(z).to("Mpc").value
    return 5.0 * np.log10(dl * HLITTLE * 1.0e5)


z, absM, rmag = np.loadtxt(DATA).T
print(f"{z.size} catalog points, z = {z.min():.2f}..{z.max():.2f}, "
      f"r = {rmag.min():.2f}..{rmag.max():.2f}")

fig, ax = plt.subplots(figsize=(9.6, 5.6), dpi=200)

ax.axvspan(0.6, 1.6, color="#7f5fd0", alpha=0.07, zorder=0)
sc = ax.scatter(z, absM, c=rmag, s=3, cmap="viridis_r", alpha=0.35,
                linewidths=0, zorder=2, rasterized=True)

zz = np.linspace(0.55, 3.05, 400)
mu = mu_proxy(zz)
for label, mlim, col, ls, lw in LIMITS:
    ax.plot(zz, mlim - mu, color=col, ls=ls, lw=lw, zorder=4, label=label)

ax.set_xlim(0.55, 3.05)
ax.set_ylim(-18.4, -25.35)
ax.set_xlabel(r"redshift $z$", fontsize=13)
ax.set_ylabel(r"$M_r - 5\log_{10}h$  (mag)", fontsize=13)
ax.grid(color="w", alpha=0.9, lw=0.7, zorder=1)

cb = fig.colorbar(sc, ax=ax, pad=0.015)
cb.set_label(r"observed $r$ magnitude", fontsize=12)
cb.solids.set_alpha(0.55)

ax.text(0.6 + 0.5, -24.45,
        "DESI ELG main span\n$0.6<z<1.6$",
        color="#6a4fc0", fontsize=10.5, ha="center", va="top", zorder=5)
ax.text(0.015, 0.975,
        "DESI DR1 LSS ELG catalog, secure [O II] detections\n"
        "Objects fainter than the proposed wide-tier\n"
        "limit are not plotted.",
        transform=ax.transAxes, fontsize=10, va="top", zorder=5,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.8",
                  boxstyle="round,pad=0.35"))
ax.text(0.985, 0.12,
        "The DESI ELG locus reaches nearly the proposed\n"
        "wide-tier continuum depth; medium/deep tiers extend\n"
        "selection-function calibration below the DESI envelope.",
        transform=ax.transAxes, fontsize=9.5, color="0.35",
        ha="right", va="top", zorder=5)
ax.legend(fontsize=8.6, loc="upper right", framealpha=0.92)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}")
