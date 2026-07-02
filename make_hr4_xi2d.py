#!/usr/bin/env python3
"""Two-dimensional galaxy two-point correlation function of the Horizon Run 4
mock galaxy (PSB) catalog, drawn directly from the author's pair-count
measurement (the maps behind Kim et al. 2015 are re-rendered here from the
raw xi(sigma,pi) grids, not reproduced from any published figure).

Input grids: quadrant-mirrored 1024x1024 xi(sigma,pi) maps on a 0.2539
h^-1 Mpc grid (extent +-130 h^-1 Mpc), in real space (no peculiar
velocities) and in redshift space (peculiar velocities applied along pi).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.io import fits

BASE = "/scratch/kjhan/Nbody/HR4/FoFPSB/L0.2/Correlation"
DX = 0.2539                       # h^-1 Mpc per bin
BAO = 107.0                       # h^-1 Mpc

panels = [("hr4psb.nopv.mir.dat.fits", "real space"),
          ("hr4psb.pv.mir.dat.fits", "redshift space")]

fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6), dpi=150,
                         constrained_layout=True)
for ax, (fname, title) in zip(axes, panels):
    xi = fits.open(f"{BASE}/{fname}")[0].data.astype(float)
    n = xi.shape[0]
    half = DX * n / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        lg = np.log10(np.where(xi > 0, xi, np.nan))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("black")
    im = ax.imshow(lg, origin="lower", extent=(-half, half, -half, half),
                   cmap=cmap, vmin=-4.0, vmax=1.0, interpolation="nearest",
                   rasterized=True)
    ax.add_patch(Circle((0, 0), BAO, fill=False, color="#4ec9b0", lw=1.6,
                        ls="--"))
    ax.text(0, BAO + 6, f"BAO ridge $s\\simeq{BAO:.0f}\\,h^{{-1}}$ Mpc",
            color="#4ec9b0", fontsize=9, ha="center")
    ax.set_xlim(-half, half); ax.set_ylim(-half, half)
    ax.set_xlabel(r"$\sigma$ [$h^{-1}$ Mpc]")
    ax.set_ylabel(r"$\pi$ [$h^{-1}$ Mpc]")
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
cb.set_label(r"$\log_{10}\,\xi(\sigma,\pi)$")
fig.suptitle("Horizon Run 4 mock-galaxy two-dimensional correlation function",
             fontsize=12)
fig.savefig("gal_corr_hr4.png", bbox_inches="tight")
print("wrote gal_corr_hr4.png")
