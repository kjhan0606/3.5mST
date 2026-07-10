#!/usr/bin/env python3
"""Two-dimensional galaxy two-point correlation function: observed BOSS
CMASS-North data (left) against the Horizon Run 4 mock galaxy (PSB) catalog
in redshift space (right), on the same log10|xi| color scale and the same
+-130 h^-1 Mpc display radius.

Left panel: obscorr pair-count measurement of BOSS CMASS-North (Nd=568776,
nbin=256, RMAX=180 h^-1 Mpc, linear binning), from the source data behind
/home/kjhan/BACKUP/YYCHOI/sdss2pcf/src/plot_bao_jk.py.
Right panel: quadrant-mirrored 1024x1024 xi(sigma,pi) map on a 0.2539
h^-1 Mpc grid (extent +-130 h^-1 Mpc), in redshift space (peculiar
velocities applied along pi); the maps behind Kim et al. 2015 are
re-rendered here from the raw xi(sigma,pi) grid, not reproduced from any
published figure.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.io import fits

BASE = "/scratch/kjhan/Nbody/HR4/FoFPSB/L0.2/Correlation"
DX = 0.2539                       # h^-1 Mpc per bin (HR4 mock grid)
BAO_HR4 = 107.0                    # h^-1 Mpc, HR4 mock BAO ridge

BOSS_FILE = "/home/kjhan/BACKUP/YYCHOI/sdss2pcf/boss/obscorr_cmass_bao_n256.out"
BOSS_RMAX = 180.0                  # h^-1 Mpc, obscorr nbin=256 RMAX=180 (bao_n256.log)
BAO_BOSS = 105.0                   # h^-1 Mpc, observed BAO ridge (plot_bao_jk.py)


def read_boss_xi2d(fn):
    with open(fn, "rb") as f:
        nx, ny = np.fromfile(f, np.int32, 2)
        return np.fromfile(f, np.float32, nx * ny).reshape(ny, nx)


def log_xi(xi):
    # scaling follows the original kjhan.pro: log10|xi| with non-positive
    # values floored (2e-9), displayed over [-4, 0]
    return np.log10(np.abs(np.where(np.isfinite(xi) & (xi > 0), xi, 2e-9)))


cmap = plt.get_cmap("jet").copy()               # cgLoadCT 33 equivalent
cmap.set_bad(cmap(0.0))

xi_hr4 = fits.open(f"{BASE}/hr4psb.pv.mir.dat.fits")[0].data.astype(float)
n_hr4 = xi_hr4.shape[0]
half = DX * n_hr4 / 2.0            # +-130 h^-1 Mpc, sets the shared display radius

xi_boss = read_boss_xi2d(BOSS_FILE)
half_boss = BOSS_RMAX               # -RMAX..+RMAX, 2*nbin+1 mirrored grid

panels = [(xi_boss, half_boss, BAO_BOSS, "BOSS CMASS-North (observed, redshift space)"),
          (xi_hr4, half, BAO_HR4, "HR4 mock (redshift space)")]

fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6), dpi=150,
                         constrained_layout=True)
for ax, (xi, h, bao, title) in zip(axes, panels):
    im = ax.imshow(log_xi(xi), origin="lower", extent=(-h, h, -h, h),
                   cmap=cmap, vmin=-4.0, vmax=0.0, interpolation="nearest",
                   rasterized=True)
    ax.add_patch(Circle((0, 0), bao, fill=False, color="red", lw=1.6))
    ax.text(0, -bao - 16, f"BAO ridge $s\\simeq{bao:.0f}\\,h^{{-1}}$ Mpc",
            color="white", fontsize=9, ha="center")
    ax.set_xlim(-half, half); ax.set_ylim(-half, half)
    ax.set_xlabel(r"$\sigma$ [$h^{-1}$ Mpc]")
    ax.set_ylabel(r"$\pi$ [$h^{-1}$ Mpc]")
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
cb.set_label(r"$\log_{10}\,\xi(\sigma,\pi)$")
fig.savefig("gal_corr_hr4.png", bbox_inches="tight")
print("wrote gal_corr_hr4.png")
