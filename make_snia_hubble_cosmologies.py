"""SN Ia Hubble diagram against four cosmologies, with a residual panel.

Uses the real Pantheon+SH0ES sample (Brout et al. 2022; Scolnic et al. 2022;
Riess et al. 2022), 1701 spectroscopically confirmed SNe Ia. The Cepheid
calibrators are removed and the Hubble-flow cosmology sample (zHD > 0.01) is
plotted. Because MU_SH0ES is tied to the SH0ES distance ladder, the reference
cosmology uses the Pantheon+ flat-LCDM best fit (H0=73.04, Om0=0.334).

The lower-panel legend lists only the two scatter bands that are unique to it
(current and target per-SN standardization scatter); the cosmology-curve
entries duplicated the upper-panel legend and are omitted.

Data file: pantheonplus_SH0ES.dat
  https://github.com/PantheonPlusSH0ES/DataRelease (Pantheon+_Data/4_DISTANCES_AND_COVAR)
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.cosmology import FlatLambdaCDM, LambdaCDM, Flatw0waCDM

OUT = Path("snia_hubble_cosmologies.png")
DATA = Path("pantheonplus_SH0ES.dat")
H0 = 73.04          # Pantheon+SH0ES (Riess et al. 2022)
OM0 = 0.334         # Pantheon+ flat-LCDM best fit (Brout et al. 2022)

LCDM = FlatLambdaCDM(H0=H0, Om0=OM0)                        # reference
OCDM = LambdaCDM(H0=H0, Om0=OM0, Ode0=0.0)                  # open, no Lambda
EDS = FlatLambdaCDM(H0=H0, Om0=1.0)                         # Einstein-de Sitter
DDE = Flatw0waCDM(H0=H0, Om0=OM0, w0=-0.857, wa=-0.153)     # CPL dynamic DE

C_LCDM, C_OCDM, C_EDS, C_DDE = "#21507e", "#d0392b", "#2f9e44", "#7048b6"

# ---- real Pantheon+SH0ES sample ----
names = open(DATA).readline().split()
raw = np.genfromtxt(DATA, skip_header=1, usecols=[
    names.index("zHD"), names.index("MU_SH0ES"),
    names.index("MU_SH0ES_ERR_DIAG"), names.index("IS_CALIBRATOR")])
zall, muall, errall, iscal = raw.T
keep = (iscal < 0.5) & (zall > 0.01)        # drop Cepheid calibrators, keep Hubble flow
zsn, mu_obs, err = zall[keep], muall[keep], errall[keep]
print(f"Pantheon+ Hubble-flow SNe: {zsn.size}, z = {zsn.min():.3f}..{zsn.max():.3f}")

zz = np.linspace(max(0.01, zsn.min()), zsn.max(), 400)
mu_lcdm = LCDM.distmod(zz).value
mu_ocdm = OCDM.distmod(zz).value
mu_eds = EDS.distmod(zz).value
mu_dde = DDE.distmod(zz).value
print(f"median residual vs reference LCDM: {np.median(mu_obs - LCDM.distmod(zsn).value):+.3f} mag")

fig = plt.figure(figsize=(8.1, 6.35), dpi=192)
gs = GridSpec(2, 1, height_ratios=[3.0, 1.6], hspace=0.06)
ax = fig.add_subplot(gs[0])
axr = fig.add_subplot(gs[1], sharex=ax)

# ---- upper panel ----
C_SN = "#e8902a"   # warm amber: high-contrast against the four cosmology curves
ax.errorbar(zsn, mu_obs, yerr=err, fmt="o", ms=3.0, mfc=C_SN, mec="#9c5a10",
            mew=0.25, alpha=0.70, ecolor="#d9a066", elinewidth=0.4, lw=0, zorder=2,
            label="SN Ia (Pantheon+; Brout et al. 2022)")
ax.plot(zz, mu_lcdm, "-", color=C_LCDM, lw=2.2, zorder=6,
        label=r"$\Lambda$CDM ($\Omega_m=0.33,\ \Omega_\Lambda=0.67$)")
ax.plot(zz, mu_ocdm, "--", color=C_OCDM, lw=1.8, zorder=6,
        label=r"Open CDM ($\Omega_m=0.33,\ \Omega_\Lambda=0$)")
ax.plot(zz, mu_eds, "-.", color=C_EDS, lw=1.8, zorder=6,
        label=r"Einstein--de Sitter ($\Omega_m=1,\ \Omega_\Lambda=0$)")
ax.plot(zz, mu_dde, ":", color=C_DDE, lw=2.4, zorder=6,
        label=r"Dynamic DE ($w_0=-0.857,\ w_a=-0.153$)")
ax.set_ylabel(r"Distance Modulus $\mu$ (mag)", fontsize=11)
ax.set_title("Type Ia Supernova Hubble Diagram", fontsize=13, weight="bold")
ax.set_ylim(32, 47)
ax.legend(fontsize=8.2, loc="lower right", framealpha=0.92)
ax.tick_params(labelbottom=False)
ax.grid(alpha=0.12)

# ---- lower panel: residuals vs LCDM ----
res_o = mu_ocdm - mu_lcdm
res_e = mu_eds - mu_lcdm
res_d = mu_dde - mu_lcdm
res_sn = mu_obs - LCDM.distmod(zsn).value

# scatter bands (these are the ONLY lower-panel legend entries)
axr.axhspan(-0.12, 0.12, color="#b7e4c7", alpha=0.55, zorder=0,
            label=r"Current scatter ($\sigma\approx0.12$ mag)")
axr.axhspan(-0.08, 0.08, color="#74c476", alpha=0.55, zorder=0,
            label=r"Target scatter ($\sigma\approx0.08$ mag)")
axr.errorbar(zsn, res_sn, yerr=err, fmt="o", ms=2.8, mfc=C_SN, mec="#9c5a10",
             mew=0.2, alpha=0.65, ecolor="#d9a066", elinewidth=0.35, lw=0, zorder=2)
# cosmology residual curves drawn WITHOUT legend entries (no duplication)
axr.axhline(0.0, color=C_LCDM, lw=1.8, zorder=6)
axr.plot(zz, res_o, "--", color=C_OCDM, lw=1.7, zorder=6)
axr.plot(zz, res_e, "-.", color=C_EDS, lw=1.7, zorder=6)
axr.plot(zz, res_d, ":", color=C_DDE, lw=2.0, zorder=6)
axr.set_xlabel(r"Redshift $z$", fontsize=11)
axr.set_ylabel(r"$\Delta\mu$ (mag)", fontsize=11)
axr.set_xlim(0.0, 2.3)
axr.set_ylim(-0.55, 0.3)
axr.legend(fontsize=8.2, loc="lower left", ncol=2, framealpha=0.92)
axr.grid(alpha=0.12)

fig.savefig(OUT, dpi=192, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}")
