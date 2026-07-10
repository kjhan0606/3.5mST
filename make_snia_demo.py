#!/usr/bin/env python3
"""Type Ia SN at peak: observed AB magnitude per filter vs redshift, for the
3.5 m ST. Uses the Nugent SN Ia template (phase 0) normalised to M_B = -19.3."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import slitless_etc as etc
import galaxy_etc as g

HR = 3600.0
MB = -19.3                                     # standard SN Ia peak, rest B
d = np.loadtxt("sn_templates/SNIa_peak.dat")
snia = dict(wl=d[:, 0], flam=d[:, 1], desc="SN Ia peak (Nugent)")
cfg = etc.realistic_cfg(diameter_cm=350.0, obstruction=0.15, pix_scale=0.11, R=1000.0)
FILT = etc.INSTRUMENT_ELEMENTS["3.5mST"]["Imaging"]     # g/z/J/H/K
zz = np.linspace(0.1, 3.0, 45)

# depths (5sigma point source) for wide (0.75 hr) and deep (12 hr) tiers, in H
lamH, wH = FILT["H~1.6"]
dep_wide = etc.imaging_maglimit(cfg, lamH*1e4, wH*1e4, 0.75*HR)
dep_deep = etc.imaging_maglimit(cfg, lamH*1e4, wH*1e4, 12*HR)

fig, (ax, axm) = plt.subplots(1, 2, figsize=(14.4, 5.2), dpi=150)
for (fn, (lu, wu)), col in zip(FILT.items(), plt.cm.viridis(np.linspace(0, 0.85, len(FILT)))):
    m = [g.app_mag(snia, z, lu*1e4, wu*1e4, M_abs=MB, ref_lam=4400.0, ref_width=1000.0) for z in zz]
    ax.plot(zz, m, color=col, lw=2, label=fn)
ax.axhline(dep_wide, ls="--", color="0.4", lw=1)
ax.axhline(dep_deep, ls=":", color="0.4", lw=1)
ax.text(0.12, dep_wide-0.15, f"wide 0.75 hr ({dep_wide:.1f})", fontsize=7, color="0.3")
ax.text(0.12, dep_deep-0.15, f"deep 12 hr ({dep_deep:.1f})", fontsize=7, color="0.3")
ax.invert_yaxis(); ax.set_xlabel("redshift z"); ax.set_ylabel("observed peak AB mag")
ax.legend(fontsize=8, title="filter"); ax.grid(alpha=0.2)

# ---- rest-frame NUV access: observed mag of the rest-NUV band + exposure to 5sigma ----
# The optical-twin / NUV-divergence diagnostic sits near rest 2770 A (Mg II / NUV).
LAM_NUV = 2770.0                                   # rest A
WOBS = 0.12e4                                       # observed filter width [A]
znuv = np.linspace(0.30, 3.0, 40)                   # rest-NUV enters the band above z~0.3
mnuv, tnuv = [], []
for z in znuv:
    lo = LAM_NUV * (1 + z)                           # observed wavelength of rest-NUV
    m = g.app_mag(snia, z, lo, WOBS, M_abs=MB, ref_lam=4400.0, ref_width=1000.0)
    mnuv.append(m)
    tnuv.append(g.exptime_to_snr(cfg, m, lo, WOBS, snr=5.0) / HR)
axm.plot(znuv, mnuv, color="#7d3c98", lw=2.2, label="rest-NUV (2770 Å) observed AB")
axm.invert_yaxis(); axm.set_xlabel("redshift z"); axm.set_ylabel("observed AB mag", color="#7d3c98")
axm.tick_params(axis="y", labelcolor="#7d3c98")
axt = axm.twinx()
axt.plot(znuv, tnuv, color="#d97757", lw=2.2, ls="--", label="exposure to 5σ")
axt.set_yscale("log"); axt.set_ylabel("exposure to 5σ [hr]", color="#d97757")
axt.tick_params(axis="y", labelcolor="#d97757"); axt.axhline(1.0, ls=":", color="0.6", lw=1)
axm.grid(alpha=0.2)
fig.tight_layout(); fig.savefig("snia_demo_combined.png", bbox_inches="tight")
print("wrote snia_demo_combined.png")
print("\nSN Ia rest-NUV (2770A) observed AB and hr-to-5sigma:")
for z in (0.5, 1.0, 1.5, 2.0):
    lo = LAM_NUV*(1+z); m = g.app_mag(snia, z, lo, WOBS, M_abs=MB, ref_lam=4400., ref_width=1000.)
    print(f"  z={z:.1f}  obsλ={lo/1e4:.2f}µm  AB={m:.1f}  5σ in {g.exptime_to_snr(cfg,m,lo,WOBS,5.0)/HR:.3f} hr")

print(f"\nSN Ia peak observed AB (M_B=-19.3):   wide depth(H)={dep_wide:.1f}, deep={dep_deep:.1f}")
print(f"{'z':>4} " + " ".join(f"{fn:>7}" for fn in FILT))
for z in (0.5, 1.0, 1.5, 2.0, 2.5):
    cells = [f"{g.app_mag(snia, z, lu*1e4, wu*1e4, M_abs=MB, ref_lam=4400., ref_width=1000.):.1f}"
             for lu, wu in FILT.values()]
    print(f"{z:>4.1f} " + " ".join(f"{c:>7}" for c in cells))
