#!/usr/bin/env python3
"""Demo: redshift tracks of galaxy AB magnitude per filter and the exposure
time to reach 5 sigma, for the 3.5 m ST. Uses the bundled FSPS templates."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import slitless_etc as etc
import galaxy_etc as g

HR = 3600.0
M_ABS = -21.0                      # ~L* galaxy in rest V
cfg = etc.realistic_cfg(diameter_cm=350.0, obstruction=0.15, pix_scale=0.11, R=1000.0)
tmpl, order = g.load_templates()
zz = np.linspace(0.1, 5.0, 60)
FILTERS = etc.INSTRUMENT_ELEMENTS["3.5mST"]["Imaging"]     # g/z/J/H/K
cmap = plt.cm.turbo(np.linspace(0.05, 0.95, len(order)))

# ---- Figure 1: mag vs z, (left) per filter for one galaxy, (right) per type in H
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=150)
demo_type = "Sc"
for (fname, (lu, wu)), col in zip(FILTERS.items(), plt.cm.viridis(np.linspace(0, 0.85, len(FILTERS)))):
    m = [g.app_mag(tmpl[demo_type], z, lu*1e4, wu*1e4, M_ABS) for z in zz]
    a1.plot(zz, m, color=col, lw=2, label=fname)
a1.invert_yaxis(); a1.set_xlabel("redshift z"); a1.set_ylabel("apparent AB mag")
a1.set_title(f"{demo_type} galaxy ($M_V=-21$): mag per filter vs z")
a1.legend(fontsize=8, title="3.5 m ST filter"); a1.grid(alpha=0.2)

lamH, wH = FILTERS["H~1.6"]
for name, col in zip(order, cmap):
    m = [g.app_mag(tmpl[name], z, lamH*1e4, wH*1e4, M_ABS) for z in zz]
    a2.plot(zz, m, color=col, lw=1.8, label=name)
a2.invert_yaxis(); a2.set_xlabel("redshift z"); a2.set_ylabel("apparent AB mag (H, 1.6 µm)")
a2.set_title("Galaxy type vs z, H band ($M_V=-21$)")
a2.legend(fontsize=7, ncol=2); a2.grid(alpha=0.2)
fig.tight_layout(); fig.savefig("galaxy_mag_vs_z.png", bbox_inches="tight")
print("wrote galaxy_mag_vs_z.png")

# ---- Figure 2: hours to 5 sigma vs z, (left) M_V ladder incl. dwarfs, (right) per type
fig2, (b1, b2) = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=150)
TMAX = 100.0                    # cap at 100 hr for readability
mvcol = plt.cm.plasma(np.linspace(0.05, 0.9, len(g.MV_LEVELS)))
for (lab, mv), col in zip(g.MV_LEVELS, mvcol):
    t = []
    for z in zz:
        m = g.app_mag(tmpl["Sc"], z, lamH*1e4, wH*1e4, mv)
        t.append(min(g.exptime_to_snr(cfg, m, lamH*1e4, wH*1e4, 5.0) / HR, TMAX))
    b1.plot(zz, t, color=col, lw=1.9, label=f"{lab} ($M_V={mv:.0f}$)")
b1.set_yscale("log"); b1.set_xlabel("redshift z")
b1.set_ylabel("exposure to 5σ [hr], H band")
b1.set_title("Sc galaxy: hours to 5σ vs luminosity (to dwarfs)")
b1.axhline(1.0, ls=":", color="k", lw=1); b1.axhline(TMAX, ls="--", color="0.6", lw=1)
b1.grid(alpha=0.2, which="both"); b1.legend(fontsize=7)

for name, col in zip(order, cmap):
    t = []
    for z in zz:
        m = g.app_mag(tmpl[name], z, lamH*1e4, wH*1e4, M_ABS)
        t.append(min(g.exptime_to_snr(cfg, m, lamH*1e4, wH*1e4, 5.0) / HR, TMAX))
    b2.plot(zz, t, color=col, lw=1.7, label=name)
b2.set_yscale("log"); b2.set_xlabel("redshift z")
b2.set_ylabel("exposure to 5σ [hr], H band")
b2.set_title("Hours to 5σ per type ($M_V=-21$)")
b2.axhline(1.0, ls=":", color="k", lw=1); b2.grid(alpha=0.2, which="both")
b2.legend(fontsize=7, ncol=2)
fig2.tight_layout(); fig2.savefig("galaxy_exptime_vs_z.png", bbox_inches="tight")
print("wrote galaxy_exptime_vs_z.png")
