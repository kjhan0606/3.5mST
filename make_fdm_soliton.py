#!/usr/bin/env python3
"""Fuzzy (wave) dark-matter soliton core radius versus halo mass.

Plots the published core--halo relation of Schive et al. (2014, PRL 113,
261302) at z=0,
    r_c ~= 1.6 kpc (m/1e-22 eV)^-1 (M_halo/1e9 Msun)^(-1/3),
for several boson masses, with the boson-mass range preferred by
dwarf-spheroidal kinematics, m=(1-6)x1e-22 eV (Calabrese & Spergel 2016),
shown as a band, and the dwarf-galaxy halo-mass range 1e8-1e10 Msun where
the models diverge most strongly shaded.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Mh = np.logspace(8.0, 12.0, 300)


def rc(m22, mh):
    return 1.6 / m22 * (mh / 1e9) ** (-1.0 / 3.0)


fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=150)

ax.fill_between(Mh, rc(6.0, Mh), rc(1.0, Mh), color="#3898EC", alpha=0.22,
                label=r"dSph kinematics, $m=(1\!-\!6)\times10^{-22}$ eV"
                      " (Calabrese & Spergel 2016)")
for m22, ls, lw, c in [(0.5, ":", 1.6, "#8e44ad"),
                       (1.0, "-", 2.2, "#14224a"),
                       (2.0, "--", 1.6, "#2980b9"),
                       (6.0, "-.", 1.6, "#27ae60")]:
    ax.loglog(Mh, rc(m22, Mh), ls, color=c, lw=lw,
              label=rf"$m={m22:g}\times10^{{-22}}$ eV")

ax.axvspan(1e8, 1e10, color="#FFC107", alpha=0.14, zorder=0)
ax.text(9.7e8, 0.062, "dwarf-galaxy halos\n($10^8$–$10^{10}\\,M_\\odot$)",
        ha="center", va="bottom", fontsize=9.5, color="#7a6210")

ax.set_xlabel(r"halo mass $M_{\rm halo}$ [$M_\odot$]", fontsize=11)
ax.set_ylabel(r"soliton core radius $r_{\rm c}$ [kpc]", fontsize=11)
ax.set_xlim(1e8, 1e12)
ax.set_ylim(0.05, 20)
ax.grid(alpha=0.25, which="both", lw=0.4)
ax.legend(fontsize=8.8, loc="upper right", framealpha=0.9)
ax.set_title("Fuzzy-dark-matter core–halo relation (Schive et al. 2014, $z=0$)",
             fontsize=11)
fig.tight_layout()
fig.savefig("fdm_soliton_relation.png", bbox_inches="tight")
print("wrote fdm_soliton_relation.png")
for m22 in (0.5, 1.0, 2.0, 6.0):
    print(f"  m22={m22:g}: r_c(1e9 Msun)={rc(m22, 1e9):.2f} kpc, "
          f"r_c(1e10)={rc(m22, 1e10):.2f} kpc")
