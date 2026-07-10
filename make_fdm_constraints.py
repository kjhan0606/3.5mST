#!/usr/bin/env python3
"""Published constraints on the fuzzy-dark-matter boson mass (Figure
fig:fdmconstraints).

Every number is taken from the quoted paper. Preferred bands come from
stellar kinematics of dwarf spheroidals, m_B = (8.1 +1.6/-1.7)x10^-23 eV
(Schive et al. 2014) and m ~ (3.7-5.6)x10^-22 eV for Draco II and
Triangulum II (Calabrese & Spergel 2016). Lower bounds exclude the mass
range to their left, m > 2x10^-21 eV from the Lyman-alpha forest with a
conservative thermal history (Irsic et al. 2017, 2 sigma), m > 2.9x10^-21
eV from the Milky Way satellite census (Nadler et al. 2021, 95%),
m > 2x10^-20 eV from the Lyman-alpha forest marginalized over the IGM
(Rogers & Peiris 2021, 95%), and m > 3x10^-19 eV from the sizes and
stellar kinematics of Segue 1 and Segue 2 (Dalal & Kravtsov 2022, 99%).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

XMIN, XMAX = 2e-23, 2e-17

PREFERRED = [
    ("dSph stellar kinematics\n(Schive et al. 2014)", 6.4e-23, 9.7e-23),
    ("Draco II / Triangulum II kinematics\n(Calabrese & Spergel 2016)", 3.7e-22, 5.6e-22),
]
EXCLUDED = [
    (r"Ly$\alpha$ forest, conservative, $2\sigma$" + "\n(Iršič et al. 2017)", 2.0e-21),
    ("Milky Way satellite census, 95%\n(Nadler et al. 2021)", 2.9e-21),
    (r"Ly$\alpha$ forest, IGM marginalized, 95%" + "\n(Rogers & Peiris 2021)", 2.0e-20),
    ("Segue 1/2 sizes and kinematics, 99%\n(Dalal & Kravtsov 2022)", 3.0e-19),
]

rows = PREFERRED + EXCLUDED
ny = len(rows)

fig, ax = plt.subplots(figsize=(10.0, 4.6), dpi=180)

# kinematically preferred region carried down the whole chart
ax.axvspan(6.4e-23, 5.6e-22, color="#2f9e44", alpha=0.10, zorder=0)

for i, (label, lo, hi) in enumerate(PREFERRED):
    y = ny - 1 - i
    ax.barh(y, hi - lo, left=lo, height=0.52, color="#2f9e44", alpha=0.85,
            edgecolor="#1c6e2e", lw=1.0, zorder=3)
    ax.text(hi * 1.35, y, label, fontsize=8.6, va="center", zorder=4)

for j, (label, bound) in enumerate(EXCLUDED):
    y = ny - 1 - len(PREFERRED) - j
    ax.barh(y, bound - XMIN, left=XMIN, height=0.52, color="#d0392b",
            alpha=0.28, edgecolor="none", zorder=2)
    ax.plot([bound, bound], [y - 0.26, y + 0.26], color="#a02015", lw=2.4,
            zorder=3)
    ax.annotate("", xy=(bound * 0.42, y), xytext=(bound, y),
                arrowprops=dict(arrowstyle="-|>", color="#a02015", lw=1.6),
                zorder=3)
    ax.text(bound * 1.35, y, label, fontsize=8.6, va="center", zorder=4)

ax.text(1.9e-22, ny - 0.42, "kinematically preferred", color="#1c6e2e",
        fontsize=8.6, ha="center", va="bottom", style="italic")
ax.text(XMIN * 1.35, len(EXCLUDED) - 0.62, "excluded", color="#a02015",
        fontsize=8.6, ha="left", va="top", style="italic")

ax.set_xscale("log")
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(-0.75, ny - 0.05)
ax.set_yticks([])
ax.set_xlabel(r"fuzzy-dark-matter boson mass $m$ [eV]", fontsize=12)
ax.grid(axis="x", which="major", alpha=0.25, lw=0.6)
fig.tight_layout()
fig.savefig("fdm_mass_constraints.png", bbox_inches="tight")
print("wrote fdm_mass_constraints.png")
