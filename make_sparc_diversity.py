#!/usr/bin/env python3
"""The rotation-curve diversity problem in real data (Figure fig:sparcdiv).

Observed rotation curves of IC 2574 and UGC 5721 from the public SPARC
database of 175 late-type galaxies (Lelli, McGaugh & Schombert 2016,
Rotmod_LTG mass models, astroweb.cwru.edu/SPARC). The two dwarfs reach
nearly the same outermost rotation speed, so they occupy halos of nearly
the same mass, yet the inner curve of UGC 5721 rises several times faster
than that of IC 2574. This pair is the standard illustration of the
diversity problem (Oman et al. 2015) and of the coexistence of dense and
cored dwarfs that self-interacting dark matter explains through core
formation and gravothermal collapse (Ren et al. 2019, which fits both
galaxies with the same cross section).
"""
import os
import urllib.request
import zipfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

URL = "https://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip"
CACHE = "sparc_cache"
GALAXIES = [
    ("IC2574", "IC 2574", "#2a6cb5", "o", "cored, slowly rising"),
    ("UGC05721", "UGC 5721", "#c0392b", "s", "dense, steeply rising"),
]


def get_rotmod(name):
    os.makedirs(CACHE, exist_ok=True)
    zpath = os.path.join(CACHE, "Rotmod_LTG.zip")
    if not os.path.exists(zpath):
        print("fetching", URL)
        urllib.request.urlretrieve(URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        with z.open(f"{name}_rotmod.dat") as f:
            rows = [ln.split() for ln in f.read().decode().splitlines()
                    if ln.strip() and not ln.startswith("#")]
    a = np.array(rows, float)
    return a[:, 0], a[:, 1], a[:, 2]          # R [kpc], Vobs, errV [km/s]


fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=180)
for key, label, col, mk, note in GALAXIES:
    r, v, ev = get_rotmod(key)
    vout = np.mean(v[-3:])
    v2 = np.interp(2.0, r, v)
    ax.errorbar(r, v, yerr=ev, fmt=mk, ms=4.5, color=col, mfc=col, mec=col,
                elinewidth=1.0, capsize=2, lw=1.4, ls="-",
                label=f"{label} ({note})")
    print(f"{key}: V_out = {vout:.1f} km/s, V(2 kpc) = {v2:.1f} km/s, "
          f"{r.size} points, R = {r.min():.2f}..{r.max():.2f} kpc")

ax.axvline(2.0, color="0.55", ls=":", lw=1.2, zorder=1)
ax.text(2.06, 5, r"$R=2$ kpc", color="0.4", fontsize=9, ha="left", va="bottom")
ax.set_xlim(0, 13.5)
ax.set_ylim(0, 95)
ax.set_xlabel("radius [kpc]", fontsize=12)
ax.set_ylabel(r"observed rotation velocity $V_{\rm obs}$ [km s$^{-1}$]",
              fontsize=12)
ax.legend(fontsize=10, loc="lower right", framealpha=0.92)
ax.grid(alpha=0.2)
ax.set_title("Two dwarfs of nearly equal halo mass with opposite inner structure\n"
             "(SPARC observed rotation curves)", fontsize=11)
fig.tight_layout()
fig.savefig("sparc_diversity.png", bbox_inches="tight")
print("wrote sparc_diversity.png")
