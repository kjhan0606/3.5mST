#!/usr/bin/env python3
"""Observed rest-frame UV-optical spectra of the optical twins SN 2011fe and
SN 2011by near maximum light.

Both spectra are the flux-calibrated Swift/UVOT UV-grism observations
(erg/s/cm^2/A) retrieved from the Open Supernova Catalog archive (Guillochon
et al. 2017; astrocatalogs github), taken within half a day of B maximum for
each event, so the two are compared with the same instrument at the same
phase. They are deredshifted with the catalog redshifts and scaled to match
over 4000-4500 A, reproducing the near-identical optical and strongly
divergent near-UV of Foley & Kirshner (2013).
"""
import json
import os
import urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "https://raw.githubusercontent.com/astrocatalogs/sne-2010-2014/master"
CACHE = "osc_cache"
PICK = {"SN2011fe": ("55814.0", "black", "SN 2011fe"),
        "SN2011by": ("55691.0", "#2472e8", "SN 2011by")}


def get_spectrum(snname, time_str):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{snname}.json")
    if not os.path.exists(path):
        print("fetching", snname)
        urllib.request.urlretrieve(f"{BASE}/{snname}.json", path)
    d = json.load(open(path))
    k = list(d.keys())[0]
    z = float(d[k]["redshift"][0]["value"])
    for s in d[k]["spectra"]:
        if s.get("time") == time_str and "UV-grism" in str(s.get("instrument")):
            w = np.array([float(r[0]) for r in s["data"]])
            f = np.array([float(r[1]) for r in s["data"]])
            good = np.isfinite(f) & (f > 0)
            return w[good] / (1.0 + z), f[good], z
    raise KeyError(f"{snname} spectrum at {time_str} not found")


NUV_LO, NUV_HI = 2500.0, 3040.0  # proposed rest-frame NUV filter (Sec. 4.1, Fig. snuvlimit)

fig, ax = plt.subplots(figsize=(7.6, 4.9), dpi=150)
ax.axvspan(NUV_LO, NUV_HI, color="#e8873a", alpha=0.22, zorder=0, lw=0)
ax.text(0.5 * (NUV_LO + NUV_HI), 0.92, "NUV filter\n2500–3040Å",
        transform=ax.get_xaxis_transform(), ha="center", va="top",
        fontsize=8.5, color="#a8541a", zorder=3)
ref = None
for sn, (t, col, lab) in PICK.items():
    w, f, z = get_spectrum(sn, t)
    m = (w > 1950) & (w < 5050)
    w, f = w[m], f[m]
    opt = (w > 4000) & (w < 4500)
    if ref is None:
        ref = np.mean(f[opt])                     # normalise everything to 11fe
        scale = 1.0
    else:
        scale = ref / np.mean(f[opt])             # match 11by to 11fe over 4000-4500 A
    ax.plot(w, np.log10(np.clip(f * scale / ref, 1e-4, None)), color=col,
            lw=2.2, zorder=2, label=lab)
    print(f"{sn}: z={z}, {len(w)} points, scale={scale:.3g}")
ax.set_xlim(2250, 5000)
ax.set_ylim(-1.9, 0.55)
ax.set_xlabel(r"Rest wavelength [$\AA$]")
ax.set_ylabel(r"Relative $\log f_\lambda$")
ax.legend(fontsize=10, loc="upper right", frameon=False)
ax.text(0.97, 0.03, "Swift/UVOT UV grism, near $B$ maximum\n"
        "scaled to match over 4000–4500 $\\AA$",
        transform=ax.transAxes, fontsize=8.5, ha="right", va="bottom", color="0.25")
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig("snia_uvtwins.png", bbox_inches="tight")
print("wrote snia_uvtwins.png")
