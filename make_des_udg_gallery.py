#!/usr/bin/env python3
"""Gallery of ultra-diffuse-galaxy candidates over the DES southern sky.

Twelve UDG candidates are drawn from the published SMUDGes catalog
(Zaritsky et al. 2023, ApJS 267, 27; VizieR J/ApJS/267/27), which selects
r_eff >= 5.3'' and mu_0(g) >= 24 mag/arcsec^2 over the full DECam Legacy
Surveys footprint. The selection here keeps the DES-accessible southern sky
(Dec < +5 deg), splits the central-surface-brightness range into twelve equal
bins, and shows the angularly largest candidate per bin, in DECam Legacy
Surveys DR10 colour imaging (Dey et al. 2019).
"""
import os
import urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from astroquery.vizier import Vizier

NSEL = 12
PIXSCALE = 0.262                      # Legacy Surveys native [arcsec/pix]
CACHE = "lsbg_cache"


def cutout(ra, dec, size_pix, tag):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{tag}.jpg")
    if not os.path.exists(path):
        url = ("https://www.legacysurvey.org/viewer/jpeg-cutout?"
               f"ra={ra:.5f}&dec={dec:.5f}&layer=ls-dr10&pixscale={PIXSCALE}"
               f"&size={size_pix}")
        print("fetching", tag)
        urllib.request.urlretrieve(url, path)
    return np.asarray(Image.open(path))

v = Vizier(columns=["SMDG", "RAJ2000", "DEJ2000", "Re", "mu0g", "gmag"],
           row_limit=-1)
cat = v.get_catalogs("J/ApJS/267/27")[0]
cat = cat[np.isfinite(cat["mu0g"]) & np.isfinite(cat["Re"])]
cat = cat[np.array(cat["DEJ2000"]) < 5.0]          # DES-accessible south
print(f"SMUDGes candidates at Dec < +5: {len(cat)}, "
      f"mu0g {cat['mu0g'].min():.1f}-{cat['mu0g'].max():.1f}")

# keep candidates of moderate angular size so each is well framed in its
# cutout; the very largest SMUDGes entries are often nearby giants or cirrus
cat = cat[np.array(cat["Re"]) <= 25.0]
edges = np.linspace(float(cat["mu0g"].min()), float(cat["mu0g"].max()) + 1e-3,
                    NSEL + 1)
sel, used = [], set()
for k in range(NSEL):
    inb = cat[(cat["mu0g"] >= edges[k]) & (cat["mu0g"] < edges[k + 1])]
    if len(inb):
        row = inb[np.argmax(inb["Re"])]
    else:                                          # empty bin: nearest unused in mu0
        rest = cat[[str(x) not in used for x in cat["SMDG"]]]
        row = rest[np.argmin(np.abs(rest["mu0g"] - 0.5 * (edges[k] + edges[k + 1])))]
    used.add(str(row["SMDG"]))
    sel.append(row)
print(f"selected {len(sel)} candidates")

fig, axes = plt.subplots(2, 6, figsize=(12.9, 4.78), dpi=150)
fig.subplots_adjust(left=0.004, right=0.996, top=0.90, bottom=0.008,
                    wspace=0.03, hspace=0.03)     # uniform thin white gaps
for ax, row in zip(axes.ravel(), sel):
    reff = float(row["Re"])                        # arcsec
    fov = max(8.0 * reff, 40.0)
    npx = int(np.clip(fov / PIXSCALE, 128, 512))
    name = str(row["SMDG"]).strip()
    img = cutout(float(row["RAJ2000"]), float(row["DEJ2000"]), npx,
                 f"udg_{name}")
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.04, 0.96, f"$\\mu_{{0,g}}$={float(row['mu0g']):.1f}",
            transform=ax.transAxes, color="white", fontsize=8.5, va="top")
    ax.text(0.04, 0.05, f"$r_{{\\rm e}}$={reff:.1f}$''$",
            transform=ax.transAxes, color="white", fontsize=7.5, va="bottom")
    bar = 10.0 / PIXSCALE / npx
    ax.plot([0.70, 0.70 + bar], [0.06, 0.06], color="white", lw=2,
            transform=ax.transAxes)
    ax.text(0.70 + bar / 2, 0.09, "10$''$", color="white", fontsize=6.5,
            ha="center", transform=ax.transAxes)
for ax in axes.ravel()[len(sel):]:
    ax.axis("off")
fig.suptitle("Ultra-diffuse-galaxy candidates over the DES southern sky from "
             "SMUDGes (Zaritsky et al. 2023), DECam Legacy Surveys DR10 imaging",
             fontsize=11, y=0.98)
fig.savefig("des_udg_gallery.png", bbox_inches="tight")
print("wrote des_udg_gallery.png")
for r in sel:
    print(f"  {str(r['SMDG']).strip():24s} RA {float(r['RAJ2000']):9.4f} "
          f"Dec {float(r['DEJ2000']):8.4f}  mu0g {float(r['mu0g']):.2f}  "
          f"Re {float(r['Re']):.1f}\"  g {float(r['gmag']):.1f}")
