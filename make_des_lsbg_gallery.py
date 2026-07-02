#!/usr/bin/env python3
"""Gallery of low-surface-brightness galaxies from the published DES catalog.

Twelve LSBGs are drawn from the Dark Energy Survey DR1 LSBG catalog of
Tanoglidis et al. (2021, ApJS 252, 18; VizieR J/ApJS/252/18), selected to span
the requested mean g-band surface brightness window 25-30 mag/arcsec^2 (the
published catalog itself extends to mu_g = 27.2, so the selection spans
25.0-27.2). Within twelve equal surface-brightness bins the angularly largest
object is chosen so the diffuse structure is well resolved. The colour images
are DECam Legacy Surveys DR10 cutouts (Dey et al. 2019), which include the DES
imaging in the south.
"""
import os
import urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from astroquery.vizier import Vizier

MU_LO, MU_HI = 25.0, 30.0            # requested window [mag/arcsec^2]
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


v = Vizier(columns=["ID", "RAJ2000", "DEJ2000", "gmu", "gReff", "gmagSE", "n"],
           row_limit=-1)
cat = v.get_catalogs("J/ApJS/252/18")[1]
cat = cat[np.isfinite(cat["gmu"]) & np.isfinite(cat["gReff"])]
m = (cat["gmu"] >= MU_LO) & (cat["gmu"] <= MU_HI)
cat = cat[m]
print(f"catalog objects with {MU_LO} <= mu_g <= {MU_HI}: {len(cat)} "
      f"(published range tops out at {cat['gmu'].max():.2f})")

# twelve equal surface-brightness bins; take the angularly largest per bin
edges = np.linspace(MU_LO, float(cat["gmu"].max()) + 1e-3, NSEL + 1)
sel = []
for k in range(NSEL):
    inb = cat[(cat["gmu"] >= edges[k]) & (cat["gmu"] < edges[k + 1])]
    if len(inb) == 0:
        continue
    sel.append(inb[np.argmax(inb["gReff"])])
print(f"selected {len(sel)} galaxies")

fig, axes = plt.subplots(2, 6, figsize=(13.2, 4.9), dpi=150)
for ax, row in zip(axes.ravel(), sel):
    reff = float(row["gReff"])                    # arcsec (angular, DES DR1)
    fov = max(8.0 * reff, 40.0)                   # arcsec
    npx = int(np.clip(fov / PIXSCALE, 128, 512))
    img = cutout(float(row["RAJ2000"]), float(row["DEJ2000"]), npx,
                 f"lsbg{int(row['ID'])}")
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.04, 0.96, f"$\\bar\\mu_g$={float(row['gmu']):.1f}",
            transform=ax.transAxes, color="white", fontsize=8.5, va="top")
    ax.text(0.04, 0.05, f"$r_{{\\rm e}}$={reff:.1f}$''$",
            transform=ax.transAxes, color="white", fontsize=7.5, va="bottom")
    bar = 10.0 / PIXSCALE / npx                   # 10 arcsec, axes fraction
    ax.plot([0.70, 0.70 + bar], [0.06, 0.06], color="white", lw=2,
            transform=ax.transAxes)
    ax.text(0.70 + bar / 2, 0.09, "10$''$", color="white", fontsize=6.5,
            ha="center", transform=ax.transAxes)
fig.suptitle("Low-surface-brightness galaxies from the DES DR1 LSBG catalog "
             "(Tanoglidis et al. 2021), DECam Legacy Surveys DR10 imaging",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("des_lsbg_gallery.png", bbox_inches="tight")
print("wrote des_lsbg_gallery.png")
for r in sel:
    print(f"  ID {int(r['ID']):6d}  RA {float(r['RAJ2000']):9.4f}  "
          f"Dec {float(r['DEJ2000']):8.4f}  mu_g {float(r['gmu']):.2f}  "
          f"reff {float(r['gReff']):.1f}\"  g {float(r['gmagSE']):.1f}")
