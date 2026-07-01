#!/usr/bin/env python3
"""Generate a diverse set of galaxy SED templates with FSPS and save them so the
ETC is standalone (no FSPS needed at run time).

Each template is written to galaxy_templates/<name>.dat as two columns
(rest-frame wavelength [A], f_lambda [Lsun/A, arbitrary norm]), trimmed to the
rest range that maps into 0.36-5 um for z = 0-3. A galaxy_templates/index.txt
lists the types in Hubble-sequence order with a one-line description.
"""
import os
os.environ.setdefault("SPS_HOME", "/home/kjhan/fsps_data")
import numpy as np
import fsps

WL_MIN, WL_MAX = 900.0, 55000.0     # rest A (covers 0.36-5 um observed at z=0-3)

# (name, description, params).  sfh: 0=SSP, 1=constant SFR, 4=delayed-tau.
TYPES = [
    ("Elliptical",     "old passive, delayed-tau", dict(sfh=4, tau=0.3, tage=11.0, logzsol=0.1,  dust2=0.10)),
    ("S0",             "lenticular, mostly old",   dict(sfh=4, tau=0.8, tage=10.0, logzsol=0.0,  dust2=0.15)),
    ("Sa",             "early spiral",             dict(sfh=4, tau=1.5, tage=9.0,  logzsol=0.0,  dust2=0.25)),
    ("Sb",             "spiral",                   dict(sfh=4, tau=3.0, tage=8.0,  logzsol=0.0,  dust2=0.35)),
    ("Sc",             "late spiral",              dict(sfh=4, tau=6.0, tage=7.0,  logzsol=-0.2, dust2=0.45)),
    ("Sd",             "very late spiral, const SFR", dict(sfh=1, tage=5.0,  logzsol=-0.3, dust2=0.40)),
    ("Irregular",      "blue dwarf/irregular",     dict(sfh=1, tage=1.5,  logzsol=-0.7, dust2=0.20)),
    ("Starburst",      "young starburst, emission lines", dict(sfh=1, tage=0.05, logzsol=-0.4, dust2=0.50)),
    ("Dusty_starburst","reddened starburst (Av~1.6)", dict(sfh=1, tage=0.10, logzsol=0.0,  dust2=1.50)),
    ("Quiescent",      "red-and-dead SSP",         dict(sfh=0, tage=12.0, logzsol=0.2,  dust2=0.05)),
]

os.makedirs("galaxy_templates", exist_ok=True)
sp = fsps.StellarPopulation(zcontinuous=1, imf_type=1, dust_type=2,
                            add_neb_emission=True, sfh=4, tau=1.0)

index = []
for name, desc, p in TYPES:
    sp.params["sfh"] = p["sfh"]
    sp.params["const"] = 1.0 if p["sfh"] == 1 else 0.0
    if "tau" in p:
        sp.params["tau"] = p["tau"]
    sp.params["logzsol"] = p["logzsol"]
    sp.params["dust2"] = p["dust2"]
    wl, spec = sp.get_spectrum(tage=p["tage"], peraa=True)
    m = (wl >= WL_MIN) & (wl <= WL_MAX)
    w, f = wl[m], spec[m]
    f = f / np.median(f[f > 0])                       # arbitrary norm, ~O(1)
    out = f"galaxy_templates/{name}.dat"
    np.savetxt(out, np.column_stack([w, f]),
               header=f"{name}: {desc} (FSPS)  |  rest_lambda[A]  f_lambda[arb]")
    # crude D4000 and UV/optical slope as a sanity print
    r = np.trapz(f[(w > 4050) & (w < 4250)], w[(w > 4050) & (w < 4250)])
    b = np.trapz(f[(w > 3750) & (w < 3950)], w[(w > 3750) & (w < 3950)]) or np.nan
    print(f"{name:16} N={len(w):5d}  D4000={r/b:4.2f}  {desc}")
    index.append(f"{name}\t{desc}")

with open("galaxy_templates/index.txt", "w") as fh:
    fh.write("# FSPS galaxy templates, Hubble-sequence order\n")
    fh.write("\n".join(index) + "\n")
print("wrote galaxy_templates/ (%d types) + index.txt" % len(TYPES))
