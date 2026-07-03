#!/usr/bin/env python3
"""Galaxy-template photometry for the space ETC.

Loads the bundled FSPS galaxy SEDs (galaxy_templates/, made by
make_galaxy_templates.py), redshifts them, normalises to a chosen absolute
magnitude, and returns the observed AB magnitude in a filter as a function of
redshift, plus the exposure time to reach a target S/N. No FSPS needed here --
the templates are read from disk, so the ETC is standalone.
"""
import os
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import slitless_etc as etc

# absolute-magnitude ladder (rest V), giant galaxies down to dwarfs
MV_LEVELS = [
    ("giant elliptical", -22.0),
    ("L* galaxy",        -21.0),
    ("large spiral",     -20.0),
    ("sub-L*",           -19.0),
    ("small galaxy",     -17.0),
    ("bright dwarf",     -15.0),
    ("dwarf",            -13.0),
    ("faint dwarf",      -11.0),
]

COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)
PC10_CM = 10.0 * 3.0856775814913673e18      # 10 pc in cm
C_AAS = 2.99792458e18                        # speed of light in A/s
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "galaxy_templates")


def load_templates(dirpath=TEMPLATE_DIR):
    """Return (dict name->{wl,flam,desc}, ordered list of names)."""
    order = []
    idx = os.path.join(dirpath, "index.txt")
    if os.path.exists(idx):
        for line in open(idx):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            order.append((parts[0], parts[1] if len(parts) > 1 else ""))
    else:
        for fn in sorted(os.listdir(dirpath)):
            if fn.endswith(".dat"):
                order.append((fn[:-4], ""))
    tmpl = {}
    for name, desc in order:
        d = np.loadtxt(os.path.join(dirpath, name + ".dat"))
        tmpl[name] = dict(wl=d[:, 0], flam=d[:, 1], desc=desc)
    return tmpl, [n for n, _ in order]


def _tophat_fnu(lam, fnu, lam0, width):
    """Photon-weighted AB average of f_nu over a top-hat filter [lam0+-width/2]."""
    lo, hi = lam0 - width / 2.0, lam0 + width / 2.0
    m = (lam >= lo) & (lam <= hi)
    if m.sum() < 2:
        return float(np.interp(lam0, lam, fnu))
    w = lam[m]                                # photon weighting dlam/lam
    return float(np.trapezoid(fnu[m] / w, w) / np.trapezoid(1.0 / w, w))


def _abs_scale(tmpl, M_abs, ref_lam=5500.0, ref_width=1000.0):
    """Scale factor so the rest-frame AB mag in the reference band equals M_abs."""
    wl, flam = tmpl["wl"], tmpl["flam"]
    fnu = flam * wl * wl / C_AAS
    fbar = _tophat_fnu(wl, fnu, ref_lam, ref_width)
    target = 10 ** (-0.4 * (M_abs + 48.6))    # f_nu at 10 pc giving M_abs
    return target / fbar


def app_mag(tmpl, z, lam0_A, width_A, M_abs=-21.0, ref_lam=5500.0, ref_width=1000.0):
    """Observed AB magnitude of a galaxy of absolute mag M_abs at redshift z,
    through a top-hat filter centred at lam0_A [A] with width_A [A]."""
    wl, flam = tmpl["wl"], tmpl["flam"]
    fnu10 = flam * wl * wl / C_AAS * _abs_scale(tmpl, M_abs, ref_lam, ref_width)
    if z <= 0:
        lam_obs, fnu_obs = wl, fnu10
    else:
        DL = COSMO.luminosity_distance(z).to("cm").value
        lam_obs = wl * (1.0 + z)
        fnu_obs = (1.0 + z) * (PC10_CM / DL) ** 2 * fnu10
    fbar = _tophat_fnu(lam_obs, fnu_obs, lam0_A, width_A)
    return np.inf if fbar <= 0 else -2.5 * np.log10(fbar) - 48.6


def exptime_to_snr(cfg, app_ab, lam0_A, width_A, snr=5.0, aper_fwhm_mult=1.0):
    """Exposure time [s] for an unresolved source of apparent AB mag to reach snr."""
    fn = lambda t: etc.imaging_snr(cfg, app_ab, lam0_A, width_A, t, aper_fwhm_mult)
    return etc.exposure_for_snr(fn, snr)


if __name__ == "__main__":
    tmpl, order = load_templates()
    print(f"{len(order)} templates:", ", ".join(order))
    cfg = etc.realistic_cfg(diameter_cm=350.0, pix_scale=0.11, R=1000.0)
    lam0, w = 1.6e4, 0.4e4        # H band
    print(f"\nApparent AB in H (1.6um) for M_V=-21, and hr-to-5sigma (3.5m ST):")
    print(f"{'type':16} " + "".join(f"z={z:<7.1f}" for z in (0.5, 1.0, 2.0)))
    for name in order:
        cells = []
        for z in (0.5, 1.0, 2.0):
            m = app_mag(tmpl[name], z, lam0, w)
            t = exptime_to_snr(cfg, m, lam0, w) / 3600.0
            cells.append(f"{m:5.1f}/{t:5.2f}h")
        print(f"{name:16} " + " ".join(cells))
