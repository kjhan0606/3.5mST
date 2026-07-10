#!/usr/bin/env python3
"""Two diffuse/faint science cases for the 3.5 m ST ETC:
 (1) intracluster light (ICL): exposure vs redshift to reach a target surface
     brightness, including the (1+z)^4 cosmological surface-brightness dimming;
 (2) dark / almost-dark galaxies: exposure to detect the Halpha line as a
     function of star-formation rate in a nearby void.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import slitless_etc as etc

HR = 3600.0
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)
cfg = etc.realistic_cfg(diameter_cm=350.0, obstruction=0.15, pix_scale=0.11, R=1000.0,
                        band_min_A=3600.0, band_max_A=30000.0)


# ---------- (1) ICL surface-brightness detection ----------
def sb_rate_per_pix(cfg, mu_ab, band):
    """e-/s/pixel from a uniform source of AB surface brightness mu [mag/arcsec^2]."""
    lo, hi = band
    lam = np.linspace(lo, hi, 300)
    fnu = 3631.0e-23 * 10 ** (-0.4 * mu_ab)                 # erg/s/cm2/Hz/arcsec2
    flam = fnu * 2.99792458e18 / lam ** 2
    surf = np.trapezoid(flam * etc._photon_factor(lam) * cfg.throughput_imaging(lam), lam)
    return surf * cfg.area_cm2 * cfg.omega_pix


def sb_snr(cfg, mu_ab, band, t, bin_arcsec2):
    src = sb_rate_per_pix(cfg, mu_ab, band)
    sky = etc.background_per_pixel(cfg, band, imaging=True)
    n_bin = bin_arcsec2 / cfg.omega_pix
    var_pix = (src + sky + cfg.dark_current) * t + cfg.n_reads(t) * cfg.read_noise ** 2
    return src * t * np.sqrt(n_bin) / np.sqrt(var_pix)


def sb_exptime(cfg, mu_ab, band, snr, bin_arcsec2):
    return etc.exposure_for_snr(lambda t: sb_snr(cfg, mu_ab, band, t, bin_arcsec2), snr)


BAND_H = (14000.0, 18000.0)          # H-ish imaging band (1.4-1.8 um)
BIN = 100.0                           # 10x10 arcsec^2 binning for diffuse ICL
zz = np.linspace(0.05, 1.0, 40)
fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=150)
for mu_rest, col in [(25.0, "#3898ec"), (26.0, "#4ec9b0"), (27.0, "#d97757"), (28.0, "#7d3c98")]:
    t = []
    for z in zz:
        mu_obs = mu_rest + 10.0 * np.log10(1 + z)          # (1+z)^4 SB dimming
        t.append(min(sb_exptime(cfg, mu_obs, BAND_H, 5.0, BIN) / HR, 300.0))
    ax.plot(zz, t, color=col, lw=2, label=fr"$\mu_{{\rm rest}}={mu_rest:.0f}$ AB/arcsec$^2$")
ax.set_yscale("log"); ax.set_xlabel("cluster redshift z")
ax.set_ylabel("exposure to 5σ over 10×10″ [hr]")
ax.axhline(1.0, ls=":", color="k", lw=1); ax.grid(alpha=0.2, which="both"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("icl_exptime_vs_z.png", bbox_inches="tight")
print("wrote icl_exptime_vs_z.png")
for z in (0.1, 0.3, 0.5):
    row = [f"{sb_exptime(cfg, 26.0+10*np.log10(1+z), BAND_H, 5.0, BIN)/HR:.2f}"]
    print(f"  ICL mu_rest=26 at z={z}: obs mu={26+10*np.log10(1+z):.1f}, 5σ(10x10) in {row[0]} hr")


# ---------- (2) dark-galaxy Halpha detection vs SFR ----------
KENN = 7.9e-42                       # SFR = KENN * L_Halpha  (Kennicutt 1998)
z_void = 0.05
DL = COSMO.luminosity_distance(z_void).to("cm").value
lam_Ha = 6563.0 * (1 + z_void)
sfr = np.logspace(-3.5, 0.0, 40)
texp = []
for s in sfr:
    L = s / KENN
    F = L / (4 * np.pi * DL ** 2)
    texp.append(min(etc.exposure_for_snr(
        lambda t: etc.line_sn(cfg, F, lam_Ha, t, source_fwhm=0.5), 5.0) / HR, 300.0))
fig2, ax2 = plt.subplots(figsize=(7.6, 5.2), dpi=150)
ax2.loglog(sfr, texp, color="#3898ec", lw=2.2)
ax2.axhline(1.0, ls=":", color="k", lw=1); ax2.axhline(12.0, ls="--", color="0.6", lw=1)
ax2.text(1.2e-3, 1.1, "1 hr", fontsize=8); ax2.text(1.2e-3, 13, "12 hr deep", fontsize=8)
ax2.set_xlabel(r"star-formation rate [$M_\odot\,{\rm yr}^{-1}$]")
ax2.set_ylabel("exposure to 5σ Hα [hr]")
ax2.grid(alpha=0.2, which="both")
fig2.tight_layout(); fig2.savefig("darkgal_exptime_vs_sfr.png", bbox_inches="tight")
print("wrote darkgal_exptime_vs_sfr.png")
for s in (1e-3, 3e-3, 1e-2, 1e-1):
    L = s/KENN; F = L/(4*np.pi*DL**2)
    tt = etc.exposure_for_snr(lambda t: etc.line_sn(cfg, F, lam_Ha, t, source_fwhm=0.5), 5.0)/HR
    print(f"  SFR={s:.0e} Msun/yr: F(Ha)={F:.2e}, 5σ in {tt:.3f} hr")
