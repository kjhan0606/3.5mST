#!/usr/bin/env python3
r"""
Generate realistic instrument/background data files for slitless_etc.py:

  etc_zodi.csv        zodiacal light spectrum built from the *real* CALSPEC solar
                      reference (sun_reference_stis) scattered-solar shape,
                      normalised to a Leinert-1998 reference brightness, plus the
                      interplanetary-dust thermal term.
  etc_qe.csv          detector quantum efficiency: e2v deep-depletion CCD (optical
                      arm) + Teledyne HgCdTe H2RG (near-IR arm), typical curves.
  etc_throughput.csv  total end-to-end throughput = 3 telescope reflections
                      (protected silver) x relay/dichroic x grism blaze x QE.

Sources: solar = STScI CALSPEC sun_reference_stis_002; zodi normalisation and
colour = Leinert et al. 1998 (A&AS 127, 1); detector QE and coating curves are
representative published values for e2v CCD and Teledyne H2RG devices.
All numbers live here so they can be swapped for measured hardware curves.
"""
import numpy as np

H, C, C_A = 6.62607015e-27, 2.99792458e10, 2.99792458e18
SR_PER_ARCSEC2 = 1.0 / 4.254517e10
LAM = np.arange(3400.0, 30050.0, 25.0)          # Angstrom grid


def ab_to_flambda(mu_ab, lam_A):
    return 10.0**(-(mu_ab + 48.60)/2.5) * C_A / lam_A**2


def planck_lambda(lam_A, T):
    lam = lam_A * 1e-8
    return (2*H*C**2/lam**5)/np.expm1(H*C/(lam*1.380649e-16*T)) * 1e-8   # per A / sr


# ------------------------------------------------------------------ zodiacal
def build_zodi(mu_ref=22.1, lam_ref=5000.0, tau_ipd=1.0e-7, T_ipd=265.0):
    from astropy.utils.data import download_file
    from astropy.io import fits
    url = ("https://archive.stsci.edu/hlsps/reference-atlases/cdbs/"
           "current_calspec/sun_reference_stis_002.fits")
    d = fits.getdata(download_file(url, cache=True, timeout=60))
    sw, sf = np.asarray(d["WAVELENGTH"], float), np.asarray(d["FLUX"], float)
    # extend past the CALSPEC red edge with a 5772 K blackbody (solar continuum)
    bb = planck_lambda(LAM, 5772.0)
    edge = sw.max()
    solar = np.where(LAM <= edge, np.interp(LAM, sw, sf, left=sf[0]),
                     bb * (np.interp(edge, sw, sf) / planck_lambda(edge, 5772.0)))
    scale = ab_to_flambda(mu_ref, lam_ref) / np.interp(lam_ref, sw, sf)
    scattered = scale * solar                                   # per A / arcsec^2
    # Leinert 1998: zodiacal light is ~0.3 mag redder than the Sun in (V-K);
    # apply a linear-in-wavelength reddening anchored at V (0.55 um), redward only.
    dmag = 0.30 * np.clip((LAM/1e4 - 0.55) / (2.2 - 0.55), 0.0, None)
    scattered *= 10.0 ** (dmag / 2.5)
    thermal = tau_ipd * planck_lambda(LAM, T_ipd) * SR_PER_ARCSEC2
    zodi = scattered + thermal
    np.savetxt("etc_zodi.csv", np.column_stack([LAM, zodi]), delimiter=",",
               header="lambda_A, I_lambda[erg/s/cm2/A/arcsec2]  (CALSPEC solar x "
                      "Leinert-1998 norm + IPD thermal)")
    mu = -2.5*np.log10(zodi*LAM**2/C_A) - 48.6
    print(f"etc_zodi.csv        : {LAM[0]:.0f}-{LAM[-1]:.0f} A, "
          f"mu(0.5um)={np.interp(5000,LAM,mu):.2f}, mu(1.6um)={np.interp(16000,LAM,mu):.2f} AB")


# ------------------------------------------------------------------ QE
def _interp_curve(pts):
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    return np.clip(np.interp(LAM, x, y, left=0.0, right=0.0), 0, 1)


def build_qe():
    # e2v deep-depletion CCD (optical); silicon rolls off smoothly to the ~1.1 um
    # band-gap cutoff (no abrupt edge, so the dichroic blend stays continuous)
    ccd = _interp_curve([(3400,0.25),(3600,0.35),(4000,0.62),(5000,0.85),
                         (6000,0.90),(7000,0.90),(8000,0.86),(9000,0.74),
                         (9500,0.62),(10000,0.50),(10500,0.22),(11000,0.05),(11200,0.0)])
    # Teledyne H2RG HgCdTe; blue response rises from ~0.8 um, 2.5 um red cutoff
    hgcdte = _interp_curve([(7800,0.0),(8200,0.20),(8600,0.42),(9000,0.62),
                           (10000,0.74),(12000,0.81),(15000,0.83),(20000,0.83),
                           (23000,0.78),(25000,0.60),(26000,0.42),(27000,0.22),
                           (28000,0.08),(30000,0.02)])
    # dichroic beam-splitter near 1.0 um: smooth transmission to the optical arm
    # (finite ~600 A transition), so the effective system QE blends the two arms
    # instead of stepping.  qe = T*QE_CCD + (1-T)*QE_HgCdTe.
    T = 0.5 * (1.0 - np.tanh((LAM - 10000.0) / 800.0))
    qe = T * ccd + (1.0 - T) * hgcdte
    np.savetxt("etc_qe.csv", np.column_stack([LAM, qe]), delimiter=",",
               header="lambda_A, QE  (e2v CCD + Teledyne H2RG blended by a 1.0um dichroic)")
    print(f"etc_qe.csv          : peak CCD {ccd.max():.2f}, peak HgCdTe {hgcdte.max():.2f}")
    return qe


# ------------------------------------------------------------------ throughput
def build_throughput(qe):
    # protected-silver mirror reflectivity (3 surfaces)
    Rm = _interp_curve([(3400,0.82),(3800,0.90),(4500,0.96),(5000,0.975),
                        (8000,0.98),(15000,0.985),(25000,0.985),(30000,0.98)])
    optics = Rm**3 * 0.88                     # 3 reflections x relay/dichroic (0.88)
    # grism first-order blaze efficiency (broad, ~0.70 peak)
    grism = _interp_curve([(3400,0.45),(4000,0.60),(6000,0.70),(12000,0.72),
                          (18000,0.70),(24000,0.62),(30000,0.50)])
    total = optics * grism * qe
    np.savetxt("etc_throughput.csv", np.column_stack([LAM, total]), delimiter=",",
               header="lambda_A, throughput  (3xAg mirror x optics x grism x QE)")
    print(f"etc_throughput.csv  : peak {total.max():.2f} at "
          f"{LAM[np.argmax(total)]/1e4:.2f} um")


if __name__ == "__main__":
    build_zodi()
    qe = build_qe()
    build_throughput(qe)
    print("done.")
