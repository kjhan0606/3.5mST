#!/usr/bin/env python3
r"""
slitless_etc.py  --  Emission-line exposure-time / depth calculator for a
wide-field slitless grism survey (3.5 m segmented-mirror R=1000 concept).

Fidelity target: Euclid/Roman-class ETC.  Key ingredients that the proposal's
closed-form Eq.(16) collapses into single numbers are here computed
wavelength-by-wavelength:

  * Zodiacal (diffuse) background modelled as a *spectrum*: scattered solar
    (continuum, normalised to a Leinert-1998 reference surface brightness) plus
    an interplanetary-dust thermal graybody.  (Leinert et al. 1998, A&AS 127, 1)
  * Optional telescope self-emission graybody (matters lambda > ~2.5 um).
  * Wavelength-dependent total throughput and detector quantum efficiency.
  * Full per-pixel noise budget: source shot + sky + dark + read (ramp-sampled),
    added in quadrature over the line footprint n_pix.
  * Slitless background is the sky integrated over the *band-limiting filter*,
    not one resolution element (the feature that makes slitless surveys shallow).
  * F_5sigma(lambda, t) obtained by solving S/N = target including source shot
    noise (a quadratic in F), so it is exact, not only background-limited.

Methodology follows the standard space-ETC construction used for Roman (Hirata
et al.; WFIRST/Roman ETC) and Euclid (Euclid Red Book, Laureijs et al. 2011).
Detector terms are baseline values typical of near-IR HgCdTe (H2RG/H4RG) and
optical CCD arrays; every number is an attribute of InstrumentConfig so it can
be replaced by measured hardware values or by a CSV throughput/QE/zodi file.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
import numpy as np

# ---------------------------------------------------------------- constants (CGS)
H  = 6.62607015e-27          # erg s
C  = 2.99792458e10           # cm / s
KB = 1.380649e-16            # erg / K
C_A = 2.99792458e18          # Angstrom / s
SR_PER_ARCSEC2 = 1.0 / 4.254517e10   # steradian in one arcsec^2


def planck_lambda(lam_A, T):
    """Planck radiance B_lambda(T) [erg s^-1 cm^-2 Angstrom^-1 sr^-1]."""
    lam = np.asarray(lam_A, float) * 1e-8            # cm
    B = (2.0 * H * C**2 / lam**5) / np.expm1(H * C / (lam * KB * T))  # per cm / sr
    return B * 1e-8                                   # per Angstrom / sr


def ab_to_flambda(mu_ab, lam_A):
    """AB surface brightness [mag/arcsec^2] -> f_lambda [erg s^-1 cm^-2 A^-1 arcsec^-2]."""
    fnu = 10.0 ** (-(mu_ab + 48.60) / 2.5)           # erg s^-1 cm^-2 Hz^-1 arcsec^-2
    return fnu * C_A / np.asarray(lam_A, float) ** 2


# ------------------------------------------------------------- zodiacal + thermal
def zodi_flambda(lam_A, mu_ref=22.5, lam_ref=5000.0, T_scat=5772.0,
                 tau_ipd=1.0e-7, T_ipd=265.0):
    """Zodiacal light surface brightness spectrum [erg s^-1 cm^-2 A^-1 arcsec^-2].

    Scattered-solar continuum is a solar blackbody (T_scat) normalised to
    mu_ref (AB mag/arcsec^2) at lam_ref, reddened by the ~0.3 mag in (V-K) that
    the interplanetary dust imposes relative to the Sun (Leinert et al. 1998),
    applied linearly in wavelength redward of V as in make_etc_data.py so the
    analytic and tabulated backgrounds agree.  The interplanetary-dust thermal
    term is tau_ipd * B_lambda(T_ipd) and is negligible below ~3 um.
    Typical mu_ref(0.5um): ~22.1 (ecliptic) to ~23.3 (pole)  [Leinert et al. 1998].
    """
    lam_A = np.asarray(lam_A, float)
    I_ref = ab_to_flambda(mu_ref, lam_ref)
    scat = I_ref * planck_lambda(lam_A, T_scat) / planck_lambda(lam_ref, T_scat)
    dmag = 0.30 * np.clip((lam_A / 1e4 - 0.55) / (2.2 - 0.55), 0.0, None)
    scat = scat * 10.0 ** (dmag / 2.5)
    therm = tau_ipd * planck_lambda(lam_A, T_ipd) * SR_PER_ARCSEC2
    return scat + therm


def telescope_flambda(lam_A, T_tel=270.0, emissivity=0.10):
    """Telescope thermal self-emission radiance [erg s^-1 cm^-2 A^-1 arcsec^-2]."""
    return emissivity * planck_lambda(lam_A, T_tel) * SR_PER_ARCSEC2


# Diffuse galactic light (cirrus) slope b = S_nu(optical)/S_nu(100um), measured
# by Ienaka et al. 2013 (ApJ 767, 80; MBM32) in B, g, V, R.  Extended flat in
# S_nu ratio outside the measured 0.44-0.65 um range (documented approximation;
# CIBER, Arai et al. 2015, finds the DGL continues smoothly into the near-IR).
_DGL_B = ([4400.0, 4900.0, 5500.0, 6500.0],
          [1.61e-3, 2.25e-3, 4.00e-3, 3.37e-3])


def cirrus_flambda(lam_A, I100_MJysr):
    """Galactic cirrus (diffuse galactic light) surface brightness
    [erg s^-1 cm^-2 A^-1 arcsec^-2] scaled from the 100 um intensity.

    S_nu(DGL) = b(lambda) * S_nu(100um) with b from Ienaka et al. 2013.
    """
    lam_A = np.asarray(lam_A, float)
    b = np.interp(lam_A, _DGL_B[0], _DGL_B[1])          # flat outside range
    fnu = b * I100_MJysr * 1e6 * 1e-23 * SR_PER_ARCSEC2  # erg/s/cm^2/Hz/arcsec^2
    return fnu * C_A / lam_A ** 2


def cirrus_I100_from_lat(gal_lat_deg, I100_pole=0.8):
    """100 um cirrus intensity [MJy/sr] from a plane-parallel cosecant law,
    normalised to the darkest high-latitude sky (~0.8 MJy/sr, Lockman Hole;
    Lockman, Jahoda & McCammon 1986).  For a real field use the measured
    IRAS/SFD map value instead via cirrus_I100_MJysr."""
    s = abs(np.sin(np.radians(gal_lat_deg)))
    return I100_pole / max(s, 0.10)          # cap the law below |b|~6 deg


def zodi_mu_from_ecl_lat(ecl_lat_deg):
    """Zodiacal reference brightness mu(0.5um) [AB/arcsec^2] versus ecliptic
    latitude at solar elongation ~90 deg, interpolated in sin|beta| between the
    two Leinert et al. 1998 anchors already used by this ETC, 22.1 in the
    ecliptic and 23.3 at the pole."""
    s = abs(np.sin(np.radians(ecl_lat_deg)))
    return 22.1 + (23.3 - 22.1) * s


# --- Galactic extinction, CCM89 (Cardelli, Clayton & Mathis 1989), R_V=3.1 ---
def extinction_A_lambda(lam_A, EBV, RV=3.1):
    """A(lambda) [mag] for the CCM89 mean law (IR + optical/NIR pieces,
    0.3 <= 1/lambda[um] <= 3.3)."""
    x = 1e4 / np.asarray(lam_A, float)            # 1/um
    a = np.empty_like(x); b = np.empty_like(x)
    ir = x < 1.1
    a[ir] = 0.574 * x[ir] ** 1.61
    b[ir] = -0.527 * x[ir] ** 1.61
    y = x[~ir] - 1.82
    a[~ir] = (1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4
              + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7)
    b[~ir] = (1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4
              - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7)
    return EBV * (RV * a + b)


# ---------------------------------------------------------------- instrument model
@dataclass
class InstrumentConfig:
    # --- telescope ---
    diameter_cm: float = 350.0          # 3.5 m primary
    obstruction: float = 0.15           # linear central-obstruction fraction
    # --- spectrograph / focal plane (R and pixel size are the key inputs) ---
    R: float = 1000.0                   # spectral resolving power lambda/dlambda
    pix_scale: float = 0.11             # arcsec / pixel  (spatial plate scale)
    res_element_pix: float = 2.0        # detector pixels sampling one resolution element
    read_noise: float = 8.0             # e- / pix per exposure (CDS-equivalent), NIR arm
    dark_current: float = 0.010         # e- / s / pix, NIR HgCdTe arm
    n_exp: int = 3                      # independent exposures (the three rolls)
    # --- two-arm focal plane: optical CCD arm below the dichroic split ---
    # (CCD273-class values, Euclid VIS heritage: read ~3.6 e-, dark <~1e-3 e-/s;
    #  Cropper et al. 2016.  dichroic_split_A=0 disables the split: single array.)
    dichroic_split_A: float = 10000.0   # optical arm at lambda < this [A]
    read_noise_opt: float = 3.6         # e- / pix per exposure, CCD arm
    dark_opt: float = 0.001             # e- / s / pix, CCD arm
    n_groups: int = 2                   # samples up the ramp per exposure (2 = plain CDS);
                                        # >2 applies the Rauscher et al. 2007 slope-fit gain
    # --- cosmic rays (JWST ETC convention: 8 events/s/cm^2, 9 pix/hit; JDox,
    #     measured L2 rate 2.3-4.3 ions/cm^2/s, Giardino et al. 2025) ---
    cr_rate: float = 8.0                # events / s / cm^2
    cr_pix_per_hit: float = 9.0         # pixels affected per hit
    pix_pitch_um: float = 18.0          # physical pixel pitch [um] (H2RG/H4RG)
    include_cr: bool = False            # fold CR losses into S/N explicitly
                                        # (default off: extraction_eff lump partly covers it)
    # --- throughput model (total: optics x disperser x QE) ---
    eta_peak: float = 0.30              # peak end-to-end efficiency (optics x grism x QE)
    band_min_A: float = 3600.0          # optical edge
    band_max_A: float = 30000.0         # near-IR edge
    edge_roll_A: float = 1500.0         # cosine roll-off width at band edges
    # --- backgrounds ---
    zodi_mu_ref: float = 22.1           # AB/arcsec^2 at 0.5 um (typical ecliptic; Leinert 1998)
    ecl_lat_deg: float | None = None    # set to derive zodi_mu_ref from ecliptic latitude
    tel_temp: float = 270.0             # K  (passively cooled optics)
    tel_emissivity: float = 0.10
    include_thermal: bool = True
    # --- galactic cirrus (diffuse galactic light; Ienaka et al. 2013 scaling) ---
    include_cirrus: bool = True
    gal_lat_deg: float = 60.0           # survey caps sit at |b| >~ 60 deg
    cirrus_I100_MJysr: float | None = None   # measured IRAS/SFD 100um value; None ->
                                             # cosecant law from gal_lat_deg
    extraction_eff: float = 0.70        # optimal-extraction aperture + contamination loss
    full_well: float = 100000.0         # detector full well [e-] (saturation)
    flat_error: float = 0.0             # flat-field residual fraction (bright-source floor; Pandeia-style)
    psf_floor: float = 0.04             # delivered image-quality floor [arcsec] (jitter+optics+charge diffusion)
    t_single: float = 1800.0            # single-exposure length [s]; read noise scales with
                                        # n_reads = max(n_exp, t/t_single) for long integrations
    # --- optional data files (lambda_A, value) override the analytic models ---
    throughput_csv: str = ""
    throughput_imaging_csv: str = ""    # direct-imaging path (no disperser in the beam)
    zodi_csv: str = ""

    @property
    def area_cm2(self) -> float:
        r = self.diameter_cm / 2.0
        return np.pi * r**2 * (1.0 - self.obstruction**2)

    @property
    def omega_pix(self) -> float:
        return self.pix_scale**2          # arcsec^2 / pixel

    def resolution_element_A(self, lam_A):
        """Wavelength width of one resolution element, delta_lambda = lambda / R."""
        return np.asarray(lam_A, float) / self.R

    def dispersion_A_per_pix(self, lam_A):
        """Dispersion along the trace, set by R and the pixel sampling."""
        return self.resolution_element_A(lam_A) / self.res_element_pix

    def psf_fwhm(self, lam_A):
        """Delivered PSF FWHM [arcsec]: diffraction FWHM of the centrally
        obscured Airy pattern (computed numerically from the aperture
        autocorrelation amplitude; 1.028 lambda/D at zero obscuration, slightly
        narrower for an obscured pupil) added in quadrature with the
        image-quality floor (jitter, optics, charge diffusion)."""
        lam_cm = np.asarray(lam_A, float) * 1e-8
        k = _airy_fwhm_coeff(self.obstruction)
        diff = k * lam_cm / self.diameter_cm * 206265.0
        return np.hypot(diff, self.psf_floor)

    # ---- throughput(lambda) ----
    def throughput(self, lam_A):
        lam_A = np.asarray(lam_A, float)
        if self.throughput_csv:
            t = np.loadtxt(self.throughput_csv, delimiter=",")
            return np.interp(lam_A, t[:, 0], t[:, 1], left=0.0, right=0.0)
        # analytic: flat plateau with cosine roll-offs at the band edges
        lo, hi, w = self.band_min_A, self.band_max_A, self.edge_roll_A
        eta = np.full_like(lam_A, self.eta_peak)
        left = lam_A < lo + w
        eta[left] *= 0.5 * (1 - np.cos(np.pi * np.clip((lam_A[left]-lo)/w, 0, 1)))
        right = lam_A > hi - w
        eta[right] *= 0.5 * (1 - np.cos(np.pi * np.clip((hi-lam_A[right])/w, 0, 1)))
        eta[(lam_A < lo) | (lam_A > hi)] = 0.0
        return eta

    def throughput_imaging(self, lam_A):
        """End-to-end throughput of the direct-imaging path (no grism in the
        beam).  Falls back to the spectroscopic curve when no imaging file is
        given (conservative)."""
        if self.throughput_imaging_csv:
            t = np.loadtxt(self.throughput_imaging_csv, delimiter=",")
            lam_A = np.asarray(lam_A, float)
            return np.interp(lam_A, t[:, 0], t[:, 1], left=0.0, right=0.0)
        return self.throughput(lam_A)

    def n_reads(self, t_s):
        """Number of detector reads contributing read noise: at least the n_exp
        rolls, growing as the integration is split into t_single exposures."""
        return max(float(self.n_exp), np.ceil(t_s / self.t_single))

    def detector_at(self, lam_A):
        """(read_noise, dark_current) of the arm that sees wavelength lam_A."""
        if self.dichroic_split_A > 0 and lam_A < self.dichroic_split_A:
            return self.read_noise_opt, self.dark_opt
        return self.read_noise, self.dark_current

    def read_noise_eff(self, lam_A):
        """Effective read noise per exposure [e-].  For n_groups=2 this is the
        plain CDS value.  For n_groups>2 the unweighted least-squares slope fit
        to n equally spaced samples gives sigma_eff = sigma_frame *
        sqrt(12(n-1)/(n(n+1))) with sigma_frame = CDS/sqrt(2)
        (Rauscher et al. 2007, PASP 119, 768)."""
        read, _ = self.detector_at(lam_A)
        n = max(2, int(self.n_groups))
        sigma_frame = read / np.sqrt(2.0)
        return sigma_frame * np.sqrt(12.0 * (n - 1) / (n * (n + 1.0)))

    def cr_exposure_efficiency(self):
        """Mean fraction of the accumulated signal that survives cosmic-ray
        hits in one t_single exposure.  A pixel is hit with probability
        f = rate * pix_per_hit * pitch^2 * t_single; with up-the-ramp fitting
        the slope before the hit is recovered, so a hit pixel keeps on average
        half its integration (uniform hit times), giving efficiency 1 - f/2.
        Rate and hit footprint follow the JWST ETC convention (JDox)."""
        a_pix = (self.pix_pitch_um * 1e-4) ** 2                  # cm^2
        f = min(1.0, self.cr_rate * self.cr_pix_per_hit * a_pix * self.t_single)
        return 1.0 - 0.5 * f

    # ---- background radiance spectrum (per arcsec^2) ----
    def sky_flambda(self, lam_A):
        mu_ref = (zodi_mu_from_ecl_lat(self.ecl_lat_deg)
                  if self.ecl_lat_deg is not None else self.zodi_mu_ref)
        if self.zodi_csv:
            z = np.loadtxt(self.zodi_csv, delimiter=",")
            sky = np.interp(lam_A, z[:, 0], z[:, 1])
            if self.ecl_lat_deg is not None:      # rescale the tabulated zodi
                sky = sky * 10.0 ** (-0.4 * (mu_ref - 22.1))
        else:
            sky = zodi_flambda(lam_A, mu_ref=mu_ref)
        if self.include_cirrus:
            I100 = (self.cirrus_I100_MJysr if self.cirrus_I100_MJysr is not None
                    else cirrus_I100_from_lat(self.gal_lat_deg))
            sky = sky + cirrus_flambda(lam_A, I100)
        if self.include_thermal:
            sky = sky + telescope_flambda(lam_A, self.tel_temp, self.tel_emissivity)
        return sky


# --------------------------------------------------- obscured-Airy diffraction
def _airy_intensity(v, eps):
    """Normalised intensity of an annular (obscured) aperture at reduced radius
    v = pi D theta / lambda:  U = [2J1(v)/v - eps^2 2J1(eps v)/(eps v)]/(1-eps^2)."""
    from scipy.special import j1
    v = np.asarray(v, float)
    small = v < 1e-9
    vv = np.where(small, 1e-9, v)
    U = 2.0 * j1(vv) / vv
    if eps > 0:
        U = (U - eps ** 2 * 2.0 * j1(eps * vv) / (eps * vv)) / (1.0 - eps ** 2)
    return np.where(small, 1.0, U) ** 2


_AIRY_CACHE = {}


def _airy_fwhm_coeff(eps):
    """FWHM of the obscured Airy core in units of lambda/D (1.0290 at eps=0)."""
    key = round(float(eps), 3)
    if key not in _AIRY_CACHE:
        v = np.linspace(1e-4, 4.0, 4000)
        I = _airy_intensity(v, key)
        v_half = np.interp(0.5, I[::-1], v[::-1])       # I falls monotonic in core
        _AIRY_CACHE[key] = 2.0 * v_half / np.pi
    return _AIRY_CACHE[key]


def encircled_energy(cfg, lam_A, r_arcsec):
    """Fraction of a point source's diffraction PSF (obscured Airy) inside
    radius r_arcsec, by numerical integration of the radial profile."""
    lam_cm = float(lam_A) * 1e-8
    v_of_r = np.pi * cfg.diameter_cm / lam_cm * (r_arcsec / 206265.0)
    v = np.linspace(1e-4, max(v_of_r, 60.0), 60000)
    I = _airy_intensity(v, cfg.obstruction)
    cum = np.cumsum(I * v)
    return float(np.interp(v_of_r, v, cum) / cum[-1])


def aperture_ee(cfg, lam_A, r_arcsec):
    """Encircled energy of the delivered PSF in a photometric aperture: the
    obscured-Airy diffraction EE times the EE of the Gaussian image-quality
    floor (jitter, optics, charge diffusion), treated as independent losses."""
    ee = encircled_energy(cfg, lam_A, r_arcsec)
    if cfg.psf_floor > 0:
        sig = cfg.psf_floor / 2.35482
        ee *= 1.0 - float(np.exp(-0.5 * (r_arcsec / sig) ** 2))
    return ee


# ---------------------------------------------------------------- ETC core
def _photon_factor(lam_A):
    """photons per erg for a photon of wavelength lam_A (= lambda/hc)."""
    return np.asarray(lam_A, float) * 1e-8 / (H * C)


def background_per_pixel(cfg: InstrumentConfig, band_A, imaging=False):
    """Slitless diffuse background collected per pixel per second [e-/s/pix].

    Uniform sky disperses onto every pixel across the whole band-limiting filter,
    so integrate radiance * throughput * (lambda/hc) over the transmitted band.
    With imaging=True the direct-imaging throughput (no grism) is used.
    """
    lo, hi = band_A
    lam = np.linspace(lo, hi, 400)
    I = cfg.sky_flambda(lam)                     # erg/s/cm^2/A/arcsec^2
    eta = cfg.throughput_imaging(lam) if imaging else cfg.throughput(lam)
    integrand = I * _photon_factor(lam) * eta    # e-/s/cm^2/A/arcsec^2
    surf = np.trapezoid(integrand, lam)              # e-/s/cm^2/arcsec^2
    return surf * cfg.area_cm2 * cfg.omega_pix   # e-/s/pixel


def line_footprint_pixels(cfg, source_fwhm_arcsec, lam_A):
    """Number of detector pixels a line image covers (spatial x spectral).

    The source is convolved with the diffraction PSF; along the dispersion axis
    an unresolved line spans one resolution element plus the morphological width.
    """
    theta = np.hypot(source_fwhm_arcsec, cfg.psf_fwhm(lam_A))   # observed size
    spatial = theta / cfg.pix_scale
    spectral = np.hypot(cfg.res_element_pix, theta / cfg.pix_scale)
    return max(1.0, spatial) * max(cfg.res_element_pix, spectral)


def continuum_e_per_s(cfg, cont_mag_AB, lam_A, source_fwhm=0.3):
    """Source-continuum electrons/s falling under the line footprint.

    This is the one term set by R: the continuum piled under the line spans one
    resolution element (delta_lambda = lambda/R), so higher R buries less
    continuum under the line and improves a continuum-limited line detection.
    """
    if cont_mag_AB is None:
        return 0.0
    flam = ab_to_flambda(cont_mag_AB, lam_A)               # erg/s/cm^2/A (point source)
    theta = np.hypot(source_fwhm, cfg.psf_fwhm(lam_A))
    # continuum under the line footprint spans the *spectral* extent of the
    # footprint (resolution element broadened by the source size), independent
    # of how many spatial pixels the flux is spread over (flux conservation)
    spectral_pix = np.hypot(cfg.res_element_pix, theta / cfg.pix_scale)
    dlam_eff = spectral_pix * cfg.dispersion_A_per_pix(lam_A)
    return (flam * cfg.area_cm2 * cfg.throughput(lam_A)
            * _photon_factor(lam_A) * dlam_eff * cfg.extraction_eff)


def line_sn(cfg: InstrumentConfig, F_line, lam_A, t_s, source_fwhm=0.3,
            filter_width_A=4000.0, cont_mag_AB=None):
    """S/N of an emission line of flux F_line [erg/s/cm^2] at lam_A in t_s seconds."""
    band = (max(cfg.band_min_A, lam_A - filter_width_A/2),
            min(cfg.band_max_A, lam_A + filter_width_A/2))
    teff = t_s * (cfg.cr_exposure_efficiency() if cfg.include_cr else 1.0)
    S = (F_line * cfg.area_cm2 * cfg.throughput(lam_A) * _photon_factor(lam_A)
         * teff * cfg.extraction_eff)
    Bpix = background_per_pixel(cfg, band)
    npix = line_footprint_pixels(cfg, source_fwhm, lam_A)
    Ccont = continuum_e_per_s(cfg, cont_mag_AB, lam_A, source_fwhm)
    _, dark = cfg.detector_at(lam_A)
    var = (S
           + (Bpix + dark) * npix * teff
           + Ccont * teff
           + npix * cfg.n_reads(t_s) * cfg.read_noise_eff(lam_A)**2
           + (cfg.flat_error * S)**2)
    return float(S / np.sqrt(var))


def f_limit(cfg: InstrumentConfig, lam_A, t_s, snr=5.0, source_fwhm=0.3,
            filter_width_A=4000.0, cont_mag_AB=None):
    """Line flux [erg/s/cm^2] detected at S/N = snr in t_s seconds.

    Solves snr = S/sqrt(S + B_tot + (flat*S)^2) exactly (quadratic in F, source
    shot noise and the flat-field floor included, consistent with line_sn).
    """
    band = (max(cfg.band_min_A, lam_A - filter_width_A/2),
            min(cfg.band_max_A, lam_A + filter_width_A/2))
    teff = t_s * (cfg.cr_exposure_efficiency() if cfg.include_cr else 1.0)
    Bpix = background_per_pixel(cfg, band)
    npix = line_footprint_pixels(cfg, source_fwhm, lam_A)
    Ccont = continuum_e_per_s(cfg, cont_mag_AB, lam_A, source_fwhm)
    _, dark = cfg.detector_at(lam_A)
    Btot = ((Bpix + dark) * npix * teff + Ccont * teff
            + npix * cfg.n_reads(t_s) * cfg.read_noise_eff(lam_A)**2)
    k = (cfg.area_cm2 * cfg.throughput(lam_A) * _photon_factor(lam_A)
         * teff * cfg.extraction_eff)   # e- per unit flux
    a = 1.0 - (snr * cfg.flat_error) ** 2          # =1 when flat_error=0
    if a <= 0:
        return float("inf")                        # flat floor caps S/N below snr
    S = (snr**2 + np.sqrt(snr**4 + 4.0 * a * snr**2 * Btot)) / (2.0 * a)     # required e-
    return float(S / k)


def imaging_maglimit(cfg: InstrumentConfig, lam_A, filter_width_A, t_s,
                     snr=5.0, aper_fwhm_mult=1.0):
    """Broadband imaging 5-sigma point-source limiting AB magnitude.

    Same detector and background machinery as the spectroscopic depth, but the
    source is a point source whose whole in-band flux lands in a photometric
    aperture of radius ~ aper_fwhm_mult * PSF FWHM, and the background is the sky
    over the imaging filter bandwidth.
    """
    band = (lam_A - filter_width_A/2, lam_A + filter_width_A/2)
    teff = t_s * (cfg.cr_exposure_efficiency() if cfg.include_cr else 1.0)
    Bpix = background_per_pixel(cfg, band, imaging=True)
    r_ap = max(aper_fwhm_mult * cfg.psf_fwhm(lam_A), 1.5 * cfg.pix_scale)          # aperture radius [arcsec]
    npix = np.pi * (r_ap / cfg.pix_scale)**2
    _, dark = cfg.detector_at(lam_A)
    Btot = (Bpix + dark) * npix * teff + npix * cfg.n_reads(t_s) * cfg.read_noise_eff(lam_A)**2
    # electrons per unit F_lambda [erg/s/cm^2/A] collected in the aperture;
    # the aperture loss is the physical encircled energy of the delivered PSF
    k = (cfg.area_cm2 * cfg.throughput_imaging(lam_A) * _photon_factor(lam_A)
         * teff * filter_width_A * aperture_ee(cfg, lam_A, r_ap))
    S = 0.5 * (snr**2 + np.sqrt(snr**4 + 4.0 * snr**2 * Btot))   # required e-
    F_lambda = S / k                                     # erg/s/cm^2/A
    F_nu = F_lambda * lam_A**2 / C_A                      # erg/s/cm^2/Hz
    return -2.5 * np.log10(F_nu) - 48.60                  # AB mag


# Zodiacal pointing levels (Leinert et al. 1998): AB mu at 0.5 um
ZODI_LEVELS = {"low (ecliptic pole)": 23.3, "typical": 22.1, "high (near ecliptic)": 21.0}


# Telescope presets with real detector specs (read/dark/full-well/pixel/cutoff).
# Sources: Roman WFI technical page (H4RG-10, effective noise ~6 e-, dark ~0.02,
# QE 0.89, 0.11"/pix, 2.5um cutoff); JWST NIRCam jdox (H2RG: read ~6 e-, SW dark
# ~0.002 / LW ~0.034 e-/s, SW 0.031" / LW 0.063" pix); Euclid-III NISP H2RG
# (read ~7 e-, dark ~0.02, 0.30"/pix, 2.3um). Read noise is the effective value.
TELESCOPE_PRESETS = {
    "3.5mST":         dict(det="HgCdTe (concept)", diam=350., obstruction=0.15,  pix=0.11,  eta=0.30,
                           read=8., dark=0.010, nexp=3, fw=100000., ttel=270., R=1000., band=(0.36, 3.00), lam=1.6, split=10000., realistic=True),
    "Roman WFI":      dict(det="H4RG-10", diam=240., obstruction=0.31,  pix=0.11,  eta=0.42,
                           read=6., dark=0.020, nexp=4, fw=100000., ttel=270., R=461.,  band=(0.48, 2.30), lam=1.5, split=0., realistic=False),
    "Euclid":         dict(det="H2RG (NISP)", diam=120., obstruction=0.40,  pix=0.30,  eta=0.30,
                           read=7., dark=0.020, nexp=4, fw=80000.,  ttel=140., R=450.,  band=(0.90, 2.00), lam=1.5, split=0., realistic=False),
    "JWST NIRCam SW": dict(det="H2RG (short-wave)", diam=650., obstruction=0.485, pix=0.031, eta=0.45,
                           read=6., dark=0.002, nexp=4, fw=80000.,  ttel=45.,  R=1000., band=(0.60, 2.35), lam=1.5, split=0., realistic=False),
    "JWST NIRCam LW": dict(det="H2RG (long-wave)", diam=650., obstruction=0.485, pix=0.063, eta=0.45,
                           read=6., dark=0.034, nexp=4, fw=80000.,  ttel=45.,  R=1600., band=(2.40, 5.00), lam=3.5, split=0., realistic=False),
}

# Filters (imaging) and dispersers (spectroscopy) per telescope.
# Imaging element -> (central lambda [um], width [um]); spectral -> (lam, width, R).
INSTRUMENT_ELEMENTS = {
    "3.5mST": {"Imaging": {"g~0.5": (0.50, 0.15), "z~0.9": (0.90, 0.20), "J~1.2": (1.25, 0.30),
                           "H~1.6": (1.60, 0.40), "K~2.2": (2.20, 0.50)},
               "Spectroscopy": {"R1000 opt": (0.70, 0.40, 1000), "R1000 NIR": (1.60, 0.40, 1000),
                                "R1000 K": (2.50, 0.50, 1000)}},
    "Roman WFI": {"Imaging": {"F062": (0.62, 0.28), "F087": (0.87, 0.22), "F106": (1.06, 0.27),
                              "F129": (1.29, 0.31), "F158": (1.58, 0.40), "F184": (1.84, 0.32),
                              "F213": (2.13, 0.35), "F146 (wide)": (1.46, 1.03)},
                  "Spectroscopy": {"Grism": (1.46, 0.93, 600), "Prism": (1.28, 1.05, 130)}},
    "Euclid": {"Imaging": {"Y": (1.085, 0.26), "J": (1.375, 0.39), "H": (1.772, 0.50)},
               "Spectroscopy": {"Red grism": (1.55, 0.60, 450), "Blue grism": (1.09, 0.32, 380)}},
    "JWST NIRCam SW": {"Imaging": {"F070W": (0.704, 0.128), "F090W": (0.902, 0.194),
                                   "F115W": (1.154, 0.225), "F150W": (1.501, 0.318), "F200W": (1.990, 0.457)},
                       "Spectroscopy": {"NIRSpec G140M": (1.25, 0.60, 1000), "NIRSpec PRISM": (2.00, 3.40, 100)}},
    "JWST NIRCam LW": {"Imaging": {"F277W": (2.786, 0.672), "F356W": (3.563, 0.787), "F444W": (4.421, 1.024)},
                       "Spectroscopy": {"LW grism": (4.00, 1.90, 1600), "NIRSpec G395M": (3.60, 1.60, 1000)}},
}


def imaging_snr(cfg, mag_ab, lam_A, filter_width_A, t_s, aper_fwhm_mult=1.0):
    """Broadband imaging S/N for a point source of AB magnitude mag_ab."""
    band = (lam_A - filter_width_A/2, lam_A + filter_width_A/2)
    teff = t_s * (cfg.cr_exposure_efficiency() if cfg.include_cr else 1.0)
    Bpix = background_per_pixel(cfg, band, imaging=True)
    r_ap = max(aper_fwhm_mult * cfg.psf_fwhm(lam_A), 1.5 * cfg.pix_scale)
    npix = np.pi * (r_ap / cfg.pix_scale)**2
    Flam = ab_to_flambda(mag_ab, lam_A)
    S = (cfg.area_cm2 * cfg.throughput_imaging(lam_A) * _photon_factor(lam_A)
         * teff * filter_width_A * Flam * aperture_ee(cfg, lam_A, r_ap))
    _, dark = cfg.detector_at(lam_A)
    var = (S + (Bpix + dark) * npix * teff
           + npix * cfg.n_reads(t_s) * cfg.read_noise_eff(lam_A)**2 + (cfg.flat_error * S)**2)
    return float(S / np.sqrt(var))


def exposure_for_snr(snr_func, target_snr, tlo=1.0, thi=1.0e7):
    """Invert a monotonically increasing snr_func(t_s) for the exposure [s]."""
    for _ in range(60):
        mid = np.sqrt(tlo * thi)
        if snr_func(mid) < target_snr:
            tlo = mid
        else:
            thi = mid
    return np.sqrt(tlo * thi)


def peak_fraction(cfg, lam_A):
    """Fraction of a point source's flux landing in the central pixel: exact
    integral of a centred Gaussian PSF over the pixel, erf(p/(2 sqrt2 sigma))^2
    (the peak-intensity-times-area approximation overestimates this by ~40%
    when the pixel is comparable to the PSF, as here)."""
    from math import erf
    sigma = cfg.psf_fwhm(lam_A) / 2.35482
    a = cfg.pix_scale / (2.0 * np.sqrt(2.0) * sigma)
    return float(erf(a) ** 2)


def saturation_maglimit(cfg, lam_A, filter_width_A, t_single):
    """Brightest imaging AB magnitude before the central pixel saturates in one
    exposure of t_single seconds (full-well limit)."""
    band = (lam_A - filter_width_A/2, lam_A + filter_width_A/2)
    Bpix = background_per_pixel(cfg, band, imaging=True)
    avail = cfg.full_well / t_single - Bpix - cfg.detector_at(lam_A)[1]   # e-/s left for source peak
    if avail <= 0:
        return np.nan
    Srate = avail / peak_fraction(cfg, lam_A)                       # total source e-/s
    Flam = Srate / (cfg.area_cm2 * cfg.throughput_imaging(lam_A)
                    * _photon_factor(lam_A) * filter_width_A)
    return float(-2.5 * np.log10(Flam * lam_A**2 / C_A) - 48.60)


def count_rates(cfg, mag_ab, lam_A, filter_width_A, aper_fwhm_mult=1.0):
    """Source and background electron rates for an imaging point source [e-/s]."""
    band = (lam_A - filter_width_A/2, lam_A + filter_width_A/2)
    Bpix = background_per_pixel(cfg, band, imaging=True)
    r_ap = max(aper_fwhm_mult * cfg.psf_fwhm(lam_A), 1.5 * cfg.pix_scale)
    npix = np.pi * (r_ap / cfg.pix_scale)**2
    Flam = ab_to_flambda(mag_ab, lam_A)
    Srate = (cfg.area_cm2 * cfg.throughput_imaging(lam_A) * _photon_factor(lam_A)
             * filter_width_A * Flam * aperture_ee(cfg, lam_A, r_ap))
    return {"source_e_s": Srate, "sky_e_s_pix": Bpix, "n_pix": npix,
            "dark_e_s_pix": cfg.detector_at(lam_A)[1],
            "peak_e_s": Srate*peak_fraction(cfg, lam_A)}


def noise_breakdown(cfg, lam_A, t_s, source_fwhm=0.3, filter_width_A=4000.0):
    band = (max(cfg.band_min_A, lam_A - filter_width_A/2),
            min(cfg.band_max_A, lam_A + filter_width_A/2))
    Bpix = background_per_pixel(cfg, band)
    npix = line_footprint_pixels(cfg, source_fwhm, lam_A)
    _, dark = cfg.detector_at(lam_A)
    return {
        "Bpix_e_per_s": Bpix,
        "n_pix": npix,
        "sky_e": Bpix * npix * t_s,
        "dark_e": dark * npix * t_s,
        "read_e2": npix * cfg.n_reads(t_s) * cfg.read_noise_eff(lam_A)**2,
    }


# ------------------------------------------------- redshift precision estimator
def sigma_z(cfg, F_line, lam_obs_A, t_s, z, source_fwhm=0.3,
            filter_width_A=4000.0, wavecal_A=0.0, **kw):
    """Redshift uncertainty of a single emission line: Gaussian centroid
    statistics sigma_lambda = (FWHM_eff/2.355)/(S/N) added in quadrature with a
    wavelength-calibration floor.  FWHM_eff is the resolution element broadened
    by the source size (the slitless effective resolution)."""
    sn = line_sn(cfg, F_line, lam_obs_A, t_s, source_fwhm=source_fwhm,
                 filter_width_A=filter_width_A, **kw)
    if sn <= 0:
        return np.inf
    theta = np.hypot(source_fwhm, cfg.psf_fwhm(lam_obs_A))
    fwhm_A = np.hypot(cfg.res_element_pix, theta / cfg.pix_scale) \
        * cfg.dispersion_A_per_pix(lam_obs_A)
    sig_lam = np.hypot(fwhm_A / 2.35482 / sn, wavecal_A)
    return float(sig_lam * (1.0 + z) / lam_obs_A)


# ------------------------------------------------- spectroscopic saturation
def saturation_contmag_spectro(cfg, lam_A, t_single=None):
    """Brightest continuum point source [AB mag] before its spectral trace
    saturates the full well in one exposure.  The per-pixel rate is the
    continuum in one pixel's dispersion width times the central-pixel spatial
    fraction, on top of sky and dark."""
    t1 = cfg.t_single if t_single is None else t_single
    band = (max(cfg.band_min_A, lam_A - 2000.0), min(cfg.band_max_A, lam_A + 2000.0))
    Bpix = background_per_pixel(cfg, band)
    avail = cfg.full_well / t1 - Bpix - cfg.detector_at(lam_A)[1]
    if avail <= 0:
        return np.nan
    from math import erf
    sigma = cfg.psf_fwhm(lam_A) / 2.35482
    frac_spatial = erf(cfg.pix_scale / (2.0 * np.sqrt(2.0) * sigma))   # 1-D central pixel
    Flam = avail / (cfg.area_cm2 * cfg.throughput(lam_A) * _photon_factor(lam_A)
                    * cfg.dispersion_A_per_pix(lam_A) * frac_spatial)
    return float(-2.5 * np.log10(Flam * lam_A**2 / C_A) - 48.60)


def saturation_lineflux_spectro(cfg, lam_A, t_single=None):
    """Brightest unresolved emission line [erg/s/cm^2] before its brightest
    pixel saturates in one exposure (line spread over the resolution element
    along the dispersion and the PSF across it)."""
    t1 = cfg.t_single if t_single is None else t_single
    band = (max(cfg.band_min_A, lam_A - 2000.0), min(cfg.band_max_A, lam_A + 2000.0))
    Bpix = background_per_pixel(cfg, band)
    avail = cfg.full_well / t1 - Bpix - cfg.detector_at(lam_A)[1]
    if avail <= 0:
        return np.nan
    from math import erf
    sigma = cfg.psf_fwhm(lam_A) / 2.35482
    frac_spatial = erf(cfg.pix_scale / (2.0 * np.sqrt(2.0) * sigma))
    frac_spectral = 1.0 / cfg.res_element_pix          # line spread over the res element
    rate_per_flux = (cfg.area_cm2 * cfg.throughput(lam_A) * _photon_factor(lam_A)
                     * frac_spatial * frac_spectral)   # peak-pixel e-/s per unit line flux
    return float(avail / rate_per_flux)


# ------------------------------------------------- [O II] doublet at R=1000
OII_SEP_REST_A = 2.783        # 3728.815 - 3726.032 (vacuum rest separation)


def oii_effective_fwhm_A(cfg, z):
    """Effective FWHM [A, observed] of the [O II] doublet through the
    spectrograph: two equal Gaussian components at the instrumental resolution,
    separated by the redshifted doublet splitting, measured numerically.
    The doublet is split only for R > ~1340 (3727/2.78)."""
    lam_obs = 3727.4 * (1.0 + z)
    res_A = lam_obs / cfg.R
    sep = OII_SEP_REST_A * (1.0 + z)
    x = np.linspace(-6 * res_A, 6 * res_A + sep, 4000)
    sig = res_A / 2.35482
    prof = np.exp(-0.5 * (x / sig) ** 2) + np.exp(-0.5 * ((x - sep) / sig) ** 2)
    prof /= prof.max()
    above = x[prof >= 0.5]
    return float(above.max() - above.min())


# ------------------------------------------------- slitless self-contamination
def trace_covering_fraction(cfg, density_arcmin2, filter_width_A, lam_A=16000.0,
                            width_pix=3.0):
    """Fraction of the detector covered by dispersed traces, and so the
    probability that a random source's own trace is overlapped (Poisson
    covering, 1 - exp(-N A_trace)).

    density_arcmin2 must come from a real catalog of sources bright enough to
    matter at the survey depth (e.g. the 3D-HST photometric catalogs, Skelton
    et al. 2014); no default is assumed here.
    """
    trace_len_pix = filter_width_A / cfg.dispersion_A_per_pix(lam_A)
    area_pix2 = trace_len_pix * width_pix
    area_arcsec2 = area_pix2 * cfg.omega_pix
    n_per_arcsec2 = density_arcmin2 / 3600.0
    return float(1.0 - np.exp(-n_per_arcsec2 * area_arcsec2))


def effective_extraction(cfg, lam_A, r_ap_arcsec=None, contam_frac=0.0):
    """Physically decomposed extraction efficiency: diffraction encircled
    energy inside the extraction aperture times the uncontaminated fraction.
    An alternative to the lumped extraction_eff=0.70."""
    r = r_ap_arcsec if r_ap_arcsec is not None else cfg.psf_fwhm(lam_A)
    return encircled_energy(cfg, lam_A, r) * (1.0 - contam_frac)


# ------------------------------------------------- Monte-Carlo completeness
def completeness_curve(cfg, lam_A, t_s, F_grid=None, snr_thresh=5.0, n_mc=4000,
                       source_fwhm=0.3, filter_width_A=4000.0, contam_frac=0.0,
                       seed=12345):
    """Inject-recover completeness on the ETC noise model: for each line flux,
    draw n_mc Gaussian realisations of the measured S/N and count the fraction
    exceeding snr_thresh, multiplied by the uncontaminated fraction.
    Returns (F_grid, completeness)."""
    rng = np.random.default_rng(seed)
    if F_grid is None:
        f5 = f_limit(cfg, lam_A, t_s, snr=snr_thresh, source_fwhm=source_fwhm,
                     filter_width_A=filter_width_A)
        F_grid = f5 * np.logspace(-0.6, 0.6, 25)
    comp = []
    for F in np.asarray(F_grid, float):
        sn = line_sn(cfg, F, lam_A, t_s, source_fwhm=source_fwhm,
                     filter_width_A=filter_width_A)
        meas = sn + rng.standard_normal(n_mc)          # unit-variance S/N estimator
        comp.append((meas >= snr_thresh).mean() * (1.0 - contam_frac))
    return np.asarray(F_grid, float), np.asarray(comp)


# ---------------------------------------------------------------- mission presets
# Published references used for validation:
#   Roman HLSS : Wang et al. 2022, ApJ 928, 1  (arXiv:2110.01829)
#                1.0e-16 erg/s/cm^2 at 6.5 sigma; grism 1.0-1.93 um, R=461*lam[um],
#                11 A/pix, 2.4 m, 0.11"/pix.
#   Euclid     : Euclid prep. XXX 2023, A&A 676, A34 (arXiv:2302.09372)
#                2.0e-16 erg/s/cm^2 at 3.5 sigma for a 0.5" source; red grism
#                1.25-1.85 um, R~450, 1.2 m, 0.3"/pix; Wide = 4 x 560 s.
def roman_cfg(lam_A):
    # detector values match the "Roman WFI" entry of TELESCOPE_PRESETS
    # (H4RG-10: effective read ~6 e-, dark ~0.02 e-/s; Roman WFI technical page)
    return InstrumentConfig(
        diameter_cm=240.0, obstruction=0.31, pix_scale=0.11,
        R=461.0 * (lam_A / 1e4), res_element_pix=2.0,
        read_noise=6.0, dark_current=0.020, n_exp=4, dichroic_split_A=0.0,
        eta_peak=0.32, band_min_A=10000.0, band_max_A=19300.0, edge_roll_A=800.0,
        zodi_mu_ref=22.1, extraction_eff=1.0)   # eta is already an effective throughput

def euclid_cfg(lam_A=16000.0):
    return InstrumentConfig(
        diameter_cm=120.0, obstruction=0.40, pix_scale=0.30,
        R=450.0, res_element_pix=2.0,
        read_noise=6.0, dark_current=0.020, n_exp=4, dichroic_split_A=0.0,
        eta_peak=0.25, band_min_A=12500.0, band_max_A=18500.0, edge_roll_A=600.0,
        zodi_mu_ref=22.1, extraction_eff=1.0)


def exposure_for_flux(cfg, lam_A, F_target, snr, source_fwhm, filter_width_A):
    """Exposure [s] at which f_limit(...) equals F_target (bisection)."""
    lo, hi = 1.0, 1e7
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        f = f_limit(cfg, lam_A, mid, snr=snr, source_fwhm=source_fwhm,
                    filter_width_A=filter_width_A)
        if f > F_target:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


def compare_missions():
    """Cross-check the ETC against published Roman and Euclid grism depths.

    The published numbers are *realised survey* limits; they fold in pipeline
    extraction losses, self-contamination and margin that an idealised photon
    ETC omits, so this ETC is expected to sit a little deeper (optimistic).
    """
    lam = 16000.0
    print("\n" + "=" * 72)
    print("CROSS-CHECK vs published Roman / Euclid grism sensitivities")
    print("=" * 72)

    cR = roman_cfg(lam); bandR = cR.band_max_A - cR.band_min_A
    tR = 4 * 250.0
    fR = f_limit(cR, lam, tR, snr=6.5, source_fwhm=0.3, filter_width_A=bandR)
    tR_pub = exposure_for_flux(cR, lam, 1.0e-16, 6.5, 0.3, bandR)
    print("Roman HLSS  (2.4 m, 0.11\", R=461*lam, grism 1.0-1.93um):")
    print("    published depth        : 1.0e-16 erg/s/cm^2 @ 6.5 sigma  (Wang+2022)")
    print(f"    ETC depth  @ {tR:.0f}s     : {fR:.2e}   (ratio ETC/pub = {fR/1e-16:.2f})")
    print(f"    ETC exposure for 1e-16 : {tR_pub:.0f} s  (HLSS field ~1000 s)")

    cE = euclid_cfg(); bandE = cE.band_max_A - cE.band_min_A
    tE = 4 * 560.0
    fE = f_limit(cE, lam, tE, snr=3.5, source_fwhm=0.5, filter_width_A=bandE)
    print("Euclid Wide (1.2 m, 0.3\", R~450, red grism 1.25-1.85um):")
    print("    published depth        : 2.0e-16 erg/s/cm^2 @ 3.5 sigma, 0.5\" src  (Euclid prep.XXX)")
    print(f"    ETC depth  @ {tE:.0f}s     : {fE:.2e}   (ratio ETC/pub = {fE/2e-16:.2f})")
    print("=" * 72)
    print("The idealised photon ETC sits deeper than both realised survey limits,")
    print("~1.4x for Roman and ~2.3x for Euclid, the expected optimistic offset from")
    print("pipeline extraction losses and wide-survey self-contamination that the")
    print("published limits fold in but a photon ETC does not (larger for Euclid's")
    print("coarse 0.3\" sampling).")


def realistic_cfg(**kw):
    """3.5 m concept using the tabulated CALSPEC-solar zodi and component
    throughput/QE curves written by make_etc_data.py."""
    import os
    base = dict(extraction_eff=0.70,
                throughput_csv="etc_throughput.csv" if os.path.exists("etc_throughput.csv") else "",
                throughput_imaging_csv=("etc_throughput_imaging.csv"
                                        if os.path.exists("etc_throughput_imaging.csv") else ""),
                zodi_csv="etc_zodi.csv" if os.path.exists("etc_zodi.csv") else "")
    base.update(kw)
    return InstrumentConfig(**base)


def cooling_tradeoff(temps=(150, 180, 210, 240, 270, 290), t_s=3*3600.0):
    """Near-IR depth vs telescope temperature: where does thermal self-emission
    overtake the zodiacal background and build a 'thermal wall'?"""
    cfg0 = realistic_cfg()
    lam = np.linspace(10000., 30000., 220)
    zodi = replace(cfg0, include_thermal=False).sky_flambda(lam)   # zodiacal only
    print("\n" + "=" * 66)
    print("TELESCOPE COOLING TRADEOFF (thermal self-emission vs zodiacal)")
    print("=" * 66)
    print(f"emissivity={cfg0.tel_emissivity}, F5sigma at {t_s/3600:.0f} hr for a 0.3\" source")
    print(f"{'T_tel[K]':>8}{'thermal>zodi from':>19}{'F5s 1.6um':>11}"
          f"{'F5s 2.2um':>11}{'F5s 2.7um':>11}")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(temps)))
    for T, c in zip(temps, colors):
        cfg = replace(cfg0, tel_temp=T)
        thr = telescope_flambda(lam, T, cfg.tel_emissivity)
        over = lam[thr > zodi]
        cross = over.min()/1e4 if over.size else np.inf
        f = np.array([f_limit(cfg, l, t_s, source_fwhm=0.3, filter_width_A=4000.)
                      for l in lam])
        ax.plot(lam/1e4, f, color=c, lw=2, label=f"{T} K")
        g = lambda um: f_limit(cfg, um*1e4, t_s, source_fwhm=0.3, filter_width_A=4000.)
        cs = f"{cross:.2f} um" if np.isfinite(cross) else "none"
        print(f"{T:>8}{cs:>19}{g(1.6):>11.1e}{g(2.2):>11.1e}{g(2.7):>11.1e}")
    ax.set_yscale("log"); ax.set_xlabel(r"observed wavelength [$\mu$m]")
    ax.set_ylabel(r"$F_{5\sigma}$ [erg s$^{-1}$ cm$^{-2}$]  (3 hr, 0.3\")")
    ax.set_title("Telescope-temperature tradeoff: the near-IR thermal wall")
    ax.legend(title="optics T", fontsize=8, ncol=2); ax.grid(alpha=0.2, which="both")
    fig.savefig("etc_cooling.png", bbox_inches="tight"); print("wrote etc_cooling.png")
    print("=" * 66)


# ---------------------------------------------------------------- demo / report
def main(cfg=None):
    if cfg is None:
        cfg = InstrumentConfig()
    hr = 3600.0
    print(f"Effective collecting area : {cfg.area_cm2:.3e} cm^2 "
          f"({cfg.diameter_cm/100:.1f} m, obstruction {cfg.obstruction})")
    print(f"Zodi reference            : mu(0.5um) = {cfg.zodi_mu_ref} AB/arcsec^2 "
          f"(Leinert et al. 1998)")
    print(f"Pixel / read / dark       : {cfg.pix_scale}\"  {cfg.read_noise} e-  "
          f"{cfg.dark_current} e-/s ;  {cfg.n_exp} rolls")
    print(f"Resolving power R          : {cfg.R:.0f}  ->  dlambda(1.6um) = "
          f"{cfg.resolution_element_A(16000.):.1f} A, dispersion = "
          f"{cfg.dispersion_A_per_pix(16000.):.1f} A/pix\n")

    lines = {"[OII] 3727": 3727., "Hbeta 4861": 4861., "[OIII] 5007": 5007.,
             "Halpha 6563": 6563.}
    # representative observed wavelengths at a working redshift z~1.5
    z = 1.5
    print(f"F_5sigma [erg/s/cm^2] for a 0.3\" ELG, filter width 0.4 um, "
          f"lines redshifted to z={z}:")
    header = "  line            lam_obs   " + "".join(
        f"{tt/hr:>7.2f}h" for tt in (0.75*hr, 3*hr, 12*hr, 48*hr))
    print(header)
    for name, lam0 in lines.items():
        lam = lam0 * (1 + z)
        if not (cfg.band_min_A < lam < cfg.band_max_A):
            continue
        vals = [f_limit(cfg, lam, tt) for tt in (0.75*hr, 3*hr, 12*hr, 48*hr)]
        print(f"  {name:15s} {lam/1e4:6.3f}um " +
              "".join(f"{v:8.1e}" for v in vals))

    # ---- new-physics summary (two arms, cirrus, CR, sigma_z, saturation) ----
    lam = 16000.0
    I100 = (cfg.cirrus_I100_MJysr if cfg.cirrus_I100_MJysr is not None
            else cirrus_I100_from_lat(cfg.gal_lat_deg))
    mu_cir = -2.5*np.log10(cirrus_flambda(5500., I100)*5500.**2/C_A) - 48.6
    print(f"\nDetector arms             : CCD {cfg.read_noise_opt} e-/{cfg.dark_opt} e-/s "
          f"below {cfg.dichroic_split_A/1e4:.1f} um, HgCdTe {cfg.read_noise} e-/"
          f"{cfg.dark_current} e-/s above (n_groups={cfg.n_groups})")
    if cfg.include_cirrus:
        print(f"Galactic cirrus           : |b|={cfg.gal_lat_deg:.0f} deg -> "
              f"I100={I100:.2f} MJy/sr, mu_V(DGL)={mu_cir:.1f} AB/arcsec^2 "
              f"(Ienaka+13 scaling)")
    print(f"Cosmic rays               : efficiency {cfg.cr_exposure_efficiency():.3f} "
          f"per {cfg.t_single:.0f} s exposure (JWST-ETC rate; "
          f"{'folded in' if cfg.include_cr else 'reported only'})")
    print(f"Saturation (spectroscopy) : continuum AB < "
          f"{saturation_contmag_spectro(cfg, lam):.1f}, line < "
          f"{saturation_lineflux_spectro(cfg, lam):.1e} erg/s/cm^2 per "
          f"{cfg.t_single:.0f} s")
    sz = sigma_z(cfg, 1e-16, lam, 0.75*hr, z=lam/6563.0-1.0)
    print(f"Redshift precision        : sigma_z = {sz:.1e} for Halpha of "
          f"1e-16 erg/s/cm^2 at 1.6 um, 0.75 hr")
    print(f"[OII] doublet at z=1.5    : effective FWHM "
          f"{oii_effective_fwhm_A(cfg, 1.5):.1f} A vs resolution "
          f"{3727.4*2.5/cfg.R*1000/1000:.1f} A (split needs R>1340)")

    # compare to the proposal Eq.(12) anchor at 1.6 um
    f075 = f_limit(cfg, lam, 0.75*hr)
    print(f"\nAt 1.6 um, 0.75 hr : ETC F_5sigma = {f075:.2e} erg/s/cm^2")
    print(f"                     Eq.(12) anchor = 1.0e-16 erg/s/cm^2 (Roman/Euclid)")
    nb = noise_breakdown(cfg, lam, 0.75*hr)
    tot = nb["sky_e"] + nb["dark_e"] + nb["read_e2"]
    print(f"   noise budget @0.75hr:  sky {nb['sky_e']:.0f} e-  "
          f"dark {nb['dark_e']:.0f} e-  read^2 {nb['read_e2']:.0f} e-^2  "
          f"(sky fraction {nb['sky_e']/tot*100:.0f}%)  n_pix={nb['n_pix']:.0f}")

    # ---- figure: F_5sigma(lambda) and the zodiacal spectrum ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lam = np.linspace(4000, 25000, 300)
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 7.4), dpi=150,
                                     sharex=True, gridspec_kw={"hspace": 0.08})
        # zodi spectrum (AB/arcsec^2)
        sky = cfg.sky_flambda(lam)
        fnu = sky * lam**2 / C_A
        mu = -2.5*np.log10(fnu) - 48.6
        a1.plot(lam/1e4, mu, color="#b8860b", lw=2, label="zodiacal + cirrus + telescope")
        a1.plot(lam/1e4, -2.5*np.log10(zodi_flambda(lam)*lam**2/C_A)-48.6,
                color="#333", lw=1, ls="--", label="zodiacal only")
        a1.set_ylabel(r"sky $\mu$  [AB arcsec$^{-2}$]"); a1.invert_yaxis()
        a1.legend(fontsize=8, loc="lower left"); a1.grid(alpha=0.2)
        a1.set_title("Slitless ETC: zodiacal background and 5$\\sigma$ line-flux depth",
                     fontsize=11)
        for tt, c in [(0.75*hr, "#3898ec"), (3*hr, "#4ec9b0"), (12*hr, "#d97757")]:
            f = [f_limit(cfg, l, tt) for l in lam]
            a2.plot(lam/1e4, f, color=c, lw=2, label=f"{tt/hr:.2f} hr")
        a2.axhline(1e-16, color="k", ls=":", lw=1, label="Eq.(12) anchor 0.75 hr")
        a2.set_yscale("log"); a2.set_xlabel(r"observed wavelength [$\mu$m]")
        a2.set_ylabel(r"$F_{5\sigma}$ [erg s$^{-1}$ cm$^{-2}$]")
        a2.legend(fontsize=8, ncol=2); a2.grid(alpha=0.2, which="both")
        fig.savefig("etc_f5sigma.png", bbox_inches="tight")
        print("\nwrote etc_f5sigma.png")
    except Exception as e:
        print("plot skipped:", e)


def _cli():
    import argparse
    p = argparse.ArgumentParser(
        description="Slitless emission-line depth / S-N calculator "
                    "(3.5 m R=1000 concept). R and pixel size are inputs.")
    p.add_argument("--R", type=float, default=1000.0, help="resolving power lambda/dlambda")
    p.add_argument("--pix", type=float, default=0.11, help="pixel scale [arcsec/pix]")
    p.add_argument("--diam", type=float, default=350.0, help="aperture diameter [cm]")
    p.add_argument("--eta", type=float, default=0.30, help="peak end-to-end throughput")
    p.add_argument("--read", type=float, default=8.0, help="read noise [e-/pix/exp]")
    p.add_argument("--dark", type=float, default=0.010, help="dark current [e-/s/pix]")
    p.add_argument("--nexp", type=int, default=3, help="number of exposures (rolls)")
    p.add_argument("--zodi", type=float, default=22.1, help="zodi mu at 0.5um [AB/arcsec^2]")
    p.add_argument("--source", type=float, default=0.3, help="source FWHM [arcsec]")
    p.add_argument("--no-compare", action="store_true", help="skip Roman/Euclid validation")
    p.add_argument("--realistic", action="store_true", help="use CALSPEC zodi + component throughput CSVs")
    p.add_argument("--cooling", action="store_true", help="run the telescope-cooling tradeoff")
    a = p.parse_args()
    kw = dict(diameter_cm=a.diam, R=a.R, pix_scale=a.pix, eta_peak=a.eta,
              read_noise=a.read, dark_current=a.dark, n_exp=a.nexp, zodi_mu_ref=a.zodi)
    cfg = realistic_cfg(**kw) if a.realistic else InstrumentConfig(**kw)
    main(cfg)
    if not a.no_compare:
        compare_missions()
    if a.cooling:
        cooling_tradeoff()


if __name__ == "__main__":
    _cli()
