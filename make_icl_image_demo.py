#!/usr/bin/env python3
"""How the intracluster light (ICL) of Abell 2744 would appear to the 3.5 m ST
as a function of exposure time, as a dedicated ICL / low-surface-brightness deep
program (independent of the ELG wedding-cake tiers).

The deep HST Frontier Fields WFC3/IR F160W mosaic is the true scene. It already
carries its own noise (its RMS map), so for each 3.5 m exposure we RAISE the
noise to the ETC-predicted level by ADDING noise in quadrature:
    sigma_add^2 = sigma_ETC^2 - sigma_HST^2   (clipped at 0),
so the displayed map has exactly the ETC per-pixel noise. The S/N annotated on
each panel is the ETC value (source + sky + dark + read), and the background
noise of the simulation is set to match it.
"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import block_reduce
import astropy.units as u
from scipy.ndimage import gaussian_filter
import slitless_etc as etc

HR = 3600.0
BAND_H = (14000.0, 18000.0)                 # 1.4-1.8 um imaging band
HPIX = 0.06                                 # HST WFC3/IR 60mas mosaic [arcsec/pix]
F = 2                                        # block-reduce factor -> 0.12" 3.5m pixel
PIX = HPIX * F                               # 0.12" ~ 3.5m design 0.11"
MU_FID = 27.0                                # fiducial ICL surface brightness for S/N
BINAS = 10.0                                 # S/N and depth reference box [arcsec]
BASE = ("/tmp/claude-10396/-home-kjhan-BACKUP-3-5ST/"
        "d2e05355-c544-4223-9f95-d63a9ca5c011/scratchpad")
EXPOS = [1.0, 4.0, 16.0, 64.0]               # dedicated ICL deep-survey ladder [hr]
np.random.seed(42)


def sb_rate_per_pix(cfg, mu_ab, band):
    """e-/s per 3.5 m pixel from a uniform source of AB surface brightness mu."""
    lo, hi = band
    lam = np.linspace(lo, hi, 300)
    fnu = 3631.0e-23 * 10 ** (-0.4 * mu_ab)
    flam = fnu * 2.99792458e18 / lam ** 2
    surf = np.trapezoid(flam * etc._photon_factor(lam) * cfg.throughput_imaging(lam), lam)
    return surf * cfg.area_cm2 * cfg.omega_pix


cfg = etc.realistic_cfg(diameter_cm=350.0, obstruction=0.15, pix_scale=PIX, R=1000.0,
                        band_min_A=3600.0, band_max_A=30000.0)
R0 = sb_rate_per_pix(cfg, 0.0, BAND_H)          # e-/s/pix for f=10^(-0.4 mu)=1
SKY = etc.background_per_pixel(cfg, BAND_H, imaging=True)   # e-/s/pix
DARK, READ = cfg.dark_current, cfg.read_noise
NRD = cfg.n_reads                                # reads vs total integration
NBOX = (BINAS / PIX) ** 2                        # pixels in the reference box


def etc_snr(mu, t_hr, npix=NBOX):
    """ETC S/N of a uniform source of surface brightness mu over npix, exposure t."""
    t = t_hr * HR
    s = R0 * 10 ** (-0.4 * mu)
    return s * t * np.sqrt(npix) / np.sqrt((s + SKY + DARK) * t + NRD(t) * READ ** 2)


def mu_limit(t_hr, npix=NBOX):
    """5-sigma surface-brightness limit [AB/arcsec^2] over npix at exposure t."""
    t = t_hr * HR
    src_lim = 5.0 * np.sqrt((SKY + DARK) * t + NRD(t) * READ ** 2) / (t * np.sqrt(npix))
    return -2.5 * np.log10(src_lim / R0)


def mu_noise_pix(t_hr):
    """1-sigma per-pixel background-noise surface brightness [AB/arcsec^2]."""
    t = t_hr * HR
    sig = np.sqrt((SKY + DARK) * t + NRD(t) * READ ** 2) / t
    return -2.5 * np.log10(sig / R0)


# ---- HST F160W truth (signal + its own RMS) -> 3.5 m grid, in source-rate units ----
img = fits.open(f"{BASE}/a2744_f160w.fits")[0]
rms = fits.open(f"{BASE}/a2744_f160w_rms.fits")[0].data.astype("f8")
hdr = img.header; d = img.data.astype("f8"); w = WCS(hdr)
ZP = -2.5 * np.log10(hdr["PHOTFLAM"]) - 5 * np.log10(hdr["PHOTPLAM"]) - 2.408
px, py = [int(v) for v in w.world_to_pixel(SkyCoord(3.5877 * u.deg, -30.4003 * u.deg))]
half = 1250
sl = (slice(py - half, py + half), slice(px - half, px + half))
cut, rcut = d[sl], rms[sl]
rcut = np.where((rcut > 0) & (rcut < 1e6), rcut, np.nan)          # mask no-data sentinel
KL = 10 ** (-0.4 * ZP) / HPIX ** 2               # e/s per HST pix -> linear SB f=10^(-0.4 mu)
f_hst = KL * np.clip(cut, 0, None)
sf_hst = KL * np.nan_to_num(rcut, nan=np.nanmedian(rcut))         # existing noise in f-units
# block-reduce to the 3.5 m pixel (mean preserves surface brightness; noise averages)
f2 = block_reduce(f_hst, F, func=np.mean)
sf2 = np.sqrt(block_reduce(sf_hst ** 2, F, func=np.mean)) / F
src = R0 * f2                                    # true source e-/s per 3.5 m pixel
sig_exist = R0 * sf2                             # existing HST noise, rate units
fwhm = cfg.psf_fwhm(16000.0)
src = gaussian_filter(src, (fwhm / 2.355) / PIX)
kpc_per_arcsec = 4.53                            # z=0.308, flat LCDM H0=70
mu_hst_depth = -2.5 * np.log10(5.0 * np.nanmedian(sig_exist) / np.sqrt(NBOX) / R0)
print(f"R0={R0:.3e} sky={SKY:.3f} dark={DARK} read={READ}")
print(f"HST truth median existing noise -> mu_5sigma(10'')~{mu_hst_depth:.2f}; scene {src.shape}")


def observe(t_hr):
    """Mock 3.5 m sky-subtracted count-rate map; noise added in quadrature."""
    t = t_hr * HR
    var_etc = ((src + SKY + DARK) * t + NRD(t) * READ ** 2) / t ** 2    # ETC noise, rate^2
    var_add = np.clip(var_etc - sig_exist ** 2, 0.0, None)             # quadrature top-up
    return src + np.random.normal(0.0, np.sqrt(var_add))


VMIN = 20.0                                     # fixed bright end [AB/arcsec^2]
fig, axes = plt.subplots(2, 2, figsize=(9.8, 9.8), dpi=150, constrained_layout=True)
extent = [0, src.shape[1] * PIX, 0, src.shape[0] * PIX]
bar = 100.0 / kpc_per_arcsec                     # 100 kpc in arcsec
for ax, t_hr in zip(axes.ravel(), EXPOS):
    rate = observe(t_hr)
    mu = -2.5 * np.log10(np.clip(rate, 1e-12, None) / R0)
    vmax = mu_noise_pix(t_hr)                    # dark end = this exposure's 1sigma/pix noise
    im = ax.imshow(mu, origin="lower", cmap="magma_r", vmin=VMIN, vmax=vmax, extent=extent)
    ax.set_title(f"{t_hr:g} hr    $\\mu_{{5\\sigma}}$(10$''$)={mu_limit(t_hr):.1f}    "
                 f"S/N($\\mu$27)={etc_snr(MU_FID, t_hr):.0f}", fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(Rectangle((6, 6), bar, 2.2, color="white"))
    ax.text(6, 11, "100 kpc", fontsize=7.5, color="white")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("$\\mu$ [AB mag arcsec$^{-2}$]", fontsize=7.5)
    cb.ax.tick_params(labelsize=7); cb.ax.invert_yaxis()
fig.suptitle("Dedicated ICL deep survey: Abell 2744 seen by the 3.5 m ST vs exposure time\n"
             "(HST Frontier Fields F160W truth; ETC S/N, quadrature noise; each panel "
             "stretched to its own 1$\\sigma$/pixel noise floor; $H$ band, $z=0.308$)",
             fontsize=10.3)
fig.savefig("icl_exposure_demo.png", bbox_inches="tight")
print("wrote icl_exposure_demo.png")
for t_hr in EXPOS:
    print(f"  {t_hr:5.1f} hr: mu_5sigma(10'')={mu_limit(t_hr):.2f}, mu_noise/pix(vmax)="
          f"{mu_noise_pix(t_hr):.2f}, S/N(mu=27,10'')={etc_snr(MU_FID, t_hr):.1f}")
