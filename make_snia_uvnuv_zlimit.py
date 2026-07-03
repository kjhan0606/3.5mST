#!/usr/bin/env python3
"""S/N-limited redshift of the SN Ia rest-frame U and NUV filter magnitudes
(Figure fig:snuvlimit).

The optical-twin diagnostic is a rest-frame U/NUV magnitude difference, the
NUV-red and NUV-blue subgroups of Milne et al. (2015) being offset by ~0.4 mag
in NUV-optical colour while the optical light curves match. Separating the two
subgroups for a single supernova at 4 sigma therefore needs sigma_m ~ 0.1 mag
in the NUV band (the optical side is far brighter), i.e. S/N = 2.5/(ln10 x
0.1) ~ 11, and using Delta NUV as a continuous standardization covariate at
the sigma <~ 0.08 mag Hubble-residual target needs sigma_m ~ 0.05 mag, i.e.
S/N ~ 22.

Synthetic rest-frame top-hat bands U 3300-3900 A and NUV 2500-3040 A (centre
2770 A, the diagnostic wavelength used throughout the proposal) are redshifted
with the source, and the magnitude is measured by binning the multi-roll
slitless spectrum over the redshifted band. The S/N follows the exposure-time
calculator's slitless budget, the source counts under the binned band against
the dispersed zodiacal+cirrus+thermal sky of the 4000 A band-limiting filter,
detector dark current, and read noise, with the Nugent SN Ia peak template
normalised to M_B = -19.3. The NUV band is fully inside the 0.36 micron blue
edge for z >= 0.44 and the U band for z >= 0.09.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import slitless_etc as etc
import galaxy_etc as g

HR = 3600.0
MB = -19.3
d = np.loadtxt("sn_templates/SNIa_peak.dat")
SNIA = dict(wl=d[:, 0], flam=d[:, 1], desc="SN Ia peak (Nugent)")
CFG = etc.realistic_cfg(diameter_cm=350.0, obstruction=0.15, pix_scale=0.11, R=1000.0)
FILT_W = 4000.0                       # band-limiting filter width [A], ETC default

BANDS = [
    ("rest $U$ (3300--3900Å)", 3300.0, 3900.0, "#2a6cb5"),
    ("rest NUV (2500--3040Å)", 2500.0, 3040.0, "#7d3c98"),
]
# sigma_m targets: 0.10 mag separates the 0.4 mag NUV-red/blue offset at 4 sigma,
# 0.05 mag supports the Delta NUV standardization covariate
TARGETS = [(0.10, "-"), (0.05, "--")]
EXPOSURES = [2.0, 10.0, 30.0]         # planning exposures [hr], three rolls combined


def slitless_band_snr(mag_ab, lam1, lam2, t_s):
    """S/N of a synthetic-filter magnitude binned from the slitless spectrum.

    Same budget as slitless_etc.line_sn, with the signal integrated over the
    synthetic band [lam1, lam2] and the dispersed sky of the band-limiting
    filter collected by every pixel under the bin.
    """
    lamc = 0.5 * (lam1 + lam2)
    band = (max(CFG.band_min_A, lamc - FILT_W / 2),
            min(CFG.band_max_A, lamc + FILT_W / 2))
    teff = t_s * (CFG.cr_exposure_efficiency() if CFG.include_cr else 1.0)
    flam = etc.ab_to_flambda(mag_ab, lamc)
    S = (flam * CFG.area_cm2 * CFG.throughput(lamc) * etc._photon_factor(lamc)
         * (lam2 - lam1) * teff * CFG.extraction_eff)
    Bpix = etc.background_per_pixel(CFG, band)
    theta = CFG.psf_fwhm(lamc)
    n_spatial = max(1.0, theta / CFG.pix_scale)
    n_spectral = (lam2 - lam1) / CFG.dispersion_A_per_pix(lamc)
    npix = n_spatial * n_spectral
    _, dark = CFG.detector_at(lamc)
    var = (S + (Bpix + dark) * npix * teff
           + npix * CFG.n_reads(t_s) * CFG.read_noise_eff(lamc) ** 2
           + (CFG.flat_error * S) ** 2)
    return float(S / np.sqrt(var))


def zmin_inband(l1_rest):
    return CFG.band_min_A / l1_rest - 1.0


fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), dpi=180, sharey=True)
summary = {}
for ax, (label, l1, l2, col) in zip(axes, BANDS):
    z0 = max(0.10, np.ceil(zmin_inband(l1) * 100) / 100)
    zz = np.linspace(z0, 2.6, 70)
    for sig, ls in TARGETS:
        snr = 2.5 / (np.log(10.0) * sig)
        tt = []
        for z in zz:
            lo1, lo2 = l1 * (1 + z), l2 * (1 + z)
            m = g.app_mag(SNIA, z, 0.5 * (lo1 + lo2), lo2 - lo1,
                          M_abs=MB, ref_lam=4400.0, ref_width=1000.0)
            tt.append(etc.exposure_for_snr(
                lambda t: slitless_band_snr(m, lo1, lo2, t), snr) / HR)
        tt = np.array(tt)
        ax.plot(zz, tt, color=col, ls=ls, lw=2.2,
                label=rf"$\sigma_m={sig:.2f}$ mag (S/N$\,\simeq\,${snr:.0f})")
        for thr in EXPOSURES:
            zc = np.interp(np.log(thr), np.log(tt), zz)
            summary[(label, sig, thr)] = (zc, np.interp(zc, zz, tt))
    for thr, tcol in zip(EXPOSURES, ("#1f7a4d", "#d68f1f", "#c0392b")):
        ax.axhline(thr, color=tcol, lw=1.0, ls=":")
        ax.text(2.26, thr * 1.12, f"{thr:.0f} hr", color=tcol, fontsize=8.5,
                ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_xlim(0.0, 2.3)
    ax.set_ylim(0.02, 300)
    ax.set_xlabel("redshift $z$", fontsize=12)
    ax.set_title(label, fontsize=11)
    ax.grid(alpha=0.25, which="both", lw=0.5)
    ax.legend(fontsize=9, loc="upper left")
    ax.axvline(zmin_inband(l1), color="0.5", lw=1.0, ls="--")
    ax.text(zmin_inband(l1) + 0.03, 0.025, "band fully\nabove 0.36$\\mu$m",
            fontsize=8, color="0.4", va="bottom")
axes[0].set_ylabel("exposure for a peak SN Ia [hr]", fontsize=12)
fig.suptitle("S/N-limited reach of the rest-frame $U$ and NUV twin photometry "
             "(binned multi-roll slitless spectra, $M_B=-19.3$)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("snia_uvnuv_zlimit.png", bbox_inches="tight")
print("wrote snia_uvnuv_zlimit.png\n")

for (label, sig, thr), (zc, _) in sorted(summary.items(), key=str):
    print(f"{label:26s} sigma={sig:.2f}  {thr:5.1f} hr  ->  z_lim = {zc:.2f}")

print("\nobserved synthetic-band AB at selected z:")
for label, l1, l2, _ in BANDS:
    for z in (0.5, 0.8, 1.0, 1.2, 1.5):
        if l1 * (1 + z) < CFG.band_min_A:
            continue
        lo1, lo2 = l1 * (1 + z), l2 * (1 + z)
        m = g.app_mag(SNIA, z, 0.5 * (lo1 + lo2), lo2 - lo1,
                      M_abs=MB, ref_lam=4400.0, ref_width=1000.0)
        s2 = slitless_band_snr(m, lo1, lo2, 2 * HR)
        s10 = slitless_band_snr(m, lo1, lo2, 10 * HR)
        print(f"  {label:26s} z={z:.1f}  AB={m:5.2f}  S/N(2hr)={s2:6.1f}  S/N(10hr)={s10:6.1f}")
