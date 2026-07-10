#!/usr/bin/env python3
r"""
segmented_psf.py -- Physical-optics point-spread function for the 3.5 m
19-segment concept primary mirror (the hex-tiled layout sketched in the
proposal's title figure), computed the same way JWST/Roman PSFs are actually
computed: Fraunhofer diffraction of the true segmented pupil via FFT
(pupil -> focal plane), not the closed-form obscured-Airy formula used
elsewhere in this repo.

slitless_etc.py's encircled_energy()/_airy_intensity() model the pupil as a
smooth annulus (a plain circular aperture with a single central-obstruction
fraction). That is exact for the *envelope* of a well-phased segmented
mirror's PSF but misses two real effects a segmented, strut-supported
primary actually has:
  * segment gaps and support-strut diffraction, which spread some energy out
    of the core into spikes/extended structure even for a perfectly phased
    mirror;
  * segment phasing errors (piston/tip/tilt misalignment between segments),
    which further move flux from the core into a diffuse halo and are the
    dominant error term budgeted for real segmented telescopes (e.g. JWST's
    commissioning phasing campaign).

Built on POPPY (Perrin et al. 2012 SPIE; https://poppy-optics.readthedocs.io),
the physical-optics propagation engine underneath STScI's WebbPSF/STPSF --
the same PSF-simulation approach JWST's and Roman's own ETC (pandeia.engine)
draw their PSF libraries from (confirmed from pandeia.engine's roman.py
_loadpsfs(), which loads per-filter STPSF libraries rather than an analytic
formula).

Segment count/layout (19 hex segments, rings=2 + center), gap width, and
secondary/spider geometry are illustrative placeholders -- no mechanical
design exists yet for this concept telescope -- and should be replaced with
real values once the primary-mirror design is fixed. The circumscribed
diameter (3.5 m) and central-obstruction fraction (0.15) match
InstrumentConfig's defaults in slitless_etc.py so the two PSF models are
compared on equal footing.
"""
from __future__ import annotations
import numpy as np
import astropy.units as u
import poppy


# ---------------------------------------------------------------- geometry
def _flattoflat_for_diameter(diameter_m, rings, gap_m):
    """Segment flat-to-flat size that gives the requested circumscribed
    pupil diameter for this ring count/gap (pupil_diam scales ~linearly with
    flattoflat, so one probe build is enough to solve it exactly)."""
    probe = poppy.HexSegmentedDeformableMirror(rings=rings, flattoflat=1.0 * u.m,
                                               gap=gap_m * u.m, center=True)
    return diameter_m / probe.pupil_diam.to_value(u.m)


def build_optical_system(diameter_m=3.5, rings=2, gap_m=0.003, obstruction=0.15,
                         n_supports=4, support_width_m=0.01, oversample=4,
                         fov_arcsec=2.0, pixelscale_arcsec=0.01,
                         piston_rms_nm=0.0, tilt_rms_urad=0.0, seed=12345):
    """Build a POPPY OpticalSystem for the segmented primary plus central
    obscuration/spider, with an optional random per-segment phasing error.

    Parameters
    ----------
    diameter_m : circumscribed primary-mirror diameter [m]
    rings : hex rings around the center segment (2 -> 19 segments: 1+6+12,
        matching the proposal's concept sketch)
    gap_m : inter-segment gap [m] (illustrative; JWST-scale ~2.5-3 mm)
    obstruction : linear central-obstruction fraction (matches
        InstrumentConfig.obstruction in slitless_etc.py)
    n_supports, support_width_m : secondary-mirror spider design (illustrative)
    piston_rms_nm : RMS per-segment piston (phasing) error [nm]; 0 = perfectly
        phased mirror. Surface piston; POPPY doubles it to the reflected OPD.
    tilt_rms_urad : RMS per-segment tip/tilt error [microradians]; 0 = none

    Returns
    -------
    osys : poppy.OpticalSystem
    dm   : poppy.HexSegmentedDeformableMirror (query .segmentlist,
        .set_actuator(), .get_opd() etc. for the realized pupil/error map)
    """
    flattoflat = _flattoflat_for_diameter(diameter_m, rings, gap_m)
    dm = poppy.HexSegmentedDeformableMirror(rings=rings, flattoflat=flattoflat * u.m,
                                            gap=gap_m * u.m, center=True)

    if piston_rms_nm > 0 or tilt_rms_urad > 0:
        rng = np.random.default_rng(seed)
        n_seg = len(dm.segmentlist)
        pistons = rng.normal(0.0, piston_rms_nm * 1e-9, n_seg)     # m
        tips = rng.normal(0.0, tilt_rms_urad * 1e-6, n_seg)        # rad
        tilts = rng.normal(0.0, tilt_rms_urad * 1e-6, n_seg)       # rad
        for i, seg in enumerate(dm.segmentlist):
            dm.set_actuator(seg, pistons[i], tips[i], tilts[i])

    secondary_radius = obstruction * diameter_m / 2.0
    obsc = poppy.SecondaryObscuration(secondary_radius=secondary_radius * u.m,
                                      n_supports=n_supports,
                                      support_width=support_width_m * u.m)

    osys = poppy.OpticalSystem(oversample=oversample)
    osys.add_pupil(dm)
    osys.add_pupil(obsc)
    osys.add_detector(pixelscale=pixelscale_arcsec, fov_arcsec=fov_arcsec)
    return osys, dm


def monochromatic_psf(lam_um, **kw):
    """Monochromatic PSF at lam_um [microns]. Returns (psf_hdulist, osys, dm)."""
    osys, dm = build_optical_system(**kw)
    psf = osys.calc_psf(wavelength=lam_um * 1e-6)
    return psf, osys, dm


# ------------------------------------------------------------- diagnostics
def ee_curve(psf_hdulist, r_arcsec):
    """Encircled energy at one or more radii [arcsec], via POPPY's own
    radial-profile utility (poppy.utils.measure_ee)."""
    ee_func = poppy.utils.measure_ee(psf_hdulist, normalize="None")
    r = np.atleast_1d(np.asarray(r_arcsec, float))
    return np.array([float(ee_func(rr)) for rr in r])


def fwhm_arcsec(psf_hdulist):
    """Delivered PSF FWHM [arcsec] (poppy.utils.measure_fwhm)."""
    return float(poppy.utils.measure_fwhm(psf_hdulist))


def bin_to_detector(psf_hdulist):
    """Flux-conserving block-sum of the (oversampled) PSF array down to the
    true detector pixel scale requested via pixelscale_arcsec in
    build_optical_system()/monochromatic_psf() -- calc_psf() always returns
    the internally oversampled array (its PIXELSCL header is
    pixelscale_arcsec/oversample, not pixelscale_arcsec itself), so this is
    needed to see what the detector's own pixels would actually record
    (e.g. the 3.5mST concept's 0.11"/pix imaging plate scale,
    InstrumentConfig.pix_scale in slitless_etc.py), rather than an
    arbitrarily finely resampled view.

    Returns (binned_array, detector_pixelscale_arcsec)."""
    d = psf_hdulist[0].data
    os_ = int(psf_hdulist[0].header["OVERSAMP"])
    fine_scale = float(psf_hdulist[0].header["PIXELSCL"])
    n = d.shape[0]
    nb = n // os_
    trimmed = d[:nb * os_, :nb * os_]
    binned = trimmed.reshape(nb, os_, nb, os_).sum(axis=(1, 3))
    return binned, fine_scale * os_


# ---------------------------------------------------- ETC-table generation
def make_ee_table(out_csv="segmented_ee_table.csv", lam_um_grid=None,
                  r_arcsec_grid=None, piston_rms_nm=30.0, tilt_rms_urad=0.05,
                  **kw):
    """Precompute a (lambda, r) -> encircled-energy grid from the real
    segmented-pupil physical-optics PSF, for use as slitless_etc.py's
    InstrumentConfig.psf_ee_csv override (see aperture_ee() there). One
    monochromatic PSF is computed per wavelength (a few seconds total); EE at
    every radius in r_arcsec_grid is then read off that PSF's own radial
    profile, so this is cheap even for a fine radius grid.

    piston_rms_nm=30 (default) is an illustrative "reasonably well-phased"
    residual -- comparable in scale to segment-phasing residuals achieved
    operationally on JWST after fine-phasing -- not a value derived from a
    real error budget for this concept telescope, since none exists yet.
    Regenerate with piston_rms_nm=0 for the ideal (perfectly phased) case.
    """
    if lam_um_grid is None:
        lam_um_grid = np.array([0.5, 0.7, 0.9, 1.2, 1.6, 2.2, 2.8])
    if r_arcsec_grid is None:
        r_arcsec_grid = np.concatenate([np.linspace(0.02, 0.5, 25),
                                        np.linspace(0.55, 1.5, 10)])

    rows = []
    for lam_um in lam_um_grid:
        psf, _, _ = monochromatic_psf(lam_um, piston_rms_nm=piston_rms_nm,
                                      tilt_rms_urad=tilt_rms_urad, **kw)
        ee = ee_curve(psf, r_arcsec_grid)
        for r, e in zip(r_arcsec_grid, ee):
            rows.append((lam_um * 1e4, r, e))       # lam_A, r_arcsec, ee

    rows = np.asarray(rows)
    header = "lam_A,r_arcsec,ee"
    np.savetxt(out_csv, rows, delimiter=",", header=header, comments="")
    print(f"wrote {out_csv}: {len(lam_um_grid)} wavelengths x "
          f"{len(r_arcsec_grid)} radii, piston_rms={piston_rms_nm:.0f} nm, "
          f"tilt_rms={tilt_rms_urad:.2f} urad")
    return out_csv


if __name__ == "__main__":
    make_ee_table()
