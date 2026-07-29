# Mission ETC

`mission_etc` separates observing questions that were previously exposed
through one mixed interface. `slitless_etc.py` remains the numerical kernel
and its existing API is unchanged.

## Modes

| Mode | Scientific question | Required source unit |
|---|---|---|
| `imaging` | Broadband point-source depth and saturation | AB magnitude |
| `line` | S/N or limiting flux of one unresolved line | erg s^-1 cm^-2 |
| `template` | Detector-sampled line profile or transient spectrum | erg s^-1 cm^-2 A^-1 |
| `mir` | Cooled MIR imaging depth | microJy |
| `validate` | Internal regression checks | named built-in case |

The template mode requires observed wavelength in Angstrom and physical
`F_lambda`. It rejects incomplete channel coverage. Generic ASCII input does
not infer its original spectral resolution. Supply `--native-R` for observed
data so that the original line-spread function is not applied twice.

## Spectroscopy channels

The compact-object channels use order-sorting bands at `R=5000`.
The transient channels provide the `R=1000` alternative needed for broad,
rapidly fading kilonova features.

```text
compact-blue       4300-5100 A    R=5000
compact-red        6200-6900 A    R=5000
compact-nir       10000-15000 A   R=5000
transient-optical  4000-10000 A   R=1000
transient-nir     10000-15000 A   R=1000
```

These are proposal requirements. They are not measured hardware passbands.
Replace the inherited throughput, PSF, detector, and wavelength-calibration
terms when subsystem measurements become available.

## Examples

List modes and channel definitions.

```bash
python -m mission_etc modes
python -m mission_etc channels
```

Calculate an H-alpha line limit and the S/N of a supplied line.

```bash
python -m mission_etc line compact-red 6562.8 1800
python -m mission_etc line compact-red 6562.8 1800 \
  --flux-erg-s-cm2 2e-17
```

Simulate a calibrated spectrum, save all detector and noise columns, bin the
result to `R=1000`, and measure H-alpha.

```bash
python -m mission_etc template compact-red spectrum.txt 1800 \
  --native-R 8800 --uncertainty-column 2 \
  --ab-magnitude 22 --normalization-min-A 6400 \
  --normalization-max-A 6800 --bin-R 1000 \
  --line-center-A 6562.8 --output-csv detector_spectrum.csv
```

Calculate optical/NIR and MIR imaging depths.

```bash
python -m mission_etc imaging "SDSS r" 1800 --magnitude-ab 25
python -m mission_etc mir NC1 180 --flux-uJy 100
```

Run the deterministic inverse-S/N regression check.

```bash
python -m mission_etc validate
```

## Public observed templates

`mission_etc.templates` retains the source state, epoch, wavelength frame,
flux unit, uncertainty, native resolving power, citation, and download URL.
The supported reference cases are:

| Object | Public product | ETC use | Resolution restriction |
|---|---|---|---|
| AT2017gfo | ENGRAVE v1.0 unsmoothed X-shooter arms | Optical and NIR kilonova evolution | UVB `R=5100`, VIS `R=8800`, NIR `R=5600` |
| U Gem | LAMOST DR11 spectrum 89907231 | 2012 outburst continuum and H-alpha profile | Native `R` is about 1800, so it cannot validate `R=5000` peak recovery |
| U Gem | Naylor et al. 2005 CDS phase spectra | Donor radial velocity and phase smearing | Relative flux only; normalize before count-rate use |

The AT2017gfo release requires citation of Pian et al. (2017, Nature, 551,
67) and Smartt et al. (2017, Nature, 551, 75). Its narrow ground-based
telluric intervals are not intrinsic kilonova features. The reference-case
script quality-masks those intervals before interpolation for a broad-feature
space forecast and records the operation in the template notes.

Download the checksum-pinned products and generate the current reference
figure and CSV.

```bash
python -m mission_etc.reference_cases \
  --cache-dir compact_template_cache \
  --output-dir appendix_x_assets
```

The downloaded archive is deliberately not version controlled.

## Resolution handling

For Gaussian line-spread functions, the code applies only the kernel required
to move from the template resolution to the requested mission resolution.
For example, a VIS X-shooter template at `R=8800` is broadened to `R=5000`
with the quadrature difference of the two LSF widths. A LAMOST spectrum at
`R=1800` is never sharpened to `R=5000`; its result metadata reports an
effective resolution of `R=1800`. Source angular extent is then added in
quadrature as a separate slitless morphology term.

## Inverse planning and Monte Carlo

The programmatic API solves for an integrated-feature S/N, line S/N, or
centroid precision. A phase ceiling can stop the solution before an exposure
smears too much of a binary orbit.

```python
from mission_etc import solve_exposure_time

solution = solve_exposure_time(
    spectrum,
    "compact-red",
    metric="line_snr",
    target_value=20,
    line_center_A=6562.8,
    phase_ceiling_s=600,
)
```

`monte_carlo_line_recovery` draws extracted detector realizations. Shot,
background, dark, and read terms vary per spectral pixel. Relative
calibration and contamination residuals vary coherently across a channel.
The result reports centroid bias, empirical scatter, line-S/N scatter, and
the recovery fraction. This empirical scatter should be used when it exceeds
the local Fisher estimate.

## Multi-roll scene recovery

The scene calculation maps direct-image positions and physical spectra onto
two-dimensional detector images. A shared sparse scene model is fitted to all
selected roll angles. The reference case injects a velocity-shifted public
U Gem spectrum among two overlapping traces. The comparison holds the total
science exposure at 60 seconds. The one-roll case uses one 60 second exposure
and the three-roll case uses 20 seconds at each of three orientations.

```bash
python -m mission_etc.scene_reference \
  --cache-dir compact_template_cache \
  --output-dir appendix_x_assets --trials 80
```

The resulting CSV records the injected velocity, recovery fraction, empirical
bias and scatter, effective resolution, exposure per roll, and total exposure.
The LAMOST input has a native resolving power near 1800. The experiment tests
trace separation and velocity recovery but does not validate double-peak
recovery at the proposed resolving power of 5000.

## Template noise model

The output CSV keeps source, sky, dark, contamination, read variance,
systematic variance, total variance, and S/N in separate columns. In slitless
mode each detector pixel receives diffuse sky integrated over the full
band-limiting filter. The target contributes only its local spectral-bin flux.
Contamination adds photon noise and a separately configurable model-residual
term. Integrated S/N and lower-resolution binning treat the relative flux
floor and contamination-model residual as channel-correlated terms. They
therefore do not average away as independent pixel noise.
