# Slitless emission-line ETC (3.5 m R=1000 concept)

A wavelength-resolved exposure-time / depth calculator for a wide-field slitless
grism survey, written for the 3.5 m segmented-mirror R=1000 cosmology proposal.
Fidelity target is a Euclid/Roman-class ETC.

## What it models
- **Zodiacal background as a spectrum** — the real STScI CALSPEC solar reference
  spectrum scattered by interplanetary dust, reddened by ~0.3 mag in (V−K) after
  Leinert et al. 1998, plus the interplanetary-dust thermal term.
- **Telescope thermal self-emission** (graybody), which sets a near-IR "thermal
  wall" whose onset depends on the optics temperature.
- **Wavelength-dependent throughput and detector QE** — e2v CCD (optical) and
  Teledyne H2RG HgCdTe (near-IR) blended through a 1.0 µm dichroic.
- **Full per-pixel noise budget** — source shot + sky + dark + read (ramp),
  summed over the line footprint, with the slitless background integrated over
  the band-limiting filter.
- Exact 5σ line-flux depth by solving the quadratic (source shot noise retained).
- **R and pixel size are inputs** (CLI or `InstrumentConfig`).

## Files
| file | purpose |
|---|---|
| `slitless_etc.py` | the calculator (depth, S/N, Roman/Euclid cross-check, cooling tradeoff) |
| `make_etc_data.py` | builds the CALSPEC-solar zodiacal spectrum and component QE/throughput CSVs |
| `etc_zodi.csv`, `etc_qe.csv`, `etc_throughput.csv` | generated input curves |
| `etc_f5sigma.png`, `etc_cooling.png` | example outputs |

## Usage
```bash
python make_etc_data.py                       # regenerate the data curves (downloads CALSPEC solar)
python slitless_etc.py --realistic --cooling  # full run + Roman/Euclid cross-check + cooling tradeoff
python slitless_etc.py --R 1000 --pix 0.11    # override resolving power / pixel scale
```

## Validation
Run on the published Roman and Euclid grism parameters, the calculator reproduces
the Roman HLSS depth (1×10⁻¹⁶ erg s⁻¹ cm⁻² at 6.5σ; Wang et al. 2022) to ~5% and
the Euclid Wide depth (2×10⁻¹⁶ at 3.5σ; Euclid prep. XXX 2023) to a factor of two.

## References
Leinert et al. 1998 (A&AS 127, 1); Wang et al. 2022 (ApJ 928, 1);
Euclid Collaboration, Euclid preparation XXX 2023 (A&A 676, A34);
solar spectrum: STScI CALSPEC `sun_reference_stis_002`.
