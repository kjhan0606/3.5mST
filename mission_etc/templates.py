"""Public observed-spectrum templates and provenance-aware loaders.

The ETC uses these spectra as source templates.  Their measurement errors are
retained for provenance and optional template perturbations, but are not added
to the predicted mission detector noise by default.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Mapping
from urllib.request import urlopen
import shutil
import zipfile

import numpy as np

from .spectroscopy import ObservedSpectrum


AT2017GFO_ARCHIVE_URL = (
    "https://sid.erda.dk/share_redirect/df1fMhon6Z/AT2017gfo.zip"
)
AT2017GFO_ARCHIVE_SHA256 = (
    "55bb6550e4400d96a8a0e8f9952de829639c849986044ac5ddd8f86af6da2b83"
)
UGEM_LAMOST_URL = "https://www.lamost.org/dr11/v2.0/spectrum/fits/89907231"
UGEM_LAMOST_SHA256 = (
    "715d033bfb73d41fdb60fd13918a9f125b4f33de2357f8b9a1fc80e2c151186a"
)
UGEM_CDS_BASE_URL = (
    "https://cdsarc.cds.unistra.fr/ftp/J/MNRAS/361/1091/sp"
)


@dataclass(frozen=True)
class SpectrumProvenance:
    """Scientific and technical metadata needed to reuse a spectrum."""

    key: str
    object_name: str
    source_class: str
    state: str
    epoch_mjd: float | None
    phase_days: float | None
    wavelength_frame: str
    flux_unit: str
    flux_calibrated: bool
    native_resolving_power: float
    citation: str
    source_url: str
    notes: str = ""

    def __post_init__(self) -> None:
        if self.wavelength_frame not in {"observed", "rest"}:
            raise ValueError("Wavelength frame must be 'observed' or 'rest'.")
        if self.native_resolving_power <= 0:
            raise ValueError("Native resolving power must be positive.")


@dataclass(frozen=True)
class CalibratedSpectrum:
    """Observed spectrum with uncertainty, mask, and complete provenance."""

    wavelength_A: np.ndarray
    flux_lambda: np.ndarray
    uncertainty_flux_lambda: np.ndarray
    valid: np.ndarray
    provenance: SpectrumProvenance

    def __post_init__(self) -> None:
        wavelength = np.asarray(self.wavelength_A, dtype=float)
        flux = np.asarray(self.flux_lambda, dtype=float)
        uncertainty = np.asarray(self.uncertainty_flux_lambda, dtype=float)
        valid = np.asarray(self.valid, dtype=bool)
        if any(item.ndim != 1 for item in (wavelength, flux, uncertainty, valid)):
            raise ValueError("Spectrum columns must be one-dimensional.")
        if len({item.size for item in (wavelength, flux, uncertainty, valid)}) != 1:
            raise ValueError("Spectrum columns must have equal lengths.")
        if wavelength.size < 3 or np.any(np.diff(wavelength) <= 0):
            raise ValueError("Wavelength must be strictly increasing with at least 3 rows.")
        if np.any(wavelength <= 0) or not np.all(np.isfinite(wavelength)):
            raise ValueError("Wavelength values must be finite and positive.")
        valid &= np.isfinite(flux) & np.isfinite(uncertainty) & (uncertainty >= 0)
        object.__setattr__(self, "wavelength_A", wavelength)
        object.__setattr__(self, "flux_lambda", flux)
        object.__setattr__(self, "uncertainty_flux_lambda", uncertainty)
        object.__setattr__(self, "valid", valid)

    def to_observed_spectrum(
        self,
        *,
        allow_relative_flux: bool = False,
    ) -> ObservedSpectrum:
        """Convert valid rows to the ETC input type without hiding provenance."""

        if not self.provenance.flux_calibrated and not allow_relative_flux:
            raise ValueError(
                "This template is not flux calibrated. Normalize it to a physical "
                "magnitude before an ETC count-rate calculation."
            )
        if np.count_nonzero(self.valid) < 3:
            raise ValueError("Fewer than three valid spectral samples remain.")
        metadata: Mapping[str, object] = {
            "template_key": self.provenance.key,
            "object_name": self.provenance.object_name,
            "source_class": self.provenance.source_class,
            "state": self.provenance.state,
            "epoch_mjd": self.provenance.epoch_mjd,
            "phase_days": self.provenance.phase_days,
            "wavelength_frame": self.provenance.wavelength_frame,
            "flux_unit": self.provenance.flux_unit,
            "citation": self.provenance.citation,
            "source_url": self.provenance.source_url,
            "template_notes": self.provenance.notes,
        }
        return ObservedSpectrum(
            self.wavelength_A[self.valid],
            self.flux_lambda[self.valid],
            name=f"{self.provenance.object_name}, {self.provenance.state}",
            uncertainty_flux_lambda=self.uncertainty_flux_lambda[self.valid],
            native_resolving_power=self.provenance.native_resolving_power,
            metadata=metadata,
        )


def mask_wavelength_ranges(
    spectrum: CalibratedSpectrum,
    ranges_A: tuple[tuple[float, float], ...],
    *,
    reason: str,
) -> CalibratedSpectrum:
    """Exclude known unusable intervals while retaining an audit trail."""

    valid = np.array(spectrum.valid, copy=True)
    labels: list[str] = []
    for lower, upper in ranges_A:
        if lower <= 0 or upper <= lower:
            raise ValueError("Mask ranges must be positive and increasing.")
        valid &= ~(
            (spectrum.wavelength_A >= lower)
            & (spectrum.wavelength_A <= upper)
        )
        labels.append(f"{lower:g}--{upper:g} Angstrom")
    notes = spectrum.provenance.notes
    if notes:
        notes += " "
    notes += f"Masked {', '.join(labels)} because {reason}."
    return replace(
        spectrum,
        valid=valid,
        provenance=replace(spectrum.provenance, notes=notes),
    )


def combine_calibrated_spectra(
    spectra: tuple[CalibratedSpectrum, ...],
    *,
    joins_A: tuple[float, ...],
    key: str,
    state: str,
) -> CalibratedSpectrum:
    """Join calibrated arms at explicit wavelengths without interpolation."""

    if len(spectra) < 2 or len(joins_A) != len(spectra) - 1:
        raise ValueError("N spectra require N-1 join wavelengths.")
    if np.any(np.diff(np.asarray(joins_A, dtype=float)) <= 0):
        raise ValueError("Join wavelengths must be strictly increasing.")
    first = spectra[0].provenance
    if any(item.provenance.object_name != first.object_name for item in spectra):
        raise ValueError("Only spectra of the same object can be joined.")
    wavelength_parts: list[np.ndarray] = []
    flux_parts: list[np.ndarray] = []
    uncertainty_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    boundaries = (-np.inf, *joins_A, np.inf)
    for index, item in enumerate(spectra):
        selected = (
            (item.wavelength_A >= boundaries[index])
            & (item.wavelength_A < boundaries[index + 1])
        )
        if np.count_nonzero(selected) < 2:
            raise ValueError("A requested arm contributes fewer than two rows.")
        wavelength_parts.append(item.wavelength_A[selected])
        flux_parts.append(item.flux_lambda[selected])
        uncertainty_parts.append(item.uncertainty_flux_lambda[selected])
        valid_parts.append(item.valid[selected])
    wavelength = np.concatenate(wavelength_parts)
    if np.any(np.diff(wavelength) <= 0):
        raise ValueError("Joined wavelengths are not strictly increasing.")
    provenance = SpectrumProvenance(
        key=key,
        object_name=first.object_name,
        source_class=first.source_class,
        state=state,
        epoch_mjd=first.epoch_mjd,
        phase_days=first.phase_days,
        wavelength_frame=first.wavelength_frame,
        flux_unit=first.flux_unit,
        flux_calibrated=all(item.provenance.flux_calibrated for item in spectra),
        native_resolving_power=min(
            item.provenance.native_resolving_power for item in spectra
        ),
        citation=first.citation,
        source_url=first.source_url,
        notes=(
            "Arms joined at "
            + ", ".join(f"{item:g} Angstrom" for item in joins_A)
            + ". The conservative minimum arm resolving power is recorded."
        ),
    )
    return CalibratedSpectrum(
        wavelength,
        np.concatenate(flux_parts),
        np.concatenate(uncertainty_parts),
        np.concatenate(valid_parts),
        provenance,
    )


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_sha256: str | None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_sha256 is None or _sha256(destination) == expected_sha256:
            return destination
        raise ValueError(f"Checksum mismatch for existing file {destination}.")
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urlopen(url) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    if expected_sha256 is not None and _sha256(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch after downloading {url}.")
    temporary.replace(destination)
    return destination


def load_engrave_xshooter_arm(
    path: str | Path,
    *,
    arm: str,
    epoch_mjd: float,
    phase_days: float,
) -> CalibratedSpectrum:
    """Load one unsmoothed, photometrically calibrated AT2017gfo arm."""

    arm_name = arm.upper()
    resolving_power = {"UVB": 5100.0, "VIS": 8800.0, "NIR": 5600.0}
    if arm_name not in resolving_power:
        raise ValueError("X-shooter arm must be UVB, VIS, or NIR.")
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError("ENGRAVE arm file must contain four numeric columns.")
    # The release labels column four "variance/error". Its scale and the
    # accompanying README show that it is the 1-sigma flux error.
    uncertainty = np.abs(data[:, 3])
    valid = (
        np.isfinite(data[:, 1])
        & np.isfinite(uncertainty)
        & (uncertainty > 0)
    )
    provenance = SpectrumProvenance(
        key=f"at2017gfo-{phase_days:.2f}d-{arm_name.lower()}",
        object_name="AT2017gfo",
        source_class="kilonova",
        state=f"{phase_days:+.2f} d after GW170817",
        epoch_mjd=float(epoch_mjd),
        phase_days=float(phase_days),
        wavelength_frame="observed",
        flux_unit="erg s^-1 cm^-2 Angstrom^-1",
        flux_calibrated=True,
        native_resolving_power=resolving_power[arm_name],
        citation=(
            "Pian et al. 2017, Nature, 551, 67; "
            "Smartt et al. 2017, Nature, 551, 75"
        ),
        source_url="https://www.engrave-eso.org/AT2017gfo-Data-Release/",
        notes=(
            "ENGRAVE v1.0 unsmoothed X-shooter arm, telluric corrected and "
            "scaled to published photometry."
        ),
    )
    return CalibratedSpectrum(
        data[:, 0],
        data[:, 1],
        uncertainty,
        valid,
        provenance,
    )


def fetch_engrave_xshooter_arm(
    cache_dir: str | Path,
    *,
    arm: str,
    epoch_mjd: float = 57983.969,
    phase_days: float = 1.43,
) -> CalibratedSpectrum:
    """Download the ENGRAVE archive once and extract a selected X-shooter arm."""

    cache = Path(cache_dir)
    archive = _download(
        AT2017GFO_ARCHIVE_URL,
        cache / "AT2017gfo.zip",
        AT2017GFO_ARCHIVE_SHA256,
    )
    arm_name = arm.upper()
    member_name = (
        "AT2017gfo/flux_corrected_unsmoothed_spectra/"
        f"{arm_name}_AT2017gfo_ENGRAVE_v1.0_XSHOOTER_"
        f"MJD-{epoch_mjd:.3f}_Phase+{phase_days:.2f}d.dat"
    )
    output = cache / Path(member_name).name
    if not output.exists():
        with zipfile.ZipFile(archive) as bundle:
            try:
                payload = bundle.read(member_name)
            except KeyError as exc:
                raise KeyError(
                    f"No ENGRAVE spectrum for MJD {epoch_mjd:.3f}, "
                    f"phase {phase_days:.2f} d, arm {arm_name}."
                ) from exc
        output.write_bytes(payload)
    return load_engrave_xshooter_arm(
        output,
        arm=arm_name,
        epoch_mjd=epoch_mjd,
        phase_days=phase_days,
    )


def load_lamost_ugem_outburst(path: str | Path) -> CalibratedSpectrum:
    """Load the public LAMOST DR11 U Gem outburst-peak spectrum."""

    from astropy.io import fits

    with fits.open(path) as hdul:
        header = hdul[0].header
        table = hdul["COADD"].data[0]
        wavelength = np.asarray(table["WAVELENGTH"], dtype=float)
        flux_raw = np.asarray(table["FLUX"], dtype=float)
        ivar_raw = np.asarray(table["IVAR"], dtype=float)
        and_mask = np.asarray(table["ANDMASK"])
        or_mask = np.asarray(table["ORMASK"])
    scale = 1.0e-17
    uncertainty = np.full_like(flux_raw, np.inf)
    good_ivar = np.isfinite(ivar_raw) & (ivar_raw > 0)
    uncertainty[good_ivar] = scale / np.sqrt(ivar_raw[good_ivar])
    flux = scale * flux_raw
    valid = (
        good_ivar
        & np.isfinite(flux)
        & (and_mask == 0)
        & (or_mask == 0)
    )
    # ORMASK is intentionally conservative. If it removes every row in an
    # older release, retain rows that pass the stronger ANDMASK criterion.
    if np.count_nonzero(valid) < 3:
        valid = good_ivar & np.isfinite(flux) & (and_mask == 0)
    provenance = SpectrumProvenance(
        key="ugem-lamost-dr11-outburst",
        object_name="U Gem",
        source_class="dwarf nova",
        state="2012 December outburst peak",
        epoch_mjd=float(header.get("MJD", 56265.0)),
        phase_days=None,
        wavelength_frame="observed",
        flux_unit="erg s^-1 cm^-2 Angstrom^-1",
        flux_calibrated=True,
        native_resolving_power=1800.0,
        citation=(
            "Zhi et al. 2020, ApJS, 250, 2; "
            "LAMOST DR11 low-resolution spectrum 89907231"
        ),
        source_url=UGEM_LAMOST_URL,
        notes=(
            "Vacuum wavelength and heliocentric correction are recorded in "
            "the DR11 FITS header. Flux and inverse variance use LAMOST's "
            "10^-17 cgs flux-density convention."
        ),
    )
    return CalibratedSpectrum(
        wavelength,
        flux,
        uncertainty,
        valid,
        provenance,
    )


def fetch_lamost_ugem_outburst(cache_dir: str | Path) -> CalibratedSpectrum:
    """Download and load the checksum-pinned LAMOST U Gem spectrum."""

    path = _download(
        UGEM_LAMOST_URL,
        Path(cache_dir) / "ugem_lamost_dr11_obsid89907231.fits.gz",
        UGEM_LAMOST_SHA256,
    )
    return load_lamost_ugem_outburst(path)


def load_cds_ugem_phase_spectrum(
    path: str | Path,
    *,
    bjd: float,
    orbital_phase: float,
) -> CalibratedSpectrum:
    """Load one phase-resolved U Gem spectrum from Naylor et al. (2005)."""

    data = np.loadtxt(path, skiprows=3)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("CDS U Gem spectrum must contain wavelength, flux, error.")
    valid = (
        np.isfinite(data[:, 1])
        & np.isfinite(data[:, 2])
        & (data[:, 2] > 0)
        & (data[:, 1] != 0)
    )
    provenance = SpectrumProvenance(
        key=f"ugem-cds-phase-{orbital_phase:.4f}",
        object_name="U Gem",
        source_class="dwarf nova",
        state=f"quiescent orbital phase {orbital_phase:.4f}",
        epoch_mjd=float(bjd - 2400000.5),
        phase_days=None,
        wavelength_frame="observed",
        flux_unit="relative detector flux",
        flux_calibrated=False,
        native_resolving_power=5100.0,
        citation="Naylor, Allan & Long 2005, MNRAS, 361, 1091",
        source_url=(
            "https://cdsarc.cds.unistra.fr/ftp/J/MNRAS/361/1091/"
        ),
        notes=(
            "Phase-resolved 7480--8320 Angstrom spectrum. It constrains donor "
            "radial velocity and phase smearing, not absolute count rate."
        ),
    )
    return CalibratedSpectrum(data[:, 0], data[:, 1], data[:, 2], valid, provenance)
