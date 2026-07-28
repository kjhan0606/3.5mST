"""Template-based slitless spectroscopy calculations.

This module predicts an extracted, detector-sampled spectrum.  It is intended
for compact objects and transients whose science depends on a line profile or
on a broad spectral shape rather than on one isolated line flux.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import slitless_etc as legacy

from .config import SpectralChannel, build_instrument, get_channel
from .results import (
    BinnedSpectrum,
    DoublePeakMeasurement,
    LineMeasurement,
    SpectralEtcResult,
)


_C_KMS = 299792.458


@dataclass(frozen=True)
class ObservedSpectrum:
    """An observed-frame physical spectrum in cgs flux-density units."""

    wavelength_A: np.ndarray
    flux_lambda: np.ndarray
    name: str = "input spectrum"
    uncertainty_flux_lambda: np.ndarray | None = None
    native_resolving_power: float | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        wavelength = np.asarray(self.wavelength_A, dtype=float)
        flux = np.asarray(self.flux_lambda, dtype=float)
        if wavelength.ndim != 1 or flux.ndim != 1:
            raise ValueError("Wavelength and flux arrays must be one-dimensional.")
        if wavelength.size != flux.size or wavelength.size < 3:
            raise ValueError("Wavelength and flux arrays must have the same length >= 3.")
        if not np.all(np.isfinite(wavelength)) or not np.all(np.isfinite(flux)):
            raise ValueError("A spectrum cannot contain NaN or infinite values.")
        if np.any(wavelength <= 0) or np.any(np.diff(wavelength) <= 0):
            raise ValueError("Wavelengths must be positive and strictly increasing.")
        uncertainty = self.uncertainty_flux_lambda
        if uncertainty is not None:
            uncertainty = np.asarray(uncertainty, dtype=float)
            if uncertainty.shape != flux.shape:
                raise ValueError("Flux uncertainty must match the flux array.")
            if np.any(~np.isfinite(uncertainty)) or np.any(uncertainty < 0):
                raise ValueError("Flux uncertainty must be finite and non-negative.")
        native_r = self.native_resolving_power
        if native_r is not None and native_r <= 0:
            raise ValueError("Native resolving power must be positive.")
        object.__setattr__(self, "wavelength_A", wavelength)
        object.__setattr__(self, "flux_lambda", flux)
        object.__setattr__(self, "uncertainty_flux_lambda", uncertainty)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def from_ascii(
        cls,
        path: str | Path,
        wavelength_column: int = 0,
        flux_column: int = 1,
        uncertainty_column: int | None = None,
        delimiter: str | None = None,
        skiprows: int = 0,
        name: str | None = None,
        native_resolving_power: float | None = None,
    ) -> "ObservedSpectrum":
        """Read wavelength [Angstrom] and F_lambda [cgs/A] from a text table."""

        input_path = Path(path)
        data = np.loadtxt(input_path, delimiter=delimiter, skiprows=skiprows)
        if data.ndim != 2:
            raise ValueError("The input spectrum must be a table with at least two columns.")
        return cls(
            data[:, wavelength_column],
            data[:, flux_column],
            name=name or input_path.stem,
            uncertainty_flux_lambda=(
                None
                if uncertainty_column is None
                else data[:, uncertainty_column]
            ),
            native_resolving_power=native_resolving_power,
        )

    def scaled_to_ab_magnitude(
        self,
        magnitude_ab: float,
        band_min_A: float,
        band_max_A: float,
    ) -> "ObservedSpectrum":
        """Scale the spectrum to a mean AB magnitude over a rectangular band."""

        mask = (
            (self.wavelength_A >= band_min_A)
            & (self.wavelength_A <= band_max_A)
        )
        if np.count_nonzero(mask) < 2:
            raise ValueError("The normalization band does not overlap the spectrum.")
        wavelength = self.wavelength_A[mask]
        fnu = self.flux_lambda[mask] * wavelength**2 / legacy.C_A
        mean_fnu = np.trapezoid(fnu, wavelength) / (wavelength[-1] - wavelength[0])
        if not np.isfinite(mean_fnu) or mean_fnu <= 0:
            raise ValueError("The normalization band must have positive mean flux.")
        target_fnu = 10.0 ** (-0.4 * (float(magnitude_ab) + 48.60))
        return ObservedSpectrum(
            self.wavelength_A,
            self.flux_lambda * target_fnu / mean_fnu,
            name=f"{self.name}, AB={magnitude_ab:g}",
            uncertainty_flux_lambda=(
                None
                if self.uncertainty_flux_lambda is None
                else self.uncertainty_flux_lambda * target_fnu / mean_fnu
            ),
            native_resolving_power=self.native_resolving_power,
            metadata=self.metadata,
        )


def _detector_grid(
    channel: SpectralChannel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return constant-resolving-power detector-bin centers and widths."""

    dlog = 1.0 / (channel.resolving_power * channel.sampling_pix)
    n_pixels = int(np.ceil(np.log(channel.band_max_A / channel.band_min_A) / dlog))
    edges = channel.band_min_A * np.exp(np.arange(n_pixels + 1) * dlog)
    edges[-1] = channel.band_max_A
    centers = np.sqrt(edges[:-1] * edges[1:])
    return centers, np.diff(edges), edges


def _sample_and_convolve(
    spectrum: ObservedSpectrum,
    channel: SpectralChannel,
    intrinsic_fwhm_arcsec: float,
    pixel_scale_arcsec: float,
    oversample: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float | str]]:
    """Flux-conserving convolution without applying the input LSF twice."""

    centers, widths, edges = _detector_grid(channel)
    n_pixels = centers.size
    fractions = (np.arange(oversample) + 0.5) / oversample
    work_log = (
        np.log(edges[:-1])[:, None]
        + fractions[None, :]
        * (np.log(edges[1:]) - np.log(edges[:-1]))[:, None]
    )
    work_wavelength = np.exp(work_log).ravel()
    work_flux = np.interp(
        work_wavelength,
        spectrum.wavelength_A,
        spectrum.flux_lambda,
        left=np.nan,
        right=np.nan,
    )
    if np.any(~np.isfinite(work_flux)):
        raise ValueError(
            "The input spectrum does not cover the full selected channel. "
            "Provide measured or physically modelled flux over the complete band."
        )

    native_r = spectrum.native_resolving_power
    if native_r is None:
        spectral_kernel_fwhm_pix = channel.sampling_pix
        effective_point_source_r = channel.resolving_power
        resolution_status = "input treated as unresolved relative to mission LSF"
    elif native_r > channel.resolving_power:
        input_fwhm_pix = (
            channel.sampling_pix * channel.resolving_power / native_r
        )
        spectral_kernel_fwhm_pix = np.sqrt(
            max(0.0, channel.sampling_pix**2 - input_fwhm_pix**2)
        )
        effective_point_source_r = channel.resolving_power
        resolution_status = "native and mission Gaussian LSFs combined in quadrature"
    else:
        spectral_kernel_fwhm_pix = 0.0
        effective_point_source_r = native_r
        resolution_status = (
            "input resolution is lower than the mission setting; no sharpening applied"
        )

    morphology_fwhm_pix = max(0.0, intrinsic_fwhm_arcsec) / pixel_scale_arcsec
    lsf_fwhm_detector_pix = np.hypot(
        spectral_kernel_fwhm_pix,
        morphology_fwhm_pix,
    )
    sigma_work_pix = lsf_fwhm_detector_pix * oversample / 2.354820045
    if sigma_work_pix > 0:
        convolved_work = gaussian_filter1d(
            work_flux,
            sigma=sigma_work_pix,
            mode="nearest",
            truncate=5.0,
        )
    else:
        convolved_work = work_flux
    convolved = np.mean(convolved_work.reshape(n_pixels, oversample), axis=1)
    raw = np.interp(centers, spectrum.wavelength_A, spectrum.flux_lambda)
    morphology_r = (
        float("inf")
        if morphology_fwhm_pix == 0
        else channel.resolving_power * channel.sampling_pix / morphology_fwhm_pix
    )
    effective_r = 1.0 / np.sqrt(
        effective_point_source_r**-2 + morphology_r**-2
    )
    resolution_metadata: dict[str, float | str] = {
        "native_template_resolving_power": (
            float("inf") if native_r is None else float(native_r)
        ),
        "spectral_kernel_fwhm_detector_pix": float(spectral_kernel_fwhm_pix),
        "morphology_fwhm_detector_pix": float(morphology_fwhm_pix),
        "effective_resolving_power": float(effective_r),
        "resolution_status": resolution_status,
    }
    return centers, widths, raw, convolved, resolution_metadata


def simulate_spectrum(
    spectrum: ObservedSpectrum,
    channel: SpectralChannel | str,
    exposure_s: float,
    source_fwhm_arcsec: float = 0.0,
    extraction_height_fwhm: float = 1.5,
    contamination_flux_fraction: float = 0.0,
    contamination_model_fraction: float = 0.10,
    **instrument_overrides: float | int | bool | str | None,
) -> SpectralEtcResult:
    """Predict a detector-sampled extracted spectrum and its noise budget.

    ``source_fwhm_arcsec`` is the intrinsic angular FWHM, before convolution
    with the delivered PSF.  Point-like compact objects should use zero.
    ``contamination_flux_fraction`` is the contaminating count rate relative
    to the target.  Its Poisson noise is always included.  The second fraction
    describes the residual after contamination modelling.
    """

    selected = get_channel(channel) if isinstance(channel, str) else channel
    if exposure_s <= 0:
        raise ValueError("Exposure time must be positive.")
    if source_fwhm_arcsec < 0 or extraction_height_fwhm <= 0:
        raise ValueError("Source size must be non-negative and extraction height positive.")
    if contamination_flux_fraction < 0:
        raise ValueError("Contamination flux fraction cannot be negative.")
    if not 0 <= contamination_model_fraction <= 1:
        raise ValueError("Contamination model fraction must lie in [0, 1].")

    cfg = build_instrument(selected, **instrument_overrides)
    wavelength, width, input_flux, convolved_flux, resolution_metadata = (
        _sample_and_convolve(
        spectrum,
        selected,
        source_fwhm_arcsec,
        cfg.pix_scale,
        )
    )

    throughput = cfg.throughput(wavelength)
    photon_factor = wavelength * 1e-8 / (legacy.H * legacy.C)
    physical_flux = np.clip(convolved_flux, 0.0, None)
    source_e = (
        physical_flux
        * width
        * cfg.area_cm2
        * throughput
        * photon_factor
        * exposure_s
        * cfg.extraction_eff
    )

    delivered_fwhm = np.hypot(source_fwhm_arcsec, cfg.psf_fwhm(wavelength))
    spatial_pixels = np.maximum(
        1.0,
        extraction_height_fwhm * delivered_fwhm / cfg.pix_scale,
    )
    sky_rate_per_pixel = legacy.background_per_pixel(
        cfg,
        (selected.band_min_A, selected.band_max_A),
    )
    sky_e = sky_rate_per_pixel * spatial_pixels * exposure_s
    dark_rate = np.array([cfg.detector_at(float(item))[1] for item in wavelength])
    dark_e = dark_rate * spatial_pixels * exposure_s
    read_variance = np.array(
        [
            cfg.read_noise_variance_total(float(item), exposure_s, float(n_pixels))
            for item, n_pixels in zip(wavelength, spatial_pixels)
        ]
    )

    contamination_e = contamination_flux_fraction * source_e
    shot_variance = cfg.ramp_shot_noise_factor() * (
        source_e + sky_e + dark_e + contamination_e
    )
    systematic_variance = (
        (selected.relative_flux_floor * source_e) ** 2
        + (cfg.flat_error * source_e) ** 2
        + (contamination_model_fraction * contamination_e) ** 2
    )
    total_variance = shot_variance + read_variance + systematic_variance
    snr = np.divide(
        source_e,
        np.sqrt(total_variance),
        out=np.zeros_like(source_e),
        where=total_variance > 0,
    )

    n_exposures = cfg.n_reads(exposure_s)
    peak_source_e_per_exposure = source_e / n_exposures
    peak_background_e_per_exposure = (
        np.divide(
            sky_e + dark_e,
            spatial_pixels,
            out=np.zeros_like(sky_e),
            where=spatial_pixels > 0,
        )
        / n_exposures
    )
    maximum_pixel_e = float(
        np.max(peak_source_e_per_exposure + peak_background_e_per_exposure)
    )
    template_metadata = dict(spectrum.metadata or {})
    return SpectralEtcResult(
        wavelength_A=wavelength,
        pixel_width_A=width,
        input_flux_lambda=input_flux,
        convolved_flux_lambda=convolved_flux,
        source_e=source_e,
        sky_e=sky_e,
        dark_e=dark_e,
        contamination_e=contamination_e,
        read_variance_e2=read_variance,
        systematic_variance_e2=systematic_variance,
        total_variance_e2=total_variance,
        snr=snr,
        metadata={
            "mode": "template",
            "source_name": spectrum.name,
            "channel": selected.key,
            "channel_label": selected.label,
            "exposure_s": float(exposure_s),
            "resolving_power": selected.resolving_power,
            "sampling_pix": selected.sampling_pix,
            "band_min_A": selected.band_min_A,
            "band_max_A": selected.band_max_A,
            "sky_e_s_detector_pixel": float(sky_rate_per_pixel),
            "source_fwhm_arcsec": float(source_fwhm_arcsec),
            "extraction_height_fwhm": float(extraction_height_fwhm),
            "contamination_flux_fraction": float(contamination_flux_fraction),
            "contamination_model_fraction": float(contamination_model_fraction),
            "relative_flux_floor": float(selected.relative_flux_floor),
            "wavelength_calibration_floor_kms": (
                selected.wavelength_calibration_floor_kms
            ),
            "negative_input_fraction": float(np.mean(convolved_flux < 0)),
            "maximum_pixel_e_per_exposure_upper_bound": maximum_pixel_e,
            "full_well_e": float(cfg.full_well),
            "saturated_upper_bound": bool(maximum_pixel_e >= cfg.full_well),
            "template_uncertainty_available": (
                spectrum.uncertainty_flux_lambda is not None
            ),
            "calibration_status": "proposal requirement, not measured hardware",
            **resolution_metadata,
            **template_metadata,
        },
    )


def bin_spectrum(
    result: SpectralEtcResult,
    target_resolving_power: float,
) -> BinnedSpectrum:
    """Bin a native result to a lower constant resolving power."""

    native_r = float(result.metadata["resolving_power"])
    if target_resolving_power <= 0 or target_resolving_power > native_r:
        raise ValueError("Target resolving power must be positive and <= native R.")
    dlog = 1.0 / target_resolving_power
    log_min = np.log(result.wavelength_A[0])
    log_max = np.log(result.wavelength_A[-1])
    edges = np.arange(log_min, log_max + dlog, dlog)
    if edges[-1] <= log_max:
        edges = np.append(edges, log_max + np.finfo(float).eps)
    groups = np.digitize(np.log(result.wavelength_A), edges) - 1

    wavelength_out: list[float] = []
    source_out: list[float] = []
    variance_out: list[float] = []
    for group in range(groups.min(), groups.max() + 1):
        mask = groups == group
        if not np.any(mask):
            continue
        weights = result.pixel_width_A[mask]
        wavelength_out.append(float(np.average(result.wavelength_A[mask], weights=weights)))
        source_out.append(float(np.sum(result.source_e[mask])))
        variance_out.append(result.summed_variance(mask))
    source_array = np.asarray(source_out)
    variance_array = np.asarray(variance_out)
    snr = np.divide(
        source_array,
        np.sqrt(variance_array),
        out=np.zeros_like(source_array),
        where=variance_array > 0,
    )
    return BinnedSpectrum(
        wavelength_A=np.asarray(wavelength_out),
        source_e=source_array,
        variance_e2=variance_array,
        snr=snr,
        target_resolving_power=float(target_resolving_power),
    )


def measure_emission_line(
    result: SpectralEtcResult,
    line_center_A: float,
    window_kms: float = 1500.0,
) -> LineMeasurement:
    """Measure predicted line S/N, centroid precision, and second moment."""

    if line_center_A <= 0 or window_kms <= 0:
        raise ValueError("Line center and velocity window must be positive.")
    velocity = _C_KMS * (result.wavelength_A / line_center_A - 1.0)
    line_mask = np.abs(velocity) <= window_kms
    side_mask = (
        (np.abs(velocity) >= 1.2 * window_kms)
        & (np.abs(velocity) <= 2.0 * window_kms)
    )
    if np.count_nonzero(line_mask) < 3 or np.count_nonzero(side_mask) < 2:
        raise ValueError("The selected line or its continuum sidebands leave the channel.")

    continuum_coefficients = np.polyfit(
        result.wavelength_A[side_mask],
        result.convolved_flux_lambda[side_mask],
        deg=1,
    )
    continuum = np.polyval(continuum_coefficients, result.wavelength_A[line_mask])
    residual = np.clip(
        result.convolved_flux_lambda[line_mask] - continuum,
        0.0,
        None,
    )
    line_weights = residual * result.pixel_width_A[line_mask]
    line_flux = float(np.sum(line_weights))
    if line_flux <= 0:
        raise ValueError("No positive emission-line flux is present in the selected window.")

    total_flux = np.clip(result.convolved_flux_lambda[line_mask], 0.0, None)
    line_fraction = np.divide(
        residual,
        total_flux,
        out=np.zeros_like(residual),
        where=total_flux > 0,
    )
    line_e = result.source_e[line_mask] * line_fraction
    variance = result.total_variance_e2[line_mask]
    line_signal_e = float(np.sum(line_e))
    relative_floor = float(result.metadata.get("relative_flux_floor", 0.0))
    contamination_floor = float(
        result.metadata.get("contamination_model_fraction", 0.0)
    )
    source_systematic = relative_floor * result.source_e[line_mask]
    contamination_systematic = (
        contamination_floor * result.contamination_e[line_mask]
    )
    statistical_variance = np.maximum(
        0.0,
        variance - source_systematic**2 - contamination_systematic**2,
    )
    line_variance_e2 = float(
        np.sum(statistical_variance)
        + (relative_floor * line_signal_e) ** 2
        + np.sum(contamination_systematic) ** 2
    )
    line_snr = line_signal_e / np.sqrt(line_variance_e2)

    wavelength = result.wavelength_A[line_mask]
    line_velocity = velocity[line_mask]
    centroid = float(np.average(wavelength, weights=line_weights))
    centroid_velocity = _C_KMS * (centroid / line_center_A - 1.0)
    second_moment = float(
        np.sqrt(np.average((line_velocity - centroid_velocity) ** 2, weights=line_weights))
    )

    derivative = np.gradient(line_e, line_velocity)
    fisher = float(
        np.sum(
            derivative**2
            / np.maximum(statistical_variance, np.finfo(float).tiny)
        )
    )
    velocity_sigma_stat = 1.0 / np.sqrt(fisher) if fisher > 0 else float("inf")
    wavecal_floor = float(
        result.metadata.get("wavelength_calibration_floor_kms", 0.0)
    )
    centroid_sigma = float(np.hypot(velocity_sigma_stat, wavecal_floor))
    return LineMeasurement(
        line_center_A=float(line_center_A),
        window_kms=float(window_kms),
        line_flux_erg_s_cm2=line_flux,
        line_snr=float(line_snr),
        centroid_A=centroid,
        centroid_sigma_kms=centroid_sigma,
        velocity_sigma_stat_kms=float(velocity_sigma_stat),
        wavelength_calibration_floor_kms=wavecal_floor,
        second_moment_kms=second_moment,
        n_pixels=int(np.count_nonzero(line_mask)),
    )


def measure_double_peak(
    result: SpectralEtcResult,
    line_center_A: float,
    window_kms: float = 1500.0,
    minimum_prominence_snr: float = 3.0,
) -> DoublePeakMeasurement:
    """Recover one blue and one red peak from a continuum-subtracted line."""

    if minimum_prominence_snr <= 0:
        raise ValueError("Minimum peak prominence S/N must be positive.")
    velocity = _C_KMS * (result.wavelength_A / line_center_A - 1.0)
    line_mask = np.abs(velocity) <= window_kms
    side_mask = (
        (np.abs(velocity) >= 1.2 * window_kms)
        & (np.abs(velocity) <= 2.0 * window_kms)
    )
    if np.count_nonzero(line_mask) < 7 or np.count_nonzero(side_mask) < 2:
        raise ValueError("The line window or continuum sidebands leave the channel.")
    continuum_coefficients = np.polyfit(
        result.wavelength_A[side_mask],
        result.convolved_flux_lambda[side_mask],
        deg=1,
    )
    line_wavelength = result.wavelength_A[line_mask]
    line_velocity = velocity[line_mask]
    residual = result.convolved_flux_lambda[line_mask] - np.polyval(
        continuum_coefficients,
        line_wavelength,
    )
    electron_per_flux = np.divide(
        result.source_e[line_mask],
        np.clip(result.convolved_flux_lambda[line_mask], 0.0, None),
        out=np.full_like(residual, np.nan),
        where=result.convolved_flux_lambda[line_mask] > 0,
    )
    sigma_flux = np.divide(
        np.sqrt(result.total_variance_e2[line_mask]),
        electron_per_flux,
        out=np.full_like(residual, np.nan),
        where=np.isfinite(electron_per_flux) & (electron_per_flux > 0),
    )
    finite_sigma = sigma_flux[np.isfinite(sigma_flux) & (sigma_flux > 0)]
    if finite_sigma.size == 0:
        raise ValueError("Cannot derive a flux uncertainty for peak recovery.")
    representative_sigma = float(np.median(finite_sigma))
    peaks, properties = find_peaks(
        residual,
        prominence=minimum_prominence_snr * representative_sigma,
    )
    if peaks.size < 2:
        raise ValueError("Fewer than two significant line peaks were recovered.")
    blue_candidates = np.flatnonzero(line_velocity[peaks] < 0)
    red_candidates = np.flatnonzero(line_velocity[peaks] > 0)
    if blue_candidates.size == 0 or red_candidates.size == 0:
        raise ValueError("Significant peaks were not recovered on both sides of the line.")
    prominences = properties["prominences"]
    blue_property_index = blue_candidates[np.argmax(prominences[blue_candidates])]
    red_property_index = red_candidates[np.argmax(prominences[red_candidates])]
    blue_index = peaks[blue_property_index]
    red_index = peaks[red_property_index]
    blue_velocity = float(line_velocity[blue_index])
    red_velocity = float(line_velocity[red_index])
    return DoublePeakMeasurement(
        line_center_A=float(line_center_A),
        blue_peak_A=float(line_wavelength[blue_index]),
        red_peak_A=float(line_wavelength[red_index]),
        separation_kms=red_velocity - blue_velocity,
        blue_velocity_kms=blue_velocity,
        red_velocity_kms=red_velocity,
        blue_prominence_snr=float(
            prominences[blue_property_index] / representative_sigma
        ),
        red_prominence_snr=float(
            prominences[red_property_index] / representative_sigma
        ),
    )


def phase_smearing(
    period_s: float,
    exposure_s: float,
    radial_velocity_semiamplitude_kms: float,
) -> dict[str, float]:
    """Bound orbital radial-velocity smearing during one exposure."""

    if period_s <= 0 or exposure_s <= 0 or radial_velocity_semiamplitude_kms < 0:
        raise ValueError("Period and exposure must be positive and amplitude non-negative.")
    phase_span = min(exposure_s / period_s, 1.0)
    peak_to_peak_kms = (
        2.0 * radial_velocity_semiamplitude_kms
        if phase_span >= 0.5
        else 2.0
        * radial_velocity_semiamplitude_kms
        * np.sin(np.pi * phase_span)
    )
    return {
        "phase_fraction": exposure_s / period_s,
        "maximum_velocity_span_kms": float(abs(peak_to_peak_kms)),
        "uniform_span_sigma_kms": float(abs(peak_to_peak_kms) / np.sqrt(12.0)),
    }


def maximum_phase_resolved_exposure(
    period_s: float,
    maximum_phase_fraction: float = 0.10,
) -> float:
    """Exposure ceiling set by a requested fraction of an orbital period."""

    if period_s <= 0 or not 0 < maximum_phase_fraction <= 1:
        raise ValueError("Period must be positive and phase fraction in (0, 1].")
    return float(period_s * maximum_phase_fraction)
