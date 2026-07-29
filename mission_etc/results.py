"""Typed results returned by mission ETC calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ImagingEtcResult:
    """Broadband optical or near-infrared point-source result."""

    filter_name: str
    pivot_A: float
    width_A: float
    exposure_s: float
    limiting_magnitude_ab: float
    input_magnitude_ab: float | None
    input_snr: float | None
    saturation_magnitude_ab: float
    source_e_s: float | None
    sky_e_s_pixel: float
    aperture_pixels: float


@dataclass(frozen=True)
class IsolatedLineEtcResult:
    """One unresolved emission-line calculation."""

    channel: str
    wavelength_A: float
    exposure_s: float
    target_snr: float
    limiting_flux_erg_s_cm2: float
    input_flux_erg_s_cm2: float | None
    input_snr: float | None
    footprint_pixels: float
    sky_e: float
    dark_e: float
    read_variance_e2: float


@dataclass(frozen=True)
class MirImagingEtcResult:
    """Cooled MIR imaging result in physical flux-density units."""

    channel: str
    background: str
    observatory: str
    band_min_um: float
    band_max_um: float
    exposure_s: float
    target_snr: float
    limiting_flux_uJy: float
    input_flux_uJy: float | None
    input_snr: float | None
    aperture_pixels: float
    sky_e: float
    dark_e: float
    read_variance_e2: float


@dataclass(frozen=True)
class LineMeasurement:
    """Predicted precision for one line in a simulated spectrum."""

    line_center_A: float
    window_kms: float
    line_flux_erg_s_cm2: float
    line_snr: float
    centroid_A: float
    centroid_sigma_kms: float
    velocity_sigma_stat_kms: float
    wavelength_calibration_floor_kms: float
    second_moment_kms: float
    n_pixels: int


@dataclass(frozen=True)
class DoublePeakMeasurement:
    """Recovered locations and separation of a two-peaked emission line."""

    line_center_A: float
    blue_peak_A: float
    red_peak_A: float
    separation_kms: float
    blue_velocity_kms: float
    red_velocity_kms: float
    blue_prominence_snr: float
    red_prominence_snr: float


@dataclass(frozen=True)
class ExposureTimeSolution:
    """Inverse ETC result for one explicitly defined science statistic."""

    metric: str
    target_value: float
    exposure_s: float
    achieved_value: float
    feasible: bool
    phase_ceiling_s: float | None
    saturated_upper_bound: bool
    iterations: int
    message: str


@dataclass(frozen=True)
class MonteCarloLineResult:
    """Bias, scatter, and recovery fraction from repeated noisy spectra."""

    n_trials: int
    n_recovered: int
    recovery_fraction: float
    expected_centroid_A: float
    centroid_bias_kms: float
    centroid_scatter_kms: float
    median_reported_centroid_sigma_kms: float
    expected_line_snr: float
    line_snr_median: float
    line_snr_scatter: float
    peak_separation_expected_kms: float | None
    peak_separation_bias_kms: float | None
    peak_separation_scatter_kms: float | None


@dataclass(frozen=True)
class BinnedSpectrum:
    """Counts and uncertainties after flux-conserving spectral binning."""

    wavelength_A: np.ndarray
    source_e: np.ndarray
    variance_e2: np.ndarray
    snr: np.ndarray
    target_resolving_power: float


@dataclass(frozen=True)
class SpectralEtcResult:
    """Detector-level expectation for one extracted point-source spectrum."""

    wavelength_A: np.ndarray
    pixel_width_A: np.ndarray
    input_flux_lambda: np.ndarray
    convolved_flux_lambda: np.ndarray
    source_e: np.ndarray
    sky_e: np.ndarray
    dark_e: np.ndarray
    contamination_e: np.ndarray
    read_variance_e2: np.ndarray
    systematic_variance_e2: np.ndarray
    total_variance_e2: np.ndarray
    snr: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def summed_variance(self, mask: np.ndarray) -> float:
        """Sum variance while preserving channel-correlated systematics."""

        selected = np.asarray(mask, dtype=bool)
        if selected.shape != self.wavelength_A.shape:
            raise ValueError("Variance mask must match the detector spectrum.")
        variance = float(np.sum(self.total_variance_e2[selected]))
        relative_floor = float(self.metadata.get("relative_flux_floor", 0.0))
        contamination_floor = float(
            self.metadata.get("contamination_model_fraction", 0.0)
        )
        source_systematic = relative_floor * self.source_e[selected]
        contamination_systematic = (
            contamination_floor * self.contamination_e[selected]
        )
        # Replace diagonalized per-pixel terms by one coherent channel term.
        variance -= float(
            np.sum(source_systematic**2) + np.sum(contamination_systematic**2)
        )
        variance += float(
            np.sum(source_systematic) ** 2
            + np.sum(contamination_systematic) ** 2
        )
        return max(0.0, variance)

    def integrated_snr(self, wavelength_min_A: float, wavelength_max_A: float) -> float:
        """Return S/N of the summed source counts in a wavelength interval."""

        mask = (
            (self.wavelength_A >= wavelength_min_A)
            & (self.wavelength_A <= wavelength_max_A)
        )
        if not np.any(mask):
            raise ValueError("The requested interval does not overlap the simulated spectrum.")
        signal = float(np.sum(self.source_e[mask]))
        variance = self.summed_variance(mask)
        return signal / np.sqrt(variance) if variance > 0 else float("inf")

    def to_csv(self, path: str | Path) -> Path:
        """Write the detector-level spectrum with explicit column names."""

        output = Path(path)
        columns = np.column_stack(
            (
                self.wavelength_A,
                self.pixel_width_A,
                self.input_flux_lambda,
                self.convolved_flux_lambda,
                self.source_e,
                self.sky_e,
                self.dark_e,
                self.contamination_e,
                self.read_variance_e2,
                self.systematic_variance_e2,
                self.total_variance_e2,
                self.snr,
            )
        )
        header = (
            "wavelength_A,pixel_width_A,input_flux_lambda,convolved_flux_lambda,"
            "source_e,sky_e,dark_e,contamination_e,read_variance_e2,"
            "systematic_variance_e2,total_variance_e2,snr"
        )
        np.savetxt(output, columns, delimiter=",", header=header, comments="")
        return output
