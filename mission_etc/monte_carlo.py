"""Repeated extracted-spectrum noise realizations for feasibility tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .results import MonteCarloLineResult, SpectralEtcResult
from .spectroscopy import measure_double_peak, measure_emission_line


_C_KMS = 299792.458


def realize_extracted_spectrum(
    result: SpectralEtcResult,
    rng: np.random.Generator,
) -> SpectralEtcResult:
    """Draw one background-subtracted extracted spectrum.

    Shot, dark, sky, and read terms are represented by their extracted
    Gaussian limit.  Relative calibration and contamination residuals are
    drawn once per channel, which preserves their correlated character.
    """

    relative_floor = float(result.metadata.get("relative_flux_floor", 0.0))
    contamination_floor = float(
        result.metadata.get("contamination_model_fraction", 0.0)
    )
    source_systematic = relative_floor * result.source_e
    contamination_systematic = contamination_floor * result.contamination_e
    statistical_variance = np.maximum(
        0.0,
        result.total_variance_e2
        - source_systematic**2
        - contamination_systematic**2,
    )
    statistical_noise = rng.normal(0.0, np.sqrt(statistical_variance))
    calibration_offset = rng.normal() * source_systematic
    contamination_offset = rng.normal() * contamination_systematic
    measured_source_e = (
        result.source_e
        + statistical_noise
        + calibration_offset
        + contamination_offset
    )
    electron_per_flux = np.divide(
        result.source_e,
        result.convolved_flux_lambda,
        out=np.zeros_like(result.source_e),
        where=result.convolved_flux_lambda > 0,
    )
    measured_flux = np.divide(
        measured_source_e,
        electron_per_flux,
        out=np.array(result.convolved_flux_lambda, copy=True),
        where=electron_per_flux > 0,
    )
    measured_snr = np.divide(
        measured_source_e,
        np.sqrt(result.total_variance_e2),
        out=np.zeros_like(measured_source_e),
        where=result.total_variance_e2 > 0,
    )
    metadata = dict(result.metadata)
    metadata["noise_realization"] = True
    return replace(
        result,
        convolved_flux_lambda=measured_flux,
        source_e=measured_source_e,
        snr=measured_snr,
        metadata=metadata,
    )


def monte_carlo_line_recovery(
    result: SpectralEtcResult,
    line_center_A: float,
    *,
    window_kms: float = 1500.0,
    n_trials: int = 500,
    random_seed: int | None = None,
    recover_double_peak: bool = False,
    minimum_peak_prominence_snr: float = 3.0,
) -> MonteCarloLineResult:
    """Measure line bias and scatter over repeated detector realizations."""

    if n_trials < 2:
        raise ValueError("At least two Monte Carlo trials are required.")
    expected = measure_emission_line(
        result,
        line_center_A,
        window_kms=window_kms,
    )
    expected_peak_separation: float | None = None
    if recover_double_peak:
        try:
            expected_peak_separation = measure_double_peak(
                result,
                line_center_A,
                window_kms=window_kms,
                minimum_prominence_snr=minimum_peak_prominence_snr,
            ).separation_kms
        except ValueError:
            expected_peak_separation = None

    rng = np.random.default_rng(random_seed)
    centroids: list[float] = []
    reported_sigmas: list[float] = []
    line_snrs: list[float] = []
    peak_separations: list[float] = []
    for _ in range(n_trials):
        realization = realize_extracted_spectrum(result, rng)
        try:
            measurement = measure_emission_line(
                realization,
                line_center_A,
                window_kms=window_kms,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        centroids.append(measurement.centroid_A)
        reported_sigmas.append(measurement.centroid_sigma_kms)
        line_snrs.append(measurement.line_snr)
        if recover_double_peak:
            try:
                peaks = measure_double_peak(
                    realization,
                    line_center_A,
                    window_kms=window_kms,
                    minimum_prominence_snr=minimum_peak_prominence_snr,
                )
                peak_separations.append(peaks.separation_kms)
            except ValueError:
                pass

    n_recovered = len(centroids)
    if n_recovered < 2:
        raise ValueError("Fewer than two line measurements survived the realizations.")
    centroid_array = np.asarray(centroids)
    velocity_offset = _C_KMS * (
        centroid_array / expected.centroid_A - 1.0
    )
    snr_array = np.asarray(line_snrs)

    peak_bias: float | None = None
    peak_scatter: float | None = None
    if expected_peak_separation is not None and len(peak_separations) >= 2:
        peak_array = np.asarray(peak_separations)
        peak_bias = float(np.mean(peak_array - expected_peak_separation))
        peak_scatter = float(np.std(peak_array, ddof=1))

    return MonteCarloLineResult(
        n_trials=int(n_trials),
        n_recovered=n_recovered,
        recovery_fraction=n_recovered / n_trials,
        expected_centroid_A=expected.centroid_A,
        centroid_bias_kms=float(np.mean(velocity_offset)),
        centroid_scatter_kms=float(np.std(velocity_offset, ddof=1)),
        median_reported_centroid_sigma_kms=float(np.median(reported_sigmas)),
        expected_line_snr=expected.line_snr,
        line_snr_median=float(np.median(snr_array)),
        line_snr_scatter=float(np.std(snr_array, ddof=1)),
        peak_separation_expected_kms=expected_peak_separation,
        peak_separation_bias_kms=peak_bias,
        peak_separation_scatter_kms=peak_scatter,
    )
