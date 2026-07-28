"""Regression tests for the observing-mode-specific mission ETC."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mission_etc import (
    ObservedSpectrum,
    SceneSource,
    SpectralChannel,
    bin_spectrum,
    calculate_imaging,
    calculate_mir_imaging,
    extract_multi_roll_scene,
    fit_template_velocity,
    load_engrave_xshooter_arm,
    measure_double_peak,
    measure_emission_line,
    monte_carlo_line_recovery,
    run_validation,
    scale_spectrum,
    shift_spectrum_velocity,
    simulate_multi_roll_scene,
    simulate_spectrum,
    solve_exposure_time,
)
from mission_etc.cli import main


@pytest.fixture(scope="module")
def double_peaked_spectrum() -> ObservedSpectrum:
    wavelength = np.arange(6150.0, 6950.0, 0.05)
    continuum = np.full_like(wavelength, 2.0e-17)
    blue_peak = 1.2e-16 * np.exp(-0.5 * ((wavelength - 6555.0) / 1.8) ** 2)
    red_peak = 1.0e-16 * np.exp(-0.5 * ((wavelength - 6570.0) / 2.1) ** 2)
    return ObservedSpectrum(
        wavelength,
        continuum + blue_peak + red_peak,
        name="synthetic double-peaked Halpha",
    )


@pytest.fixture(scope="module")
def simulated(double_peaked_spectrum: ObservedSpectrum):
    return simulate_spectrum(double_peaked_spectrum, "compact-red", 1800.0)


def test_ab_normalization(double_peaked_spectrum: ObservedSpectrum) -> None:
    scaled = double_peaked_spectrum.scaled_to_ab_magnitude(22.0, 6400.0, 6800.0)
    mask = (scaled.wavelength_A >= 6400.0) & (scaled.wavelength_A <= 6800.0)
    wavelength = scaled.wavelength_A[mask]
    fnu = scaled.flux_lambda[mask] * wavelength**2 / 2.99792458e18
    mean_fnu = np.trapezoid(fnu, wavelength) / (wavelength[-1] - wavelength[0])
    recovered_ab = -2.5 * np.log10(mean_fnu) - 48.60
    assert recovered_ab == pytest.approx(22.0, abs=1e-10)


def test_detector_sampling_and_finite_noise(simulated) -> None:
    expected_pixels = 5000.0 * 2.5 * np.log(6900.0 / 6200.0)
    assert simulated.wavelength_A.size == pytest.approx(expected_pixels, rel=0.01)
    assert np.all(np.diff(simulated.wavelength_A) > 0)
    assert np.all(simulated.total_variance_e2 > 0)
    assert np.all(np.isfinite(simulated.snr))


def test_flat_continuum_survives_lsf() -> None:
    wavelength = np.arange(6150.0, 6950.0, 0.1)
    flat = ObservedSpectrum(wavelength, np.full_like(wavelength, 3.0e-17))
    result = simulate_spectrum(flat, "compact-red", 100.0)
    interior = (result.wavelength_A > 6220.0) & (result.wavelength_A < 6880.0)
    assert np.median(result.convolved_flux_lambda[interior]) == pytest.approx(
        3.0e-17,
        rel=1e-10,
    )


def test_longer_exposure_improves_snr(double_peaked_spectrum: ObservedSpectrum) -> None:
    short = simulate_spectrum(double_peaked_spectrum, "compact-red", 300.0)
    long = simulate_spectrum(double_peaked_spectrum, "compact-red", 1800.0)
    assert long.integrated_snr(6540.0, 6585.0) > short.integrated_snr(6540.0, 6585.0)


def test_binning_increases_per_bin_snr(simulated) -> None:
    binned = bin_spectrum(simulated, 1000.0)
    assert binned.wavelength_A.size < simulated.wavelength_A.size
    assert np.median(binned.snr) > np.median(simulated.snr)


def test_double_peaked_line_measurement(simulated) -> None:
    measurement = measure_emission_line(simulated, 6562.8, window_kms=1200.0)
    assert measurement.line_snr > 0
    assert 6555.0 < measurement.centroid_A < 6570.0
    assert measurement.second_moment_kms > 100.0
    assert measurement.centroid_sigma_kms >= 5.0


def test_double_peak_recovery(simulated) -> None:
    peaks = measure_double_peak(
        simulated,
        6562.8,
        window_kms=1200.0,
        minimum_prominence_snr=2.0,
    )
    assert peaks.blue_peak_A < 6562.8 < peaks.red_peak_A
    assert peaks.separation_kms > 400.0


def test_native_resolution_is_not_applied_twice() -> None:
    wavelength = np.arange(6150.0, 6950.0, 0.02)
    sigma_A = 6562.8 / 5000.0 / 2.354820045
    flux = 2.0e-17 + 2.0e-16 * np.exp(
        -0.5 * ((wavelength - 6562.8) / sigma_A) ** 2
    )
    already_resolved = ObservedSpectrum(
        wavelength,
        flux,
        native_resolving_power=5000.0,
    )
    resolution_unknown = ObservedSpectrum(wavelength, flux)
    preserved = simulate_spectrum(already_resolved, "compact-red", 300.0)
    broadened = simulate_spectrum(resolution_unknown, "compact-red", 300.0)

    def width(result) -> float:
        residual = np.clip(result.convolved_flux_lambda - 2.0e-17, 0.0, None)
        return float(
            np.sqrt(
                np.average(
                    (result.wavelength_A - 6562.8) ** 2,
                    weights=residual,
                )
            )
        )

    assert width(preserved) < 0.85 * width(broadened)
    assert preserved.metadata["effective_resolving_power"] == pytest.approx(5000.0)
    assert preserved.metadata["spectral_kernel_fwhm_detector_pix"] == 0.0


def test_lower_resolution_template_is_not_sharpened(
    double_peaked_spectrum: ObservedSpectrum,
) -> None:
    low_resolution = ObservedSpectrum(
        double_peaked_spectrum.wavelength_A,
        double_peaked_spectrum.flux_lambda,
        native_resolving_power=1800.0,
    )
    result = simulate_spectrum(low_resolution, "compact-red", 300.0)
    assert result.metadata["effective_resolving_power"] == pytest.approx(1800.0)
    assert "no sharpening" in result.metadata["resolution_status"]


def test_inverse_exposure_recovers_line_snr(
    double_peaked_spectrum: ObservedSpectrum,
) -> None:
    solution = solve_exposure_time(
        double_peaked_spectrum,
        "compact-red",
        metric="line_snr",
        target_value=20.0,
        line_center_A=6562.8,
        line_window_kms=1200.0,
        minimum_exposure_s=10.0,
        maximum_exposure_s=5000.0,
    )
    assert solution.feasible
    assert solution.achieved_value == pytest.approx(20.0, rel=2.0e-4)
    assert 10.0 < solution.exposure_s < 5000.0


def test_phase_ceiling_can_make_requirement_infeasible(
    double_peaked_spectrum: ObservedSpectrum,
) -> None:
    solution = solve_exposure_time(
        double_peaked_spectrum,
        "compact-red",
        metric="line_snr",
        target_value=1000.0,
        line_center_A=6562.8,
        line_window_kms=1200.0,
        maximum_exposure_s=5000.0,
        phase_ceiling_s=30.0,
    )
    assert not solution.feasible
    assert solution.exposure_s == 30.0
    assert "phase-smearing ceiling" in solution.message


def test_monte_carlo_reports_bias_and_scatter(simulated) -> None:
    summary = monte_carlo_line_recovery(
        simulated,
        6562.8,
        window_kms=1200.0,
        n_trials=40,
        random_seed=1234,
        recover_double_peak=True,
        minimum_peak_prominence_snr=2.0,
    )
    assert summary.n_recovered >= 35
    assert abs(summary.centroid_bias_kms) < 30.0
    assert summary.centroid_scatter_kms > 0
    assert summary.peak_separation_expected_kms is not None


def test_engrave_loader_preserves_provenance(tmp_path) -> None:
    wavelength = np.arange(5300.0, 7000.0, 0.2)
    flux = np.full_like(wavelength, 2.0e-17)
    error = np.full_like(wavelength, 1.0e-18)
    path = tmp_path / "engrave_vis.dat"
    np.savetxt(path, np.column_stack((wavelength, flux, flux, error)))
    calibrated = load_engrave_xshooter_arm(
        path,
        arm="VIS",
        epoch_mjd=57983.969,
        phase_days=1.43,
    )
    observed = calibrated.to_observed_spectrum()
    assert observed.native_resolving_power == 8800.0
    assert observed.metadata["object_name"] == "AT2017gfo"
    assert observed.uncertainty_flux_lambda is not None


def test_three_roll_scene_separates_an_overlapping_trace() -> None:
    wavelength = np.arange(6400.0, 6700.0, 0.05)
    flux = 2.0e-16 + 8.0e-16 * np.exp(
        -0.5 * ((wavelength - 6562.8) / 1.2) ** 2
    )
    target_spectrum = ObservedSpectrum(
        wavelength,
        flux,
        native_resolving_power=5000.0,
    )
    contaminant = scale_spectrum(
        shift_spectrum_velocity(target_spectrum, 250.0),
        0.7,
    )
    channel = SpectralChannel(
        "scene-test",
        "scene test",
        6500.0,
        6620.0,
        1000.0,
        sampling_pix=2.5,
    )
    scene = simulate_multi_roll_scene(
        (
            SceneSource("target", target_spectrum, 0.0, 0.0),
            SceneSource("contaminant", contaminant, 0.7, 0.0),
        ),
        channel,
        20.0,
        target_id="target",
        roll_angles_deg=(0.0, 60.0, 120.0),
        random_seed=5,
        include_cirrus=False,
        include_thermal=False,
    )
    assert scene.metadata["exposure_s_per_roll"] == 20.0
    assert scene.metadata["total_exposure_s"] == 60.0
    assert scene.metadata["pixel_scale_arcsec"] > 0
    noiseless = replace(
        scene,
        exposures=tuple(
            replace(item, image_e=item.expectation_e)
            for item in scene.exposures
        ),
    )
    one_roll = extract_multi_roll_scene(noiseless, roll_indices=(0,))
    three_roll = extract_multi_roll_scene(noiseless)
    truth = scene.truth_spectra_e["target"]
    one_error = np.linalg.norm(one_roll.spectra_e["target"] - truth)
    three_error = np.linalg.norm(three_roll.spectra_e["target"] - truth)
    assert three_error < 1.0e-3 * np.linalg.norm(truth)
    assert three_error < 0.01 * one_error


def test_complete_profile_fit_recovers_velocity() -> None:
    wavelength = np.arange(6500.0, 6625.0, 0.1)
    template = (
        100.0
        + 35.0 * np.exp(-0.5 * ((wavelength - 6557.0) / 1.8) ** 2)
        + 30.0 * np.exp(-0.5 * ((wavelength - 6569.0) / 2.0) ** 2)
    )
    injected_velocity = 85.0
    shifted = np.interp(
        wavelength / (1.0 + injected_velocity / 299792.458),
        wavelength,
        template,
    )
    uncertainty = np.full_like(wavelength, 0.5)
    fit = fit_template_velocity(
        wavelength,
        shifted,
        uncertainty,
        template,
        line_center_A=6562.8,
        window_kms=1500.0,
        velocity_bound_kms=300.0,
    )
    assert fit.success
    assert fit.velocity_kms == pytest.approx(injected_velocity, abs=0.2)
    assert fit.velocity_sigma_kms < 2.0


def test_incomplete_channel_is_rejected() -> None:
    wavelength = np.arange(6400.0, 6800.0, 0.2)
    spectrum = ObservedSpectrum(wavelength, np.full_like(wavelength, 1.0e-17))
    with pytest.raises(ValueError, match="does not cover the full selected channel"):
        simulate_spectrum(spectrum, "compact-red", 100.0)


def test_line_limit_inverse_regression() -> None:
    validation = run_validation()
    assert validation["passed"] is True


def test_imaging_and_mir_modes_keep_distinct_units() -> None:
    optical = calculate_imaging("SDSS r", 300.0, magnitude_ab=23.0)
    mir = calculate_mir_imaging("NC1", 180.0, flux_uJy=100.0)
    assert optical.input_magnitude_ab == 23.0
    assert optical.input_snr is not None
    assert mir.input_flux_uJy == 100.0
    assert mir.input_snr is not None
    assert mir.limiting_flux_uJy > 0


def test_template_cli_writes_detector_csv(
    tmp_path,
    double_peaked_spectrum: ObservedSpectrum,
    capsys,
) -> None:
    input_path = tmp_path / "input_spectrum.txt"
    output_path = tmp_path / "detector_spectrum.csv"
    np.savetxt(
        input_path,
        np.column_stack(
            (double_peaked_spectrum.wavelength_A, double_peaked_spectrum.flux_lambda)
        ),
    )
    status = main(
        [
            "template",
            "compact-red",
            str(input_path),
            "300",
            "--bin-R",
            "1000",
            "--line-center-A",
            "6562.8",
            "--line-window-kms",
            "1200",
            "--output-csv",
            str(output_path),
        ]
    )
    captured = capsys.readouterr()
    assert status == 0
    assert '"n_detector_pixels"' in captured.out
    assert output_path.exists()
    header = output_path.read_text().splitlines()[0]
    assert "source_e" in header
    assert "total_variance_e2" in header
