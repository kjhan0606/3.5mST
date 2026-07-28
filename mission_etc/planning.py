"""Inverse exposure-time calculations for template spectroscopy."""

from __future__ import annotations

from collections.abc import Mapping

from scipy.optimize import brentq

from .config import SpectralChannel
from .results import ExposureTimeSolution, SpectralEtcResult
from .spectroscopy import ObservedSpectrum, measure_emission_line, simulate_spectrum


def solve_exposure_time(
    spectrum: ObservedSpectrum,
    channel: SpectralChannel | str,
    *,
    metric: str,
    target_value: float,
    wavelength_min_A: float | None = None,
    wavelength_max_A: float | None = None,
    line_center_A: float | None = None,
    line_window_kms: float = 1500.0,
    minimum_exposure_s: float = 1.0,
    maximum_exposure_s: float = 100000.0,
    phase_ceiling_s: float | None = None,
    simulation_options: Mapping[str, object] | None = None,
) -> ExposureTimeSolution:
    """Solve for the shortest exposure that meets one science requirement.

    Supported metrics are ``integrated_snr``, ``line_snr``, and
    ``centroid_sigma_kms``.  The first two increase with exposure.  The
    centroid uncertainty decreases and includes the configured wavelength
    calibration floor.
    """

    supported = {"integrated_snr", "line_snr", "centroid_sigma_kms"}
    if metric not in supported:
        raise ValueError(f"Metric must be one of {', '.join(sorted(supported))}.")
    if target_value <= 0:
        raise ValueError("Target value must be positive.")
    if minimum_exposure_s <= 0 or maximum_exposure_s <= minimum_exposure_s:
        raise ValueError("Exposure bounds must be positive and increasing.")
    if phase_ceiling_s is not None and phase_ceiling_s <= 0:
        raise ValueError("Phase ceiling must be positive.")
    if metric == "integrated_snr":
        if wavelength_min_A is None or wavelength_max_A is None:
            raise ValueError("Integrated S/N requires wavelength bounds.")
    elif line_center_A is None:
        raise ValueError("Line metrics require line_center_A.")

    upper = min(
        maximum_exposure_s,
        phase_ceiling_s if phase_ceiling_s is not None else maximum_exposure_s,
    )
    if upper < minimum_exposure_s:
        raise ValueError("Phase ceiling lies below the minimum exposure.")
    options = dict(simulation_options or {})
    evaluations = 0
    last_result: SpectralEtcResult | None = None

    def evaluate(exposure_s: float) -> tuple[float, SpectralEtcResult]:
        nonlocal evaluations, last_result
        evaluations += 1
        result = simulate_spectrum(spectrum, channel, exposure_s, **options)
        last_result = result
        if metric == "integrated_snr":
            value = result.integrated_snr(
                float(wavelength_min_A),
                float(wavelength_max_A),
            )
        else:
            measurement = measure_emission_line(
                result,
                float(line_center_A),
                window_kms=line_window_kms,
            )
            value = (
                measurement.line_snr
                if metric == "line_snr"
                else measurement.centroid_sigma_kms
            )
        return float(value), result

    def objective(exposure_s: float) -> float:
        value, _ = evaluate(exposure_s)
        if metric == "centroid_sigma_kms":
            return target_value - value
        return value - target_value

    lower_value, lower_result = evaluate(minimum_exposure_s)
    lower_objective = (
        target_value - lower_value
        if metric == "centroid_sigma_kms"
        else lower_value - target_value
    )
    if lower_objective >= 0:
        saturated = bool(lower_result.metadata["saturated_upper_bound"])
        return ExposureTimeSolution(
            metric=metric,
            target_value=float(target_value),
            exposure_s=float(minimum_exposure_s),
            achieved_value=lower_value,
            feasible=not saturated,
            phase_ceiling_s=phase_ceiling_s,
            saturated_upper_bound=saturated,
            iterations=evaluations,
            message=(
                "Requirement met at the lower exposure bound."
                if not saturated
                else "Requirement met, but the conservative full-well bound is exceeded."
            ),
        )

    upper_value, upper_result = evaluate(upper)
    upper_objective = (
        target_value - upper_value
        if metric == "centroid_sigma_kms"
        else upper_value - target_value
    )
    if upper_objective < 0:
        ceiling_text = (
            "phase-smearing ceiling"
            if phase_ceiling_s is not None and upper == phase_ceiling_s
            else "maximum exposure"
        )
        return ExposureTimeSolution(
            metric=metric,
            target_value=float(target_value),
            exposure_s=float(upper),
            achieved_value=upper_value,
            feasible=False,
            phase_ceiling_s=phase_ceiling_s,
            saturated_upper_bound=bool(
                upper_result.metadata["saturated_upper_bound"]
            ),
            iterations=evaluations,
            message=f"Requirement is not reached before the {ceiling_text}.",
        )

    root = brentq(
        objective,
        minimum_exposure_s,
        upper,
        xtol=max(1.0e-3, 1.0e-6 * upper),
        rtol=1.0e-8,
    )
    achieved, result = evaluate(root)
    saturated = bool(result.metadata["saturated_upper_bound"])
    message = "Requirement reached by inverse ETC solution."
    if saturated:
        message += " The conservative full-well bound is exceeded."
    return ExposureTimeSolution(
        metric=metric,
        target_value=float(target_value),
        exposure_s=float(root),
        achieved_value=achieved,
        feasible=not saturated,
        phase_ceiling_s=phase_ceiling_s,
        saturated_upper_bound=saturated,
        iterations=evaluations,
        message=message,
    )
