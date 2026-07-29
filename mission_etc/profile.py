"""Full-profile velocity fitting for extracted compact-object spectra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


_C_KMS = 299792.458


@dataclass(frozen=True)
class TemplateVelocityFit:
    """Velocity shift and covariance from a complete template-profile fit."""

    velocity_kms: float
    velocity_sigma_kms: float
    amplitude: float
    continuum_offset_e: float
    continuum_slope_e_per_A: float
    chi2: float
    degrees_of_freedom: int
    success: bool
    message: str
    wavelength_A: np.ndarray
    model_e: np.ndarray


def fit_template_velocity(
    wavelength_A: np.ndarray,
    observed_e: np.ndarray,
    uncertainty_e: np.ndarray,
    template_e: np.ndarray,
    *,
    line_center_A: float,
    window_kms: float = 1500.0,
    velocity_bound_kms: float = 1000.0,
    initial_velocity_kms: float = 0.0,
) -> TemplateVelocityFit:
    """Fit velocity, amplitude, and a linear continuum to a full line profile."""

    wavelength = np.asarray(wavelength_A, dtype=float)
    observed = np.asarray(observed_e, dtype=float)
    uncertainty = np.asarray(uncertainty_e, dtype=float)
    template = np.asarray(template_e, dtype=float)
    if any(item.ndim != 1 for item in (wavelength, observed, uncertainty, template)):
        raise ValueError("Profile-fit inputs must be one-dimensional.")
    if len({item.size for item in (wavelength, observed, uncertainty, template)}) != 1:
        raise ValueError("Profile-fit arrays must have equal lengths.")
    if line_center_A <= 0 or window_kms <= 0 or velocity_bound_kms <= 0:
        raise ValueError("Line center, window, and velocity bound must be positive.")
    velocity = _C_KMS * (wavelength / line_center_A - 1.0)
    mask = (
        (np.abs(velocity) <= window_kms)
        & np.isfinite(observed)
        & np.isfinite(template)
        & np.isfinite(uncertainty)
        & (uncertainty > 0)
    )
    if np.count_nonzero(mask) < 12:
        raise ValueError("The selected profile contains fewer than 12 valid bins.")
    wave_fit = wavelength[mask]
    observed_fit = observed[mask]
    sigma_fit = uncertainty[mask]
    template_fit = template[mask]
    pivot = float(np.mean(wave_fit))
    wavelength_scale = float(np.ptp(wave_fit))
    x_coordinate = (wave_fit - pivot) / wavelength_scale

    template_variance = float(np.dot(template_fit, template_fit))
    amplitude_initial = (
        max(0.0, float(np.dot(observed_fit, template_fit) / template_variance))
        if template_variance > 0
        else 1.0
    )
    continuum_initial = float(
        np.median(observed_fit - amplitude_initial * template_fit)
    )

    def model(parameters: np.ndarray) -> np.ndarray:
        velocity_shift, amplitude, continuum, slope_scaled = parameters
        doppler_factor = 1.0 + velocity_shift / _C_KMS
        shifted_template = np.interp(
            wave_fit / doppler_factor,
            wavelength,
            template,
        )
        return (
            amplitude * shifted_template
            + continuum
            + slope_scaled * x_coordinate
        )

    def residual(parameters: np.ndarray) -> np.ndarray:
        return (observed_fit - model(parameters)) / sigma_fit

    result = least_squares(
        residual,
        x0=np.asarray(
            (
                initial_velocity_kms,
                amplitude_initial,
                continuum_initial,
                0.0,
            )
        ),
        bounds=(
            (-velocity_bound_kms, 0.0, -np.inf, -np.inf),
            (velocity_bound_kms, np.inf, np.inf, np.inf),
        ),
        x_scale="jac",
        max_nfev=500,
    )
    chi2 = float(np.dot(result.fun, result.fun))
    degrees_of_freedom = max(1, wave_fit.size - result.x.size)
    fisher = result.jac.T @ result.jac
    try:
        covariance = np.linalg.inv(fisher)
        covariance *= max(1.0, chi2 / degrees_of_freedom)
        velocity_sigma = float(np.sqrt(max(0.0, covariance[0, 0])))
    except np.linalg.LinAlgError:
        velocity_sigma = float("inf")
    slope_e_per_A = float(result.x[3] / wavelength_scale)
    return TemplateVelocityFit(
        velocity_kms=float(result.x[0]),
        velocity_sigma_kms=velocity_sigma,
        amplitude=float(result.x[1]),
        continuum_offset_e=float(result.x[2]),
        continuum_slope_e_per_A=slope_e_per_A,
        chi2=chi2,
        degrees_of_freedom=degrees_of_freedom,
        success=bool(result.success),
        message=str(result.message),
        wavelength_A=wave_fit,
        model_e=model(result.x),
    )
