"""Isolated emission-line mode."""

from __future__ import annotations

import slitless_etc as legacy

from .config import SpectralChannel, build_instrument, get_channel
from .results import IsolatedLineEtcResult


def calculate_line(
    channel: SpectralChannel | str,
    wavelength_A: float,
    exposure_s: float,
    line_flux_erg_s_cm2: float | None = None,
    target_snr: float = 5.0,
    continuum_magnitude_ab: float | None = None,
    source_fwhm_arcsec: float = 0.0,
    **instrument_overrides: float | int | bool | str | None,
) -> IsolatedLineEtcResult:
    """Calculate an unresolved-line limit or the S/N of a supplied line."""

    selected = get_channel(channel) if isinstance(channel, str) else channel
    if not selected.band_min_A <= wavelength_A <= selected.band_max_A:
        raise ValueError("The line wavelength lies outside the selected channel.")
    if exposure_s <= 0 or target_snr <= 0:
        raise ValueError("Exposure time and target S/N must be positive.")
    if line_flux_erg_s_cm2 is not None and line_flux_erg_s_cm2 < 0:
        raise ValueError("Line flux cannot be negative.")

    cfg = build_instrument(selected, **instrument_overrides)
    limit = legacy.f_limit(
        cfg,
        wavelength_A,
        exposure_s,
        snr=target_snr,
        source_fwhm=source_fwhm_arcsec,
        filter_width_A=selected.width_A,
        cont_mag_AB=continuum_magnitude_ab,
    )
    input_snr = (
        legacy.line_sn(
            cfg,
            line_flux_erg_s_cm2,
            wavelength_A,
            exposure_s,
            source_fwhm=source_fwhm_arcsec,
            filter_width_A=selected.width_A,
            cont_mag_AB=continuum_magnitude_ab,
        )
        if line_flux_erg_s_cm2 is not None
        else None
    )
    noise = legacy.noise_breakdown(
        cfg,
        wavelength_A,
        exposure_s,
        source_fwhm=source_fwhm_arcsec,
        filter_width_A=selected.width_A,
    )
    return IsolatedLineEtcResult(
        channel=selected.key,
        wavelength_A=float(wavelength_A),
        exposure_s=float(exposure_s),
        target_snr=float(target_snr),
        limiting_flux_erg_s_cm2=float(limit),
        input_flux_erg_s_cm2=(
            None if line_flux_erg_s_cm2 is None else float(line_flux_erg_s_cm2)
        ),
        input_snr=None if input_snr is None else float(input_snr),
        footprint_pixels=float(noise["n_pix"]),
        sky_e=float(noise["sky_e"]),
        dark_e=float(noise["dark_e"]),
        read_variance_e2=float(noise["read_e2"]),
    )
