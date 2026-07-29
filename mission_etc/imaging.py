"""Broadband optical and near-infrared imaging mode."""

from __future__ import annotations

import slitless_etc as legacy

from .results import ImagingEtcResult


def calculate_imaging(
    filter_name: str,
    exposure_s: float,
    magnitude_ab: float | None = None,
    target_snr: float = 5.0,
    aperture_fwhm: float = 1.0,
    **instrument_overrides: float | int | bool | str | None,
) -> ImagingEtcResult:
    """Calculate point-source depth for one named standard filter."""

    if filter_name not in legacy.STANDARD_FILTERS:
        choices = ", ".join(legacy.STANDARD_FILTERS)
        raise KeyError(f"Unknown filter {filter_name!r}. Choose one of {choices}.")
    if exposure_s <= 0 or target_snr <= 0:
        raise ValueError("Exposure time and target S/N must be positive.")
    pivot_um, width_um = legacy.STANDARD_FILTERS[filter_name]
    pivot_A = pivot_um * 1e4
    width_A = width_um * 1e4
    cfg = legacy.realistic_cfg(**instrument_overrides)
    limiting_magnitude = legacy.imaging_maglimit(
        cfg,
        pivot_A,
        width_A,
        exposure_s,
        snr=target_snr,
        aper_fwhm_mult=aperture_fwhm,
    )
    saturation_magnitude = legacy.saturation_maglimit(
        cfg,
        pivot_A,
        width_A,
        min(exposure_s, cfg.t_single),
    )
    rates = legacy.count_rates(
        cfg,
        magnitude_ab if magnitude_ab is not None else limiting_magnitude,
        pivot_A,
        width_A,
        aper_fwhm_mult=aperture_fwhm,
    )
    input_snr = (
        legacy.imaging_snr(
            cfg,
            magnitude_ab,
            pivot_A,
            width_A,
            exposure_s,
            aper_fwhm_mult=aperture_fwhm,
        )
        if magnitude_ab is not None
        else None
    )
    return ImagingEtcResult(
        filter_name=filter_name,
        pivot_A=pivot_A,
        width_A=width_A,
        exposure_s=float(exposure_s),
        limiting_magnitude_ab=float(limiting_magnitude),
        input_magnitude_ab=None if magnitude_ab is None else float(magnitude_ab),
        input_snr=None if input_snr is None else float(input_snr),
        saturation_magnitude_ab=float(saturation_magnitude),
        source_e_s=(
            float(rates["source_e_s"]) if magnitude_ab is not None else None
        ),
        sky_e_s_pixel=float(rates["sky_e_s_pix"]),
        aperture_pixels=float(rates["n_pix"]),
    )
