"""Cooled mid-infrared imaging mode."""

from __future__ import annotations

import slitless_etc as legacy

from .results import MirImagingEtcResult


def calculate_mir_imaging(
    channel: str,
    exposure_s: float,
    flux_uJy: float | None = None,
    target_snr: float = 5.0,
    background: str = "nominal",
    observatory: str = "3.5mST",
    **instrument_overrides: float | int | bool | str | None,
) -> MirImagingEtcResult:
    """Calculate a cooled-MIR point-source limit for NC1 or NC2."""

    channel = channel.upper()
    if channel not in legacy.MIR_CHANNELS:
        choices = ", ".join(sorted(legacy.MIR_CHANNELS))
        raise KeyError(f"Unknown MIR channel {channel!r}. Choose one of {choices}.")
    if exposure_s <= 0 or target_snr <= 0:
        raise ValueError("Exposure time and target S/N must be positive.")
    if flux_uJy is not None and flux_uJy < 0:
        raise ValueError("Flux density cannot be negative.")

    cfg = legacy.mir_imaging_cfg(
        channel=channel,
        background=background,
        observatory=observatory,
        **instrument_overrides,
    )
    band_um = legacy.MIR_CHANNELS[channel]["band_um"]
    band_A = tuple(item * 1e4 for item in band_um)
    limit_jy = legacy.imaging_flux_limit_jy(
        cfg,
        band_A,
        exposure_s,
        snr=target_snr,
    )
    input_snr = (
        legacy.imaging_snr_jy(cfg, flux_uJy * 1e-6, band_A, exposure_s)
        if flux_uJy is not None
        else None
    )
    noise = legacy.imaging_noise_budget(cfg, band_A, exposure_s)
    return MirImagingEtcResult(
        channel=channel,
        background=background,
        observatory=observatory,
        band_min_um=float(band_um[0]),
        band_max_um=float(band_um[1]),
        exposure_s=float(exposure_s),
        target_snr=float(target_snr),
        limiting_flux_uJy=float(limit_jy * 1e6),
        input_flux_uJy=None if flux_uJy is None else float(flux_uJy),
        input_snr=None if input_snr is None else float(input_snr),
        aperture_pixels=float(noise["n_pix"]),
        sky_e=float(noise["sky_e"]),
        dark_e=float(noise["dark_e"]),
        read_variance_e2=float(noise["read_e2"]),
    )
