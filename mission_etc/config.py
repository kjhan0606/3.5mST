"""Mode and channel configuration for the mission exposure-time calculator.

The configuration layer separates user intent from the numerical kernels in
``slitless_etc.py``.  A spectral channel defines the disperser and the
band-limiting filter seen by one detector exposure.  The full science
wavelength envelope is not treated as one unfiltered R=5000 trace.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping

import slitless_etc as legacy


class EtcMode(str, Enum):
    """User-facing ETC modes with distinct inputs and outputs."""

    IMAGING = "imaging"
    LINE = "line"
    TEMPLATE = "template"
    MIR = "mir"
    VALIDATE = "validate"


@dataclass(frozen=True)
class SpectralChannel:
    """One order-sorted slitless spectroscopy setting."""

    key: str
    label: str
    band_min_A: float
    band_max_A: float
    resolving_power: float
    sampling_pix: float = 2.5
    wavelength_calibration_floor_kms: float = 5.0
    relative_flux_floor: float = 0.01
    science_use: str = ""

    def __post_init__(self) -> None:
        if self.band_min_A <= 0 or self.band_max_A <= self.band_min_A:
            raise ValueError("A spectral channel requires an increasing positive band.")
        if self.resolving_power <= 0:
            raise ValueError("Resolving power must be positive.")
        if self.sampling_pix < 2:
            raise ValueError("A resolution element must be sampled by at least two pixels.")
        if self.wavelength_calibration_floor_kms < 0:
            raise ValueError("The wavelength-calibration floor cannot be negative.")
        if not 0 <= self.relative_flux_floor < 1:
            raise ValueError("The relative flux floor must lie in [0, 1).")

    @property
    def pivot_A(self) -> float:
        return (self.band_min_A * self.band_max_A) ** 0.5

    @property
    def width_A(self) -> float:
        return self.band_max_A - self.band_min_A


# R=5000 settings use order-sorting bands short enough to fit on a practical
# detector.  The two R=1000 settings represent the broad-feature alternative
# that must be compared against R=5000 for kilonova observations.
SPECTRAL_CHANNELS: Mapping[str, SpectralChannel] = {
    "compact-blue": SpectralChannel(
        key="compact-blue",
        label="Compact-object blue line setting",
        band_min_A=4300.0,
        band_max_A=5100.0,
        resolving_power=5000.0,
        science_use="Hgamma, He II, Bowen blend, Hbeta, and nearby metal lines",
    ),
    "compact-red": SpectralChannel(
        key="compact-red",
        label="Compact-object red line setting",
        band_min_A=6200.0,
        band_max_A=6900.0,
        resolving_power=5000.0,
        science_use="Halpha and He I line-profile time series",
    ),
    "compact-nir": SpectralChannel(
        key="compact-nir",
        label="Compact-object near-infrared setting",
        band_min_A=10000.0,
        band_max_A=15000.0,
        resolving_power=5000.0,
        science_use="Paschen and near-infrared continuum diagnostics",
        wavelength_calibration_floor_kms=8.0,
        relative_flux_floor=0.015,
    ),
    "transient-optical": SpectralChannel(
        key="transient-optical",
        label="Broad-transient optical setting",
        band_min_A=4000.0,
        band_max_A=10000.0,
        resolving_power=1000.0,
        science_use="Broad kilonova and accretion-transient spectral evolution",
        wavelength_calibration_floor_kms=15.0,
        relative_flux_floor=0.02,
    ),
    "transient-nir": SpectralChannel(
        key="transient-nir",
        label="Broad-transient near-infrared setting",
        band_min_A=10000.0,
        band_max_A=15000.0,
        resolving_power=1000.0,
        science_use="Red kilonova continuum and broad absorption complexes",
        wavelength_calibration_floor_kms=20.0,
        relative_flux_floor=0.02,
    ),
}


def get_channel(name: str) -> SpectralChannel:
    """Return a named spectral channel with a useful error for invalid names."""

    try:
        return SPECTRAL_CHANNELS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(SPECTRAL_CHANNELS))
        raise KeyError(f"Unknown spectral channel {name!r}. Choose one of {choices}.") from exc


def build_instrument(
    channel: SpectralChannel | str,
    **overrides: float | int | bool | str | None,
) -> legacy.InstrumentConfig:
    """Build the low-level instrument configuration for one spectral channel.

    The current component throughput and zodiacal-light tables are inherited
    from ``slitless_etc.realistic_cfg``.  Channel edges replace the broad
    survey band and therefore set the slitless background collected by each
    detector pixel.
    """

    selected = get_channel(channel) if isinstance(channel, str) else channel
    base = legacy.realistic_cfg(
        R=selected.resolving_power,
        res_element_pix=selected.sampling_pix,
        band_min_A=selected.band_min_A,
        band_max_A=selected.band_max_A,
        edge_roll_A=min(150.0, 0.1 * selected.width_A),
    )
    return replace(base, **overrides)


def describe_modes() -> tuple[dict[str, str], ...]:
    """Machine-readable descriptions used by the CLI and GUI."""

    return (
        {
            "mode": EtcMode.IMAGING.value,
            "purpose": "Broadband point-source depth and saturation",
            "primary_input": "AB magnitude or requested S/N",
        },
        {
            "mode": EtcMode.LINE.value,
            "purpose": "One isolated emission-line depth or S/N",
            "primary_input": "Integrated line flux and line wavelength",
        },
        {
            "mode": EtcMode.TEMPLATE.value,
            "purpose": "Full observed compact-object or transient spectrum",
            "primary_input": "Wavelength and flux-density template",
        },
        {
            "mode": EtcMode.MIR.value,
            "purpose": "Cooled mid-infrared imaging depth",
            "primary_input": "Flux density and MIR imaging band",
        },
        {
            "mode": EtcMode.VALIDATE.value,
            "purpose": "Published-mission and analytic regression checks",
            "primary_input": "Named validation case",
        },
    )
