"""User-facing exposure-time calculator organized by observing mode.

The package keeps imaging, isolated-line, template-spectrum, MIR, and
validation interfaces separate.  ``slitless_etc.py`` remains the numerical
kernel and backwards-compatible API used by existing scripts.
"""

from .config import (
    EtcMode,
    SPECTRAL_CHANNELS,
    SpectralChannel,
    build_instrument,
    describe_modes,
    get_channel,
)
from .imaging import calculate_imaging
from .line import calculate_line
from .mir import calculate_mir_imaging
from .results import (
    BinnedSpectrum,
    DoublePeakMeasurement,
    ExposureTimeSolution,
    ImagingEtcResult,
    IsolatedLineEtcResult,
    LineMeasurement,
    MirImagingEtcResult,
    MonteCarloLineResult,
    SpectralEtcResult,
)
from .monte_carlo import monte_carlo_line_recovery, realize_extracted_spectrum
from .planning import solve_exposure_time
from .profile import TemplateVelocityFit, fit_template_velocity
from .scene import (
    ExtractedScene,
    MultiRollScene,
    SceneSource,
    SlitlessExposure,
    extract_multi_roll_scene,
    realize_scene_noise,
    scale_spectrum,
    shift_spectrum_velocity,
    simulate_multi_roll_scene,
)
from .spectroscopy import (
    ObservedSpectrum,
    bin_spectrum,
    maximum_phase_resolved_exposure,
    measure_double_peak,
    measure_emission_line,
    phase_smearing,
    simulate_spectrum,
)
from .templates import (
    CalibratedSpectrum,
    SpectrumProvenance,
    combine_calibrated_spectra,
    fetch_engrave_xshooter_arm,
    fetch_lamost_ugem_outburst,
    load_cds_ugem_phase_spectrum,
    load_engrave_xshooter_arm,
    load_lamost_ugem_outburst,
    mask_wavelength_ranges,
)
from .validation import run_validation

__all__ = [
    "BinnedSpectrum",
    "CalibratedSpectrum",
    "DoublePeakMeasurement",
    "EtcMode",
    "ExtractedScene",
    "ExposureTimeSolution",
    "ImagingEtcResult",
    "IsolatedLineEtcResult",
    "LineMeasurement",
    "MirImagingEtcResult",
    "MonteCarloLineResult",
    "MultiRollScene",
    "ObservedSpectrum",
    "SPECTRAL_CHANNELS",
    "SceneSource",
    "SlitlessExposure",
    "SpectralChannel",
    "SpectralEtcResult",
    "SpectrumProvenance",
    "TemplateVelocityFit",
    "bin_spectrum",
    "build_instrument",
    "calculate_imaging",
    "calculate_line",
    "calculate_mir_imaging",
    "combine_calibrated_spectra",
    "describe_modes",
    "extract_multi_roll_scene",
    "fetch_engrave_xshooter_arm",
    "fetch_lamost_ugem_outburst",
    "fit_template_velocity",
    "get_channel",
    "load_cds_ugem_phase_spectrum",
    "load_engrave_xshooter_arm",
    "load_lamost_ugem_outburst",
    "mask_wavelength_ranges",
    "maximum_phase_resolved_exposure",
    "measure_double_peak",
    "measure_emission_line",
    "monte_carlo_line_recovery",
    "phase_smearing",
    "realize_extracted_spectrum",
    "realize_scene_noise",
    "run_validation",
    "scale_spectrum",
    "shift_spectrum_velocity",
    "simulate_spectrum",
    "simulate_multi_roll_scene",
    "solve_exposure_time",
]
