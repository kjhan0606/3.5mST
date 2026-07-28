"""Internal consistency checks for the reorganized ETC interfaces."""

from __future__ import annotations

from dataclasses import asdict

from .line import calculate_line


def run_validation() -> dict[str, object]:
    """Run deterministic checks that do not depend on external mission files."""

    target_snr = 5.0
    limit = calculate_line(
        "compact-red",
        wavelength_A=6562.8,
        exposure_s=1800.0,
        target_snr=target_snr,
    )
    recovered = calculate_line(
        "compact-red",
        wavelength_A=6562.8,
        exposure_s=1800.0,
        line_flux_erg_s_cm2=limit.limiting_flux_erg_s_cm2,
        target_snr=target_snr,
    )
    relative_error = abs(float(recovered.input_snr) / target_snr - 1.0)
    return {
        "passed": relative_error < 1e-10,
        "checks": {
            "line_limit_inverse": {
                "target_snr": target_snr,
                "recovered_snr": recovered.input_snr,
                "relative_error": relative_error,
            }
        },
        "reference_result": asdict(limit),
    }
