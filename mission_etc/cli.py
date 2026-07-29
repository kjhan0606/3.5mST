"""Command-line interface for observing-mode-specific ETC calculations."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

import slitless_etc as legacy

from .config import SPECTRAL_CHANNELS, describe_modes
from .imaging import calculate_imaging
from .line import calculate_line
from .mir import calculate_mir_imaging
from .spectroscopy import (
    ObservedSpectrum,
    bin_spectrum,
    measure_emission_line,
    simulate_spectrum,
)
from .validation import run_validation


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, default=_json_default))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mission_etc",
        description="3.5 m space-telescope ETC organized by observing mode.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("modes", help="List user-facing modes and their inputs.")
    subparsers.add_parser("channels", help="List available spectroscopy channels.")

    imaging = subparsers.add_parser("imaging", help="Broadband AB point-source depth.")
    imaging.add_argument("filter", choices=tuple(legacy.STANDARD_FILTERS))
    imaging.add_argument("exposure_s", type=float)
    imaging.add_argument("--magnitude-ab", type=float)
    imaging.add_argument("--target-snr", type=float, default=5.0)

    line = subparsers.add_parser("line", help="Isolated integrated-line calculation.")
    line.add_argument("channel", choices=tuple(SPECTRAL_CHANNELS))
    line.add_argument("wavelength_A", type=float)
    line.add_argument("exposure_s", type=float)
    line.add_argument("--flux-erg-s-cm2", type=float)
    line.add_argument("--continuum-magnitude-ab", type=float)
    line.add_argument("--source-fwhm-arcsec", type=float, default=0.0)
    line.add_argument("--target-snr", type=float, default=5.0)

    template = subparsers.add_parser(
        "template",
        help="Detector simulation from an observed wavelength/F_lambda template.",
    )
    template.add_argument("channel", choices=tuple(SPECTRAL_CHANNELS))
    template.add_argument("spectrum")
    template.add_argument("exposure_s", type=float)
    template.add_argument(
        "--native-R",
        type=float,
        help=(
            "Resolving power already present in the input spectrum. "
            "Omit only for an effectively unresolved model spectrum."
        ),
    )
    template.add_argument(
        "--uncertainty-column",
        type=int,
        help="Zero-based input column containing 1-sigma F_lambda uncertainty.",
    )
    template.add_argument("--ab-magnitude", type=float)
    template.add_argument("--normalization-min-A", type=float)
    template.add_argument("--normalization-max-A", type=float)
    template.add_argument("--source-fwhm-arcsec", type=float, default=0.0)
    template.add_argument("--contamination-flux-fraction", type=float, default=0.0)
    template.add_argument("--line-center-A", type=float)
    template.add_argument("--line-window-kms", type=float, default=1500.0)
    template.add_argument("--bin-R", type=float)
    template.add_argument("--output-csv")

    mir = subparsers.add_parser("mir", help="Cooled MIR point-source depth in uJy.")
    mir.add_argument("channel", choices=tuple(sorted(legacy.MIR_CHANNELS)))
    mir.add_argument("exposure_s", type=float)
    mir.add_argument("--flux-uJy", type=float)
    mir.add_argument("--target-snr", type=float, default=5.0)
    mir.add_argument("--background", choices=("low", "nominal", "high"), default="nominal")
    mir.add_argument("--observatory", choices=("3.5mST", "NEO Surveyor"), default="3.5mST")

    subparsers.add_parser("validate", help="Run deterministic regression checks.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "modes":
        _print_json(describe_modes())
        return 0
    if args.mode == "channels":
        _print_json({key: asdict(value) for key, value in SPECTRAL_CHANNELS.items()})
        return 0
    if args.mode == "imaging":
        _print_json(
            asdict(
                calculate_imaging(
                    args.filter,
                    args.exposure_s,
                    magnitude_ab=args.magnitude_ab,
                    target_snr=args.target_snr,
                )
            )
        )
        return 0
    if args.mode == "line":
        _print_json(
            asdict(
                calculate_line(
                    args.channel,
                    args.wavelength_A,
                    args.exposure_s,
                    line_flux_erg_s_cm2=args.flux_erg_s_cm2,
                    target_snr=args.target_snr,
                    continuum_magnitude_ab=args.continuum_magnitude_ab,
                    source_fwhm_arcsec=args.source_fwhm_arcsec,
                )
            )
        )
        return 0
    if args.mode == "mir":
        _print_json(
            asdict(
                calculate_mir_imaging(
                    args.channel,
                    args.exposure_s,
                    flux_uJy=args.flux_uJy,
                    target_snr=args.target_snr,
                    background=args.background,
                    observatory=args.observatory,
                )
            )
        )
        return 0
    if args.mode == "validate":
        result = run_validation()
        _print_json(result)
        return 0 if result["passed"] else 1

    spectrum = ObservedSpectrum.from_ascii(
        args.spectrum,
        uncertainty_column=args.uncertainty_column,
        native_resolving_power=args.native_R,
    )
    norm_values = (
        args.ab_magnitude,
        args.normalization_min_A,
        args.normalization_max_A,
    )
    if any(value is not None for value in norm_values):
        if not all(value is not None for value in norm_values):
            raise SystemExit(
                "--ab-magnitude, --normalization-min-A, and "
                "--normalization-max-A must be supplied together."
            )
        spectrum = spectrum.scaled_to_ab_magnitude(
            args.ab_magnitude,
            args.normalization_min_A,
            args.normalization_max_A,
        )
    result = simulate_spectrum(
        spectrum,
        args.channel,
        args.exposure_s,
        source_fwhm_arcsec=args.source_fwhm_arcsec,
        contamination_flux_fraction=args.contamination_flux_fraction,
    )
    summary: dict[str, Any] = {
        "metadata": dict(result.metadata),
        "n_detector_pixels": int(result.wavelength_A.size),
        "median_snr_per_pixel": float(np.median(result.snr)),
        "maximum_snr_per_pixel": float(np.max(result.snr)),
    }
    if args.output_csv:
        summary["output_csv"] = result.to_csv(args.output_csv)
    if args.bin_R:
        binned = bin_spectrum(result, args.bin_R)
        summary["binned"] = {
            "target_R": binned.target_resolving_power,
            "n_bins": int(binned.wavelength_A.size),
            "median_snr": float(np.median(binned.snr)),
        }
    if args.line_center_A:
        summary["line"] = asdict(
            measure_emission_line(
                result,
                args.line_center_A,
                window_kms=args.line_window_kms,
            )
        )
    _print_json(summary)
    return 0
