"""Generate the compact-object multi-roll injection and recovery test."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .profile import fit_template_velocity
from .scene import (
    SceneSource,
    extract_multi_roll_scene,
    realize_scene_noise,
    scale_spectrum,
    shift_spectrum_velocity,
    simulate_multi_roll_scene,
)
from .spectroscopy import simulate_spectrum
from .templates import fetch_lamost_ugem_outburst


_C_KMS = 299792.458
_INJECTED_VELOCITY_KMS = 120.0


def _recover_velocity(scene, base_template_e, roll_indices):
    extraction = extract_multi_roll_scene(
        scene,
        roll_indices=roll_indices,
        maximum_iterations=1200,
    )
    fit = fit_template_velocity(
        extraction.wavelength_A,
        extraction.spectra_e["U_Gem"],
        extraction.uncertainty_e["U_Gem"],
        base_template_e,
        line_center_A=6562.8,
        window_kms=1200.0,
        velocity_bound_kms=600.0,
        initial_velocity_kms=100.0,
    )
    return extraction, fit


def _summary(
    values: np.ndarray,
    reported: np.ndarray,
    n_trials: int,
) -> dict[str, float]:
    if values.size == 0:
        raise RuntimeError("No velocity fits were recovered.")
    return {
        "n_recovered": int(values.size),
        "recovery_fraction": float(values.size / n_trials),
        "median_velocity_kms": float(np.median(values)),
        "velocity_bias_kms": float(np.mean(values - _INJECTED_VELOCITY_KMS)),
        "velocity_scatter_kms": float(
            np.std(values, ddof=1 if values.size > 1 else 0)
        ),
        "median_reported_sigma_kms": float(np.median(reported)),
    }


def generate_scene_reference(
    output_dir: str | Path,
    *,
    cache_dir: str | Path,
    n_trials: int = 80,
) -> tuple[Path, Path]:
    """Run the U Gem injection test and write one figure and one CSV."""

    if n_trials < 10:
        raise ValueError("At least ten recovery trials are required.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    ugem = fetch_lamost_ugem_outburst(cache_dir).to_observed_spectrum()
    target = shift_spectrum_velocity(
        ugem,
        _INJECTED_VELOCITY_KMS,
        name="U Gem with injected velocity",
    )
    field_a = scale_spectrum(
        shift_spectrum_velocity(ugem, -220.0),
        0.70,
        name="field contaminant A",
    )
    field_b = scale_spectrum(
        shift_spectrum_velocity(ugem, 310.0),
        0.40,
        name="field contaminant B",
    )
    sources = (
        SceneSource("U_Gem", target, 0.0, 0.0),
        SceneSource("field_A", field_a, 0.75, 0.0),
        SceneSource("field_B", field_b, -0.45, 0.42),
    )
    total_exposure_s = 60.0
    roll_angles_deg = (0.0, 60.0, 120.0)
    exposure_s_per_roll = total_exposure_s / len(roll_angles_deg)
    one_roll_scene = simulate_multi_roll_scene(
        sources,
        "compact-red",
        total_exposure_s,
        target_id="U_Gem",
        roll_angles_deg=(0.0,),
        random_seed=20260728,
    )
    three_roll_scene = simulate_multi_roll_scene(
        sources,
        "compact-red",
        exposure_s_per_roll,
        target_id="U_Gem",
        roll_angles_deg=roll_angles_deg,
        random_seed=20260729,
    )
    one_roll_template = simulate_spectrum(
        ugem,
        "compact-red",
        total_exposure_s,
        extraction_eff=1.0,
        n_exp=1,
    )
    three_roll_template = simulate_spectrum(
        ugem,
        "compact-red",
        exposure_s_per_roll,
        extraction_eff=1.0,
        n_exp=1,
    )
    one_extraction, _ = _recover_velocity(
        one_roll_scene,
        one_roll_template.source_e,
        None,
    )
    all_extraction, _ = _recover_velocity(
        three_roll_scene,
        three_roll_template.source_e,
        None,
    )

    one_values: list[float] = []
    one_reported: list[float] = []
    all_values: list[float] = []
    all_reported: list[float] = []
    for trial in range(n_trials):
        one_realization = realize_scene_noise(
            one_roll_scene,
            random_seed=810000 + trial,
        )
        three_realization = realize_scene_noise(
            three_roll_scene,
            random_seed=910000 + trial,
        )
        try:
            _, fit = _recover_velocity(
                one_realization,
                one_roll_template.source_e,
                None,
            )
            if fit.success and np.isfinite(fit.velocity_kms):
                one_values.append(fit.velocity_kms)
                one_reported.append(fit.velocity_sigma_kms)
        except (ValueError, np.linalg.LinAlgError):
            pass
        try:
            _, fit = _recover_velocity(
                three_realization,
                three_roll_template.source_e,
                None,
            )
            if fit.success and np.isfinite(fit.velocity_kms):
                all_values.append(fit.velocity_kms)
                all_reported.append(fit.velocity_sigma_kms)
        except (ValueError, np.linalg.LinAlgError):
            pass

    one_array = np.asarray(one_values)
    all_array = np.asarray(all_values)
    one_summary = _summary(one_array, np.asarray(one_reported), n_trials)
    all_summary = _summary(all_array, np.asarray(all_reported), n_trials)
    table_path = output / "compact_scene_recovery.csv"
    with table_path.open("w", newline="") as stream:
        fieldnames = (
            "roll_mode",
            "roll_angles_deg",
            "n_trials",
            "n_recovered",
            "recovery_fraction",
            "injected_velocity_kms",
            "median_velocity_kms",
            "velocity_bias_kms",
            "velocity_scatter_kms",
            "median_reported_sigma_kms",
            "effective_resolving_power",
            "number_of_rolls",
            "exposure_s_per_roll",
            "total_exposure_s",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for mode, angles, number_of_rolls, exposure, result, summary in (
            (
                "one_roll",
                "0",
                1,
                total_exposure_s,
                one_roll_template,
                one_summary,
            ),
            (
                "three_roll_joint",
                "0 60 120",
                3,
                exposure_s_per_roll,
                three_roll_template,
                all_summary,
            ),
        ):
            writer.writerow(
                {
                    "roll_mode": mode,
                    "roll_angles_deg": angles,
                    "n_trials": n_trials,
                    **summary,
                    "injected_velocity_kms": _INJECTED_VELOCITY_KMS,
                    "effective_resolving_power": result.metadata[
                        "effective_resolving_power"
                    ],
                    "number_of_rolls": number_of_rolls,
                    "exposure_s_per_roll": exposure,
                    "total_exposure_s": total_exposure_s,
                }
            )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14.0, 8.7),
        constrained_layout=True,
    )
    figure.patch.set_facecolor("white")

    ax = axes[0, 0]
    ax.imshow(three_roll_scene.direct_prior, origin="lower", cmap="gray_r")
    center = 0.5 * three_roll_scene.direct_prior.shape[0]
    pixel_scale = float(three_roll_scene.metadata["pixel_scale_arcsec"])
    positions = (
        ("U Gem", 0.0, 0.0, "#D55E00", (-8, -18)),
        ("field A", 0.75, 0.0, "#0072B2", (8, 8)),
        ("field B", -0.45, 0.42, "#009E73", (-42, 10)),
    )
    for label, x_arcsec, y_arcsec, color, text_offset in positions:
        x_pixel = center + x_arcsec / pixel_scale
        y_pixel = center + y_arcsec / pixel_scale
        ax.plot(x_pixel, y_pixel, marker="o", ms=10, mfc="none", mec=color, mew=2)
        ax.annotate(
            label,
            (x_pixel, y_pixel),
            xytext=text_offset,
            textcoords="offset points",
            color=color,
            fontsize=10,
        )
    ax.set_title("Direct-image position prior")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[0, 1]
    exposure = three_roll_scene.exposures[0]
    line_index = int(np.argmin(np.abs(three_roll_scene.wavelength_A - 6562.8)))
    line_x = exposure.trace_start_x["U_Gem"] + line_index
    x_min = max(0, int(line_x) - 125)
    x_max = min(exposure.image_shape[1], int(line_x) + 126)
    detector = (
        exposure.image_e[:, x_min:x_max]
        - exposure.background_e_per_pixel
    )
    scale = np.percentile(np.abs(detector), 98)
    display = np.arcsinh(detector / max(scale, np.finfo(float).tiny) * 6.0)
    ax.imshow(display, origin="lower", cmap="magma", aspect="auto")
    target_y = exposure.trace_y["U_Gem"]
    ax.plot(line_x - x_min, target_y, marker="+", ms=14, mew=2, color="#00E5FF")
    ax.text(
        line_x - x_min + 5,
        target_y + 3,
        "injected H-alpha",
        color="#00E5FF",
        fontsize=10,
    )
    ax.set(
        title="One 20 s detector image with overlapping traces",
        xlabel="Dispersion pixel",
        ylabel="Cross-dispersion pixel",
    )

    ax = axes[1, 0]
    profile_mask = (
        np.abs(
            _C_KMS
            * (three_roll_scene.wavelength_A / 6562.8 - 1.0)
        )
        <= 1200.0
    )
    wave = three_roll_scene.wavelength_A[profile_mask]
    truth = three_roll_scene.truth_spectra_e["U_Gem"][profile_mask]
    one_truth = one_roll_scene.truth_spectra_e["U_Gem"][profile_mask]
    one = one_extraction.spectra_e["U_Gem"][profile_mask]
    all_rolls = all_extraction.spectra_e["U_Gem"][profile_mask]
    truth_normalization = np.median(truth)
    one_normalization = np.median(one_truth)
    ax.plot(
        wave,
        truth / truth_normalization,
        color="#222222",
        lw=2.2,
        label="injected",
    )
    ax.plot(
        wave,
        one / one_normalization,
        color="#0072B2",
        lw=1.2,
        label="one roll",
    )
    ax.plot(
        wave,
        all_rolls / truth_normalization,
        color="#D55E00",
        lw=1.5,
        label="three-roll joint fit",
    )
    injected_line = 6562.8 * (1.0 + _INJECTED_VELOCITY_KMS / _C_KMS)
    ax.axvline(injected_line, color="#009E73", ls="--", lw=1.2)
    ax.set(
        title="Recovered U Gem H-alpha profile",
        xlabel="Observed wavelength [Angstrom]",
        ylabel="Counts divided by continuum",
    )
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    bins = np.linspace(
        min(np.min(one_array), np.min(all_array), 0.0) - 10.0,
        max(np.max(one_array), np.max(all_array), 180.0) + 10.0,
        24,
    )
    ax.hist(
        one_array,
        bins=bins,
        color="#0072B2",
        alpha=0.65,
        label=(
            "one roll, "
            f"bias {one_summary['velocity_bias_kms']:+.1f}, "
            f"scatter {one_summary['velocity_scatter_kms']:.1f} km/s"
        ),
    )
    ax.hist(
        all_array,
        bins=bins,
        color="#D55E00",
        alpha=0.72,
        label=(
            "three rolls, "
            f"bias {all_summary['velocity_bias_kms']:+.1f}, "
            "scatter "
            f"{all_summary['velocity_scatter_kms']:.1f} km/s"
        ),
    )
    ax.axvline(
        _INJECTED_VELOCITY_KMS,
        color="#009E73",
        ls="--",
        lw=2,
        label="injected 120 km/s",
    )
    ax.set(
        title=f"Fixed 60 s total time over {n_trials} noise realizations",
        xlabel="Recovered velocity [km/s]",
        ylabel="Number of realizations",
    )
    ax.legend(frameon=False, fontsize=9)

    for ax in axes.flat:
        ax.tick_params(labelsize=10)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    figure_path = output / "compact_scene_recovery.png"
    figure.savefig(figure_path, dpi=220, facecolor="white")
    plt.close(figure)
    return figure_path, table_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="appendix_x_assets")
    parser.add_argument("--cache-dir", default="compact_template_cache")
    parser.add_argument("--trials", type=int, default=80)
    args = parser.parse_args(argv)
    figure, table = generate_scene_reference(
        args.output_dir,
        cache_dir=args.cache_dir,
        n_trials=args.trials,
    )
    print(figure.resolve())
    print(table.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
