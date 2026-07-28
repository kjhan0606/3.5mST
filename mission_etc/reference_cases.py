"""Generate phase-2 ETC products from public U Gem and AT2017gfo spectra."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .monte_carlo import monte_carlo_line_recovery
from .planning import solve_exposure_time
from .spectroscopy import simulate_spectrum
from .templates import (
    combine_calibrated_spectra,
    fetch_engrave_xshooter_arm,
    fetch_lamost_ugem_outburst,
    mask_wavelength_ranges,
)


AT2017GFO_EPOCHS = (
    (1.43, 57983.969),
    (4.40, 57986.974),
    (7.40, 57990.000),
)


def _normalized_offset(
    wavelength_A: np.ndarray,
    flux_lambda: np.ndarray,
    offset: float,
) -> np.ndarray:
    finite = np.isfinite(flux_lambda)
    scale = np.nanpercentile(flux_lambda[finite], 90)
    return flux_lambda / scale + offset


def generate_reference_products(
    output_dir: str | Path,
    *,
    cache_dir: str | Path,
) -> tuple[Path, Path]:
    """Create a diagnostic figure and machine-readable feasibility table."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    ugem = fetch_lamost_ugem_outburst(cache).to_observed_spectrum()
    ugem_solution = solve_exposure_time(
        ugem,
        "compact-red",
        metric="line_snr",
        target_value=20.0,
        line_center_A=6562.8,
        line_window_kms=1200.0,
        minimum_exposure_s=0.05,
        maximum_exposure_s=1500.0,
        phase_ceiling_s=0.10 * 4.246 * 3600.0,
    )
    ugem_result = simulate_spectrum(
        ugem,
        "compact-red",
        ugem_solution.exposure_s,
    )
    ugem_mc = monte_carlo_line_recovery(
        ugem_result,
        6562.8,
        window_kms=1200.0,
        n_trials=300,
        random_seed=20260728,
    )

    optical_results = []
    nir_results = []
    table_rows: list[dict[str, object]] = []
    for phase_days, epoch_mjd in AT2017GFO_EPOCHS:
        uvb = fetch_engrave_xshooter_arm(
            cache,
            arm="UVB",
            epoch_mjd=epoch_mjd,
            phase_days=phase_days,
        )
        vis = fetch_engrave_xshooter_arm(
            cache,
            arm="VIS",
            epoch_mjd=epoch_mjd,
            phase_days=phase_days,
        )
        nir = fetch_engrave_xshooter_arm(
            cache,
            arm="NIR",
            epoch_mjd=epoch_mjd,
            phase_days=phase_days,
        )
        vis = mask_wavelength_ranges(
            vis,
            (
                (6860.0, 6950.0),
                (7580.0, 7700.0),
                (9300.0, 9600.0),
            ),
            reason="the ground-based telluric correction is unreliable",
        )
        nir = mask_wavelength_ranges(
            nir,
            ((13400.0, 14500.0),),
            reason="the ground-based water-vapour band leaves large residuals",
        )
        optical = combine_calibrated_spectra(
            (uvb, vis),
            joins_A=(5600.0,),
            key=f"at2017gfo-{phase_days:.2f}d-optical",
            state=f"{phase_days:+.2f} d after GW170817",
        ).to_observed_spectrum()
        nir_observed = nir.to_observed_spectrum()

        optical_solution = solve_exposure_time(
            optical,
            "transient-optical",
            metric="integrated_snr",
            target_value=20.0,
            wavelength_min_A=7425.0,
            wavelength_max_A=7575.0,
            minimum_exposure_s=1.0,
            maximum_exposure_s=50000.0,
        )
        nir_solution = solve_exposure_time(
            nir_observed,
            "transient-nir",
            metric="integrated_snr",
            target_value=20.0,
            wavelength_min_A=12375.0,
            wavelength_max_A=12625.0,
            minimum_exposure_s=1.0,
            maximum_exposure_s=50000.0,
        )
        optical_result = simulate_spectrum(
            optical,
            "transient-optical",
            optical_solution.exposure_s,
        )
        nir_result = simulate_spectrum(
            nir_observed,
            "transient-nir",
            nir_solution.exposure_s,
        )
        optical_results.append((phase_days, optical_result, optical_solution))
        nir_results.append((phase_days, nir_result, nir_solution))
        for band, solution, interval in (
            ("optical", optical_solution, "7425--7575 A"),
            ("NIR", nir_solution, "12375--12625 A"),
        ):
            table_rows.append(
                {
                    "object": "AT2017gfo",
                    "state_or_phase": f"{phase_days:+.2f} d",
                    "channel": band,
                    "metric": "integrated_snr",
                    "target": 20.0,
                    "diagnostic_interval": interval,
                    "exposure_s": solution.exposure_s,
                    "achieved": solution.achieved_value,
                    "feasible": solution.feasible,
                    "effective_R": (
                        optical_result.metadata["effective_resolving_power"]
                        if band == "optical"
                        else nir_result.metadata["effective_resolving_power"]
                    ),
                    "source": "ENGRAVE X-shooter v1.0",
                }
            )

    table_rows.insert(
        0,
        {
            "object": "U Gem",
            "state_or_phase": "2012 December outburst peak",
            "channel": "compact-red",
            "metric": "Halpha line_snr",
            "target": 20.0,
            "diagnostic_interval": "+/-1200 km/s",
            "exposure_s": ugem_solution.exposure_s,
            "achieved": ugem_solution.achieved_value,
            "feasible": ugem_solution.feasible,
            "effective_R": ugem_result.metadata["effective_resolving_power"],
            "source": "LAMOST DR11 obsid 89907231",
        },
    )
    table_path = output / "compact_etc_reference_cases.csv"
    with table_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    colors = ("#0072B2", "#D55E00", "#009E73")
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 8.5), constrained_layout=True)
    figure.patch.set_facecolor("white")

    ax = axes[0, 0]
    input_mask = (ugem.wavelength_A >= 6400.0) & (ugem.wavelength_A <= 6750.0)
    input_norm = np.nanmedian(ugem.flux_lambda[input_mask])
    detector_norm = np.nanmedian(ugem_result.convolved_flux_lambda)
    ax.plot(
        ugem.wavelength_A[input_mask],
        ugem.flux_lambda[input_mask] / input_norm,
        color="#777777",
        lw=1.1,
        label="LAMOST DR11, native R about 1800",
    )
    ax.plot(
        ugem_result.wavelength_A,
        ugem_result.convolved_flux_lambda / detector_norm,
        color="#D55E00",
        lw=1.8,
        label="Mission detector grid, no artificial sharpening",
    )
    ax.axvline(6562.8, color="#009E73", lw=1.0, alpha=0.8)
    ax.set(
        title="U Gem at outburst peak",
        xlabel="Observed wavelength [Angstrom]",
        ylabel="Normalized flux density",
        xlim=(6400.0, 6750.0),
    )
    ax.legend(frameon=False, fontsize=9)
    ax.text(
        0.02,
        0.05,
        (
            f"Halpha S/N=20 in {ugem_solution.exposure_s:.1f} s\n"
            f"MC centroid scatter={ugem_mc.centroid_scatter_kms:.1f} km/s"
        ),
        transform=ax.transAxes,
        fontsize=10,
        color="#333333",
    )

    ax = axes[0, 1]
    for index, (phase, result, _) in enumerate(optical_results):
        selected = (result.wavelength_A >= 4000.0) & (result.wavelength_A <= 10000.0)
        ax.plot(
            result.wavelength_A[selected] / 1.0e4,
            _normalized_offset(
                result.wavelength_A[selected],
                result.convolved_flux_lambda[selected],
                1.35 * (len(optical_results) - index - 1),
            ),
            color=colors[index],
            lw=1.3,
            label=f"{phase:.2f} d",
        )
    ax.set(
        title="AT2017gfo optical evolution at R=1000",
        xlabel="Observed wavelength [micron]",
        ylabel="Normalized flux and offset",
        xlim=(0.4, 1.0),
    )
    ax.legend(frameon=False, ncol=3, fontsize=9)
    ax.text(
        0.02,
        0.04,
        "Ground-based telluric intervals are quality-masked before forecasting.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#555555",
    )

    ax = axes[1, 0]
    for index, (phase, result, _) in enumerate(nir_results):
        selected = (result.wavelength_A >= 10000.0) & (result.wavelength_A <= 15000.0)
        ax.plot(
            result.wavelength_A[selected] / 1.0e4,
            _normalized_offset(
                result.wavelength_A[selected],
                result.convolved_flux_lambda[selected],
                1.35 * (len(nir_results) - index - 1),
            ),
            color=colors[index],
            lw=1.3,
            label=f"{phase:.2f} d",
        )
    ax.set(
        title="AT2017gfo near-infrared evolution at R=1000",
        xlabel="Observed wavelength [micron]",
        ylabel="Normalized flux and offset",
        xlim=(1.0, 1.5),
    )
    ax.legend(frameon=False, ncol=3, fontsize=9)

    ax = axes[1, 1]
    phases = np.asarray([item[0] for item in optical_results])
    optical_times = np.asarray([item[2].exposure_s for item in optical_results])
    nir_times = np.asarray([item[2].exposure_s for item in nir_results])
    ax.plot(phases, optical_times, "o-", color="#0072B2", lw=2, label="0.75 micron")
    ax.plot(phases, nir_times, "s-", color="#D55E00", lw=2, label="1.25 micron")
    ax.set_yscale("log")
    ax.set(
        title="Exposure for S/N=20 in one broad-feature bin",
        xlabel="Days after GW170817",
        ylabel="Exposure [s]",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)

    for ax in axes.flat:
        ax.tick_params(labelsize=10)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    figure_path = output / "compact_etc_reference_cases.png"
    figure.savefig(figure_path, dpi=220, facecolor="white")
    plt.close(figure)

    summary_path = output / "compact_etc_reference_summary.txt"
    summary_path.write_text(
        "\n".join(
            (
                "U Gem Monte Carlo line recovery",
                *(
                    f"{key}: {value}"
                    for key, value in asdict(ugem_mc).items()
                ),
                "",
                "All source spectra retain their public-data provenance in the CSV.",
            )
        )
        + "\n"
    )
    return figure_path, table_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--cache-dir", default="compact_template_cache")
    args = parser.parse_args(argv)
    figure, table = generate_reference_products(
        args.output_dir,
        cache_dir=args.cache_dir,
    )
    print(figure.resolve())
    print(table.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
