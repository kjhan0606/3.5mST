#!/usr/bin/env python3
"""Plot the Earth-centered reflected-light detection reach from L2."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import slitless_etc as etc
from neo_population_model import hg_phase_function


OUTPUT = Path(__file__).resolve().parent / "appendixC_assets" / (
    "neo_single_visit_reach.png"
)

EARTH_X_AU = 0.0
OBSERVER_X_AU = 0.01
SUN_X_AU = -1.0
EXPOSURE_S = 145.0
FILTER_NAME = "Johnson V"
FILTER_PIVOT_UM, FILTER_WIDTH_UM = etc.STANDARD_FILTERS[FILTER_NAME]
ELONGATION_MIN_DEG = 60.0
ELONGATION_MAX_DEG = 180.0
ALBEDOS = (0.15,)
DIAMETERS_M = (5.0, 10.0, 20.0, 50.0, 140.0, 1000.0)
DIAMETER_COLORS = {
    5.0: "#245AA6",
    10.0: "#1B9ED1",
    20.0: "#007C77",
    50.0: "#E0A100",
    140.0: "#C7472F",
    1000.0: "#7B3F98",
}


def geometry_grid():
    """Return a log-polar Earth grid and the L2 observing geometry."""
    theta = np.linspace(-np.pi, np.pi, 721)
    earth_distance_au = np.geomspace(0.003, 10.0, 640)
    theta_grid, earth_distance_grid = np.meshgrid(theta, earth_distance_au)
    x_grid = earth_distance_grid * np.cos(theta_grid)
    y_grid = earth_distance_grid * np.sin(theta_grid)
    l2_x = x_grid - OBSERVER_X_AU
    l2_y = y_grid
    delta_au = np.hypot(l2_x, l2_y)
    r_au = np.hypot(x_grid - SUN_X_AU, y_grid)
    delta_safe = np.maximum(delta_au, 1.0e-5)
    r_safe = np.maximum(r_au, 1.0e-5)

    # Solar elongation is measured at L2 from the Sunward horizontal direction.
    elongation_deg = np.degrees(
        np.arccos(
            np.clip(
                -l2_x / delta_safe,
                -1.0,
                1.0,
            )
        )
    )

    asteroid_to_sun_x = SUN_X_AU - x_grid
    asteroid_to_sun_y = -y_grid
    asteroid_to_l2_x = OBSERVER_X_AU - x_grid
    asteroid_to_l2_y = -y_grid
    phase_cosine = (
        asteroid_to_sun_x * asteroid_to_l2_x
        + asteroid_to_sun_y * asteroid_to_l2_y
    ) / (r_safe * delta_safe)
    phase_deg = np.degrees(
        np.arccos(np.clip(phase_cosine, -1.0, 1.0))
    )

    field_of_regard = (
        (elongation_deg >= ELONGATION_MIN_DEG)
        & (elongation_deg <= ELONGATION_MAX_DEG)
        & (earth_distance_grid >= 0.003)
        & (r_au >= 0.08)
    )
    geometry = {
        "r_au": r_safe.ravel(),
        "delta_au": delta_safe.ravel(),
        "phase_deg": phase_deg.ravel(),
    }
    return (
        theta_grid,
        earth_distance_grid,
        elongation_deg,
        field_of_regard,
        geometry,
    )


def detected_mask(
    diameter_m,
    albedo,
    geometry,
    field_of_regard,
    limiting_magnitude,
):
    absolute_magnitude = 5.0 * np.log10(
        1329.0 / ((diameter_m / 1000.0) * np.sqrt(albedo))
    )
    phase_function = hg_phase_function(geometry["phase_deg"])
    apparent_magnitude = (
        absolute_magnitude
        + 5.0
        * np.log10(geometry["r_au"] * geometry["delta_au"])
        - 2.5 * np.log10(np.maximum(phase_function, 1.0e-12))
    ).reshape(field_of_regard.shape)
    return field_of_regard & (
        apparent_magnitude <= limiting_magnitude
    )


def maximum_distance_au(albedo, diameter_m, limiting_magnitude):
    """Resolve the maximum geocentric reach of an L2 detection."""
    theta = np.linspace(-np.pi, np.pi, 1441)
    earth_distance = np.geomspace(0.001, 30.0, 4200)
    theta_grid, earth_distance_grid = np.meshgrid(
        theta, earth_distance, indexing="ij"
    )
    x_grid = earth_distance_grid * np.cos(theta_grid)
    y_grid = earth_distance_grid * np.sin(theta_grid)
    l2_x = x_grid - OBSERVER_X_AU
    l2_y = y_grid
    delta_grid = np.hypot(l2_x, l2_y)
    r_au = np.hypot(x_grid - SUN_X_AU, y_grid)

    asteroid_to_sun_x = SUN_X_AU - x_grid
    asteroid_to_sun_y = -y_grid
    asteroid_to_l2_x = OBSERVER_X_AU - x_grid
    asteroid_to_l2_y = -y_grid
    phase_cosine = (
        asteroid_to_sun_x * asteroid_to_l2_x
        + asteroid_to_sun_y * asteroid_to_l2_y
    ) / (r_au * delta_grid)
    phase_deg = np.degrees(
        np.arccos(np.clip(phase_cosine, -1.0, 1.0))
    )
    absolute_magnitude = 5.0 * np.log10(
        1329.0 / ((diameter_m / 1000.0) * np.sqrt(albedo))
    )
    apparent_magnitude = (
        absolute_magnitude
        + 5.0 * np.log10(r_au * delta_grid)
        - 2.5 * np.log10(np.maximum(hg_phase_function(phase_deg), 1.0e-12))
    )
    elongation_deg = np.degrees(
        np.arccos(np.clip(-l2_x / np.maximum(delta_grid, 1e-12), -1.0, 1.0))
    )
    detected = (
        (apparent_magnitude <= limiting_magnitude)
        & (elongation_deg >= ELONGATION_MIN_DEG)
    )
    return float(np.max(earth_distance_grid[detected]))


def add_l2_earth_markers(axis):
    axis.scatter(
        [np.pi],
        [1.0],
        s=180,
        color="#FFD447",
        edgecolor="#B86B00",
        linewidth=1.2,
        zorder=8,
    )
    axis.annotate(
        "Sun\n1 au",
        (np.pi, 1.0),
        xytext=(-12, -30),
        textcoords="offset points",
        ha="center",
        va="top",
        color="#8C5200",
        fontsize=8.5,
        weight="bold",
    )
    axis.scatter(
        [0.0],
        [OBSERVER_X_AU],
        s=74,
        marker="D",
        color="#FF4FA3",
        edgecolor="#151B23",
        linewidth=1.0,
        zorder=9,
    )
    axis.annotate(
        "L2\n0.01 au",
        (0.0, OBSERVER_X_AU),
        xytext=(24, 38),
        textcoords="offset points",
        ha="left",
        va="bottom",
        color="#A51F69",
        fontsize=8.2,
        weight="bold",
        arrowprops={
            "arrowstyle": "->",
            "color": "#A51F69",
            "lw": 0.8,
            "shrinkA": 2,
            "shrinkB": 4,
        },
    )
    axis.scatter(
        [0.5],
        [0.5],
        transform=axis.transAxes,
        marker="o",
        s=92,
        color="#2E73C5",
        edgecolor="#151B23",
        linewidth=1.0,
        zorder=12,
        clip_on=False,
    )
    axis.text(
        0.47,
        0.47,
        "Earth origin",
        transform=axis.transAxes,
        ha="right",
        va="top",
        color="#245AA6",
        fontsize=9.3,
        weight="bold",
        zorder=13,
    )


def main():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "#F7F8F9",
            "axes.edgecolor": "#687582",
            "axes.labelcolor": "#151B23",
            "xtick.color": "#4F5D6A",
            "ytick.color": "#4F5D6A",
            "text.color": "#151B23",
            "axes.titleweight": "bold",
        }
    )
    (
        theta_grid,
        delta_grid,
        _,
        field_of_regard,
        geometry,
    ) = geometry_grid()
    limiting_magnitude = etc.imaging_maglimit(
        etc.segmented_cfg(),
        FILTER_PIVOT_UM * 1.0e4,
        FILTER_WIDTH_UM * 1.0e4,
        EXPOSURE_S,
        snr=5.0,
    )

    figure, axes = plt.subplots(
        1,
        1,
        figsize=(9.2, 8.0),
        subplot_kw={"projection": "polar"},
        constrained_layout=False,
    )
    reach_by_albedo = {}

    axes = np.atleast_1d(axes)
    for axis, albedo in zip(axes, ALBEDOS):
        reach_by_albedo[albedo] = {}
        for index, diameter_m in enumerate(reversed(DIAMETERS_M)):
            mask = detected_mask(
                diameter_m,
                albedo,
                geometry,
                field_of_regard,
                limiting_magnitude,
            )
            color = DIAMETER_COLORS[diameter_m]
            axis.contourf(
                theta_grid,
                delta_grid,
                mask.astype(float),
                levels=(0.5, 1.5),
                colors=(color,),
                alpha=0.20 if index == 0 else 0.28,
                zorder=3,
            )
            axis.contour(
                theta_grid,
                delta_grid,
                mask.astype(float),
                levels=(0.5,),
                colors=(color,),
                linewidths=2.0,
                zorder=5,
            )
            reach_by_albedo[albedo][diameter_m] = maximum_distance_au(
                albedo, diameter_m, limiting_magnitude
            )

        # Show the baseline Sun-avoidance boundary without treating it as
        # a sensitivity limit.
        axis.contour(
            theta_grid,
            delta_grid,
            field_of_regard.astype(float),
            levels=(0.5,),
            colors=("#687582",),
            linestyles="--",
            linewidths=1.0,
            zorder=4,
        )
        add_l2_earth_markers(axis)
        axis.set_theta_zero_location("E")
        axis.set_theta_direction(1)
        axis.set_rscale("log")
        axis.set_rlim(0.003, 10.0)
        axis.set_rticks((0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0))
        axis.set_yticklabels(("0.01", "0.03", "0.1", "0.3", "1", "3", "10"))
        axis.set_rlabel_position(72)
        axis.set_thetagrids(
            (0, 60, 120, 180, 240, 300),
            ("L2 / opposition", "", "", "Sunward", "", ""),
            fontsize=8.5,
        )
        axis.grid(color="#8E99A5", alpha=0.16, lw=0.7)

        rows = []
        for diameter_m in DIAMETERS_M:
            label = (
                rf"{diameter_m:.0f}\,\mathrm{{m}}"
                if diameter_m < 1000.0
                else rf"{diameter_m / 1000.0:.0f}\,\mathrm{{km}}"
            )
            rows.append(
                rf"$D={label}$   "
                rf"$d_{{\oplus,\max}}={reach_by_albedo[albedo][diameter_m]:.2f}$ au"
            )
        summary_text = "\n".join(
            (
                r"$p_V=0.15$",
                rf"$t_{{\rm exp}}={EXPOSURE_S:.0f}\,\mathrm{{s}}$",
                rf"$V_{{AB,5\sigma}}={limiting_magnitude:.2f}$",
                "",
                *rows,
            )
        )

    figure.suptitle(
        "Asteroid detection area",
        fontsize=20,
        weight="bold",
        y=0.965,
    )
    figure.text(
        0.025,
        0.73,
        summary_text,
        ha="left",
        va="top",
        fontsize=10.2,
        linespacing=1.48,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "edgecolor": "#BAC2CA",
            "alpha": 0.98,
        },
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=DIAMETER_COLORS[diameter_m],
            lw=3,
            label=(
                f"{diameter_m:.0f} m asteroid"
                if diameter_m < 1000.0
                else "1 km reference"
            ),
        )
        for diameter_m in DIAMETERS_M
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#687582",
            lw=1.3,
            ls="--",
            label="60 deg Sun-avoidance boundary",
        )
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.070),
        fontsize=10.0,
    )
    figure.text(
        0.5,
        0.013,
        (
            "Contours use the standard H-diameter-albedo relation, the H,G "
            "phase law with G=0.15, and the proposal optical ETC. "
            "They exclude trailing, confusion, and orbit-linking losses."
        ),
        ha="center",
        color="#5C6875",
        fontsize=9.2,
    )
    figure.subplots_adjust(
        left=0.30,
        right=0.965,
        bottom=0.225,
        top=0.90,
        wspace=0.07,
    )
    figure.supxlabel(
        "Radial coordinate is distance from Earth [au]",
        x=0.52,
        y=0.154,
        fontsize=12,
        color="#151B23",
    )
    OUTPUT.parent.mkdir(exist_ok=True)
    figure.savefig(OUTPUT, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(figure)

    print(f"Wrote {OUTPUT}")
    print(f"{FILTER_NAME} 5-sigma limit in {EXPOSURE_S:.0f} s: "
          f"{limiting_magnitude:.3f} AB mag")
    for albedo in ALBEDOS:
        values = ", ".join(
            f"{diameter_m:g} m: {reach_by_albedo[albedo][diameter_m]:.3f} au"
            for diameter_m in DIAMETERS_M
        )
        print(f"p_V={albedo:.2f}: {values}")


if __name__ == "__main__":
    main()
