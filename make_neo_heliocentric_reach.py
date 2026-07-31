#!/usr/bin/env python3
"""Plot the reflected-light detection area in Sun-centered coordinates."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.textpath import TextPath

import slitless_etc as etc
from neo_population_model import hg_phase_function


OUTPUT = Path(__file__).resolve().parent / "appendixC_assets" / (
    "neo_single_visit_reach_heliocentric.png"
)

EARTH_ORBIT_AU = 1.0
L2_RADIUS_AU = 1.01
MARS_ORBIT_AU = 1.524
MAIN_BELT_INNER_AU = 2.1
MAIN_BELT_OUTER_AU = 3.3
JUPITER_ORBIT_AU = 5.204

EXPOSURE_S = 145.0
FILTER_NAME = "Johnson V"
FILTER_PIVOT_UM, FILTER_WIDTH_UM = etc.STANDARD_FILTERS[FILTER_NAME]
FIGURE_C8_LIMITING_MAGNITUDE = 26.39
ELONGATION_MIN_DEG = 60.0
ALBEDO = 0.15
DIAMETERS_M = (5.0, 10.0, 20.0, 50.0, 140.0, 1000.0)
DIAMETER_COLORS = {
    5.0: "#245AA6",
    10.0: "#1B9ED1",
    20.0: "#007C77",
    50.0: "#E0A100",
    140.0: "#C7472F",
    1000.0: "#7B3F98",
}


def heliocentric_geometry():
    """Return a Sun-centered log-polar grid and L2 observing geometry."""
    theta = np.linspace(-np.pi, np.pi, 961)
    radius_au = np.geomspace(0.08, 10.0, 720)
    theta_grid, radius_grid = np.meshgrid(theta, radius_au)

    x_grid = radius_grid * np.cos(theta_grid)
    y_grid = radius_grid * np.sin(theta_grid)
    observer_dx = x_grid - L2_RADIUS_AU
    observer_dy = y_grid
    delta_au = np.hypot(observer_dx, observer_dy)
    delta_safe = np.maximum(delta_au, 1.0e-6)

    # Solar elongation is the angle, at L2, between the Sun and the target.
    elongation_cosine = (L2_RADIUS_AU - x_grid) / delta_safe
    elongation_deg = np.degrees(
        np.arccos(np.clip(elongation_cosine, -1.0, 1.0))
    )

    # The phase angle is measured at the asteroid between Sun and observer.
    asteroid_to_sun_x = -x_grid
    asteroid_to_sun_y = -y_grid
    asteroid_to_l2_x = L2_RADIUS_AU - x_grid
    asteroid_to_l2_y = -y_grid
    phase_cosine = (
        asteroid_to_sun_x * asteroid_to_l2_x
        + asteroid_to_sun_y * asteroid_to_l2_y
    ) / (radius_grid * delta_safe)
    phase_deg = np.degrees(np.arccos(np.clip(phase_cosine, -1.0, 1.0)))

    field_of_regard = elongation_deg >= ELONGATION_MIN_DEG
    geometry = {
        "r_au": radius_grid,
        "delta_au": delta_safe,
        "phase_deg": phase_deg,
    }
    return theta_grid, radius_grid, field_of_regard, geometry


def detected_mask(diameter_m, geometry, field_of_regard, limiting_magnitude):
    """Return the instantaneous S/N >= 5 mask for one asteroid diameter."""
    absolute_magnitude = 5.0 * np.log10(
        1329.0 / ((diameter_m / 1000.0) * np.sqrt(ALBEDO))
    )
    phase_function = hg_phase_function(geometry["phase_deg"])
    apparent_magnitude = (
        absolute_magnitude
        + 5.0 * np.log10(geometry["r_au"] * geometry["delta_au"])
        - 2.5 * np.log10(np.maximum(phase_function, 1.0e-12))
    )
    return field_of_regard & (apparent_magnitude <= limiting_magnitude)


def earth_centered_reach_au(limiting_magnitude):
    """Return the favorable maximum Earth-centered reach by diameter."""
    theta = np.linspace(-np.pi, np.pi, 901)
    earth_distance_au = np.geomspace(0.001, 12.0, 3000)
    theta_grid, distance_grid = np.meshgrid(
        theta,
        earth_distance_au,
        indexing="ij",
    )
    x_grid = distance_grid * np.cos(theta_grid)
    y_grid = distance_grid * np.sin(theta_grid)

    observer_x = L2_RADIUS_AU - EARTH_ORBIT_AU
    delta_au = np.hypot(x_grid - observer_x, y_grid)
    heliocentric_distance_au = np.hypot(x_grid + EARTH_ORBIT_AU, y_grid)
    delta_safe = np.maximum(delta_au, 1.0e-8)
    radius_safe = np.maximum(heliocentric_distance_au, 1.0e-8)

    asteroid_to_sun_x = -EARTH_ORBIT_AU - x_grid
    asteroid_to_sun_y = -y_grid
    asteroid_to_l2_x = observer_x - x_grid
    asteroid_to_l2_y = -y_grid
    phase_cosine = (
        asteroid_to_sun_x * asteroid_to_l2_x
        + asteroid_to_sun_y * asteroid_to_l2_y
    ) / (radius_safe * delta_safe)
    phase_deg = np.degrees(np.arccos(np.clip(phase_cosine, -1.0, 1.0)))
    phase_function = hg_phase_function(phase_deg)

    elongation_cosine = -(x_grid - observer_x) / delta_safe
    elongation_deg = np.degrees(
        np.arccos(np.clip(elongation_cosine, -1.0, 1.0))
    )
    accessible = (
        (elongation_deg >= ELONGATION_MIN_DEG)
        & (heliocentric_distance_au >= 0.08)
    )
    geometry_term = (
        5.0 * np.log10(radius_safe * delta_safe)
        - 2.5 * np.log10(np.maximum(phase_function, 1.0e-12))
    )

    reach = {}
    for diameter_m in DIAMETERS_M:
        absolute_magnitude = 5.0 * np.log10(
            1329.0 / ((diameter_m / 1000.0) * np.sqrt(ALBEDO))
        )
        detected = accessible & (
            absolute_magnitude + geometry_term <= limiting_magnitude
        )
        reach[diameter_m] = float(np.max(distance_grid[detected]))
    return reach


def add_orbit_guides(axis, theta):
    """Draw Solar System orbit guides behind the detection contours."""
    axis.fill_between(
        theta,
        MAIN_BELT_INNER_AU,
        MAIN_BELT_OUTER_AU,
        color="#81715E",
        alpha=0.14,
        zorder=1,
    )
    axis.plot(
        theta,
        np.full_like(theta, EARTH_ORBIT_AU),
        color="#2E73C5",
        lw=1.4,
        ls="-.",
        zorder=7,
    )
    axis.plot(
        theta,
        np.full_like(theta, MARS_ORBIT_AU),
        color="#D45A3A",
        lw=1.4,
        ls="-.",
        zorder=7,
    )
    axis.plot(
        theta,
        np.full_like(theta, MAIN_BELT_INNER_AU),
        color="#6F6253",
        lw=1.0,
        ls=":",
        zorder=7,
    )
    axis.plot(
        theta,
        np.full_like(theta, MAIN_BELT_OUTER_AU),
        color="#6F6253",
        lw=1.0,
        ls=":",
        zorder=7,
    )
    axis.plot(
        theta,
        np.full_like(theta, JUPITER_ORBIT_AU),
        color="#75539A",
        lw=1.5,
        ls="-.",
        zorder=7,
    )


def add_arc_text(axis, text, radius_au, color, span_deg):
    """Place text along an orbit arc centered on the nine-o'clock direction."""
    font_size = 9.0
    font_properties = FontProperties(family="DejaVu Sans", weight="bold")
    glyphs = list(text)
    glyph_widths = []
    for glyph in glyphs:
        if glyph == " ":
            glyph_widths.append(0.38 * font_size)
            continue
        bounds = TextPath(
            (0.0, 0.0),
            glyph,
            size=font_size,
            prop=font_properties,
        ).get_extents()
        glyph_widths.append(max(bounds.width, 0.20 * font_size))
    weights = np.asarray(glyph_widths)
    cumulative = np.cumsum(weights) - 0.5 * weights
    offsets = (cumulative / weights.sum() - 0.5) * np.radians(span_deg)
    angles = np.pi + offsets[::-1]
    for glyph, angle in zip(glyphs, angles):
        if glyph == " ":
            continue
        rotation = np.degrees(angle) - 90.0
        axis.text(
            angle,
            radius_au,
            glyph,
            ha="center",
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            color=color,
            fontsize=font_size,
            weight="bold",
            path_effects=[
                path_effects.withStroke(linewidth=2.8, foreground="white")
            ],
            zorder=19,
        )


def add_solar_system_labels(axis):
    """Label the center, the observer, and the orbit guides."""
    axis.scatter(
        [0.5],
        [0.5],
        transform=axis.transAxes,
        s=210,
        color="#FFD447",
        edgecolor="#B86B00",
        linewidth=1.2,
        zorder=15,
        clip_on=False,
    )
    axis.text(
        0.5,
        0.455,
        "Sun",
        transform=axis.transAxes,
        ha="center",
        va="top",
        color="#8C5200",
        fontsize=9.5,
        weight="bold",
        zorder=16,
    )
    axis.scatter(
        [0.0],
        [EARTH_ORBIT_AU],
        s=90,
        color="#2E73C5",
        edgecolor="#151B23",
        linewidth=1.0,
        zorder=16,
    )
    axis.scatter(
        [0.0],
        [L2_RADIUS_AU],
        s=72,
        marker="D",
        color="#FF4FA3",
        edgecolor="#151B23",
        linewidth=0.9,
        zorder=17,
    )
    axis.annotate(
        "Earth and L2\n1.00 and 1.01 au",
        (0.0, L2_RADIUS_AU),
        xytext=(30, 34),
        textcoords="offset points",
        ha="left",
        va="bottom",
        color="#A51F69",
        fontsize=8.8,
        weight="bold",
        arrowprops={
            "arrowstyle": "->",
            "color": "#A51F69",
            "lw": 0.9,
            "shrinkA": 2,
            "shrinkB": 4,
        },
        zorder=18,
    )

    add_arc_text(
        axis,
        "Mars orbit  1.52 au",
        1.72,
        "#B6462E",
        span_deg=49,
    )
    add_arc_text(
        axis,
        "Main belt  2.1 to 3.3 au",
        2.60,
        "#645647",
        span_deg=60,
    )
    add_arc_text(
        axis,
        "Jupiter orbit  5.20 au",
        5.85,
        "#68468C",
        span_deg=48,
    )


def add_earth_reach_inset(figure, reach_by_diameter):
    """Add a compact log-scale summary that supports warning-time estimates."""
    inset = figure.add_axes((0.028, 0.295, 0.225, 0.285))
    y_positions = np.arange(len(DIAMETERS_M))
    for y_position, diameter_m in zip(y_positions, DIAMETERS_M):
        reach_au = reach_by_diameter[diameter_m]
        color = DIAMETER_COLORS[diameter_m]
        inset.hlines(
            y_position,
            0.1,
            reach_au,
            color=color,
            linewidth=2.4,
            zorder=2,
        )
        inset.scatter(
            reach_au,
            y_position,
            s=30,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        inset.annotate(
            f"{reach_au:.2f}",
            (reach_au, y_position),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.4,
            color="#2E3944",
        )

    inset.set_xscale("log")
    inset.set_xlim(0.1, 10.0)
    inset.set_ylim(-0.65, len(DIAMETERS_M) - 0.35)
    inset.set_xticks((0.1, 0.3, 1.0, 3.0, 10.0))
    inset.set_xticklabels(("0.1", "0.3", "1", "3", "10"), fontsize=7.2)
    inset.set_yticks(y_positions)
    inset.set_yticklabels(
        ("5 m", "10 m", "20 m", "50 m", "140 m", "1 km"),
        fontsize=7.4,
    )
    inset.set_title(
        "Favorable Earth-centered reach [au]",
        fontsize=8.8,
        weight="bold",
        pad=5,
    )
    inset.set_xlabel("Maximum distance from Earth", fontsize=7.5, labelpad=2)
    inset.grid(axis="x", color="#8E99A5", alpha=0.25, linewidth=0.6)
    inset.tick_params(axis="both", length=2.5, width=0.6, pad=2)
    for spine in inset.spines.values():
        spine.set_color("#AEB7C0")
        spine.set_linewidth(0.8)
    inset.set_facecolor("white")


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
    theta_grid, radius_grid, field_of_regard, geometry = heliocentric_geometry()
    current_etc_limiting_magnitude = etc.imaging_maglimit(
        etc.segmented_cfg(),
        FILTER_PIVOT_UM * 1.0e4,
        FILTER_WIDTH_UM * 1.0e4,
        EXPOSURE_S,
        snr=5.0,
    )
    limiting_magnitude = FIGURE_C8_LIMITING_MAGNITUDE
    earth_reach = earth_centered_reach_au(limiting_magnitude)

    figure, axis = plt.subplots(
        1,
        1,
        figsize=(10.4, 8.3),
        subplot_kw={"projection": "polar"},
        constrained_layout=False,
    )
    theta = theta_grid[0]
    add_orbit_guides(axis, theta)

    for index, diameter_m in enumerate(reversed(DIAMETERS_M)):
        mask = detected_mask(
            diameter_m,
            geometry,
            field_of_regard,
            limiting_magnitude,
        )
        color = DIAMETER_COLORS[diameter_m]
        axis.contourf(
            theta_grid,
            radius_grid,
            mask.astype(float),
            levels=(0.5, 1.5),
            colors=(color,),
            alpha=0.16 if index == 0 else 0.24,
            zorder=3,
        )
        axis.contour(
            theta_grid,
            radius_grid,
            mask.astype(float),
            levels=(0.5,),
            colors=(color,),
            linewidths=2.0,
            zorder=6,
        )

    axis.contour(
        theta_grid,
        radius_grid,
        field_of_regard.astype(float),
        levels=(0.5,),
        colors=("#687582",),
        linestyles="--",
        linewidths=1.1,
        zorder=8,
    )
    add_solar_system_labels(axis)
    add_earth_reach_inset(figure, earth_reach)

    axis.set_theta_zero_location("E")
    axis.set_theta_direction(1)
    axis.set_rscale("log")
    axis.set_rlim(0.08, 10.0)
    axis.set_rticks((0.1, 0.3, 1.0, 3.0, 10.0))
    axis.set_yticklabels(("0.1 au", "0.3 au", "1 au", "3 au", "10 au"))
    axis.set_rlabel_position(68)
    axis.set_thetagrids(
        (0, 60, 120, 180, 240, 300),
        ("Earth and L2 direction", "", "", "", "", ""),
        fontsize=8.4,
    )
    axis.grid(color="#8E99A5", alpha=0.17, lw=0.7)

    figure.suptitle(
        "Heliocentric asteroid detection area",
        fontsize=20,
        weight="bold",
        y=0.966,
    )
    summary_text = "\n".join(
        (
            rf"$p_V={ALBEDO:.2f}$",
            rf"$t_{{\rm exp}}={EXPOSURE_S:.0f}\,\mathrm{{s}}$",
            rf"$V_{{AB,5\sigma}}={limiting_magnitude:.2f}$",
            r"observer at Sun--Earth L2",
        )
    )
    figure.text(
        0.025,
        0.745,
        summary_text,
        ha="left",
        va="top",
        fontsize=10.0,
        linespacing=1.45,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "white",
            "edgecolor": "#BAC2CA",
            "alpha": 0.98,
        },
    )

    detection_handles = [
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
    detection_handles.append(
        Line2D(
            [0],
            [0],
            color="#687582",
            lw=1.3,
            ls="--",
            label="60 deg Sun-avoidance boundary",
        )
    )
    detection_handles.append(
        Patch(
            facecolor="#81715E",
            edgecolor="#6F6253",
            alpha=0.18,
            label="main asteroid belt",
        )
    )
    figure.legend(
        handles=detection_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.52, 0.068),
        fontsize=9.7,
    )
    figure.supxlabel(
        "Radial coordinate is heliocentric distance [au, log scale]",
        x=0.54,
        y=0.151,
        fontsize=12,
        color="#151B23",
    )
    figure.text(
        0.5,
        0.014,
        (
            "Contours use the standard H-diameter-albedo relation, the H,G "
            "phase law with G=0.15, and the proposal optical ETC. "
            "They exclude trailing, confusion, and orbit-linking losses."
        ),
        ha="center",
        color="#5C6875",
        fontsize=9.0,
    )
    figure.subplots_adjust(
        left=0.285,
        right=0.975,
        bottom=0.225,
        top=0.90,
    )
    OUTPUT.parent.mkdir(exist_ok=True)
    figure.savefig(OUTPUT, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT.resolve())
    print(
        f"Figure C.8 reference limit is {limiting_magnitude:.3f} AB mag. "
        f"The current local ETC returns {current_etc_limiting_magnitude:.3f} "
        "AB mag for the same exposure."
    )
    print(
        "Maximum Earth-centered reach: "
        + ", ".join(
            f"{diameter_m:g} m = {earth_reach[diameter_m]:.3f} au"
            for diameter_m in DIAMETERS_M
        )
    )


if __name__ == "__main__":
    main()
