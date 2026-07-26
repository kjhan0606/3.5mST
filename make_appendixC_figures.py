from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.patches import Patch

OUT = Path("appendixC_assets")
OUT.mkdir(exist_ok=True)

BG = "#ffffff"
PANEL = "#f6f7f2"
TEXT = "#151b23"
MUTED = "#5c6875"
TEAL = "#007c77"
VISIBLE_SKY = "#70d8c7"
GOLD = "#b86b00"
ORANGE = "#b55220"
RED = "#a7354d"
BLUE = "#245aa6"

mpl.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titleweight": "bold",
        "axes.titlesize": 15,
        "legend.frameon": False,
        "grid.color": MUTED,
        "grid.alpha": 0.22,
    }
)


def save(figure: plt.Figure, name: str) -> None:
    figure.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor=BG)
    plt.close(figure)


def planck_shape(wavelength_um: np.ndarray, temperature: float) -> np.ndarray:
    c2 = 1.438776877e4
    shape = wavelength_um ** -5 / np.expm1(c2 / (wavelength_um * temperature))
    return shape / np.nanmax(shape)


def thermal_windows() -> None:
    wavelength = np.geomspace(0.2, 25.0, 1200)
    figure, axis = plt.subplots(figsize=(12.5, 6.8))
    for temperature, color in [(250, BLUE), (300, TEAL), (400, ORANGE)]:
        axis.plot(
            wavelength,
            planck_shape(wavelength, temperature),
            lw=2.4,
            color=color,
            label=f"{temperature} K",
        )
    axis.axvspan(0.2, 1.5, color=BLUE, alpha=0.16)
    axis.axvspan(4.0, 5.2, color=GOLD, alpha=0.24)
    axis.axvspan(6.0, 10.0, color=GOLD, alpha=0.24)
    axis.text(0.38, 0.78, "3.5ST baseline\n0.2-1.5 um", color=BLUE, weight="bold")
    axis.text(4.05, 0.08, "4.0-5.2", color=GOLD, weight="bold")
    axis.text(6.2, 0.88, "6.0-10.0 um", color=GOLD, weight="bold")
    axis.text(
        0.25,
        0.03,
        "Normalized spectral shape. Absolute detectability also depends on diameter and distance.",
        color=MUTED,
        fontsize=10,
    )
    axis.set_xscale("log")
    axis.set_xlim(0.2, 25)
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("Wavelength [um]")
    axis.set_ylabel("Normalized thermal spectral radiance")
    axis.set_title("NEO thermal emission peaks in the mid-infrared")
    axis.grid(True, which="both")
    axis.legend(title="Blackbody temperature", labelcolor=TEXT)
    save(figure, "neo_thermal_windows.png")


def diameter_proxy() -> None:
    apparent_mag = np.linspace(18, 31, 521)
    albedo = np.geomspace(0.01, 0.60, 401)
    magnitude_grid, albedo_grid = np.meshgrid(apparent_mag, albedo)
    heliocentric_distance_au = 1.0
    phase_function = 1.0
    observer_distances_au = [0.01, 0.10, 1.00]
    diameter_colors = mpl.colors.LinearSegmentedColormap.from_list(
        "diameter_blue_yellow_red",
        [
            (0.00, "#082f73"),
            (0.28, "#1769aa"),
            (0.48, "#22a7c7"),
            (0.64, "#f4e43a"),
            (0.79, "#f49a24"),
            (0.91, "#d73027"),
            (1.00, "#7f0000"),
        ],
    )

    figure, axes = plt.subplots(
        1,
        len(observer_distances_au),
        figsize=(15.4, 5.8),
        sharex=True,
        sharey=True,
    )
    diameter_norm = mpl.colors.LogNorm(vmin=0.03, vmax=1.0e4)
    contour_levels = [0.3, 1, 10, 30, 140, 1000, 3000]
    contour_format = {
        level: (
            f"{level:g} m" if level < 1000 else f"{level / 1000:g} km"
        )
        for level in contour_levels
    }
    label_albedo = 0.015
    diameter_map = None

    for axis, observer_distance_au in zip(axes, observer_distances_au):
        diameter_m = (
            1.329e6
            * heliocentric_distance_au
            * observer_distance_au
            * 10 ** (-0.2 * magnitude_grid)
            / np.sqrt(albedo_grid * phase_function)
        )
        diameter_map = axis.pcolormesh(
            magnitude_grid,
            albedo_grid,
            diameter_m,
            cmap=diameter_colors,
            norm=diameter_norm,
            shading="auto",
        )
        visible_levels = [
            level
            for level in contour_levels
            if diameter_m.min() < level < diameter_m.max()
        ]
        contours = axis.contour(
            magnitude_grid,
            albedo_grid,
            diameter_m,
            levels=visible_levels,
            colors="white",
            linewidths=1.0,
            alpha=0.92,
        )
        label_levels = []
        label_positions = []
        for level in visible_levels:
            label_magnitude = -5.0 * np.log10(
                level
                * np.sqrt(label_albedo * phase_function)
                / (
                    1.329e6
                    * heliocentric_distance_au
                    * observer_distance_au
                )
            )
            if 18.35 < label_magnitude < 30.65:
                label_levels.append(level)
                label_positions.append((label_magnitude, label_albedo))
        if label_levels:
            contour_labels = axis.clabel(
                contours,
                fmt=contour_format,
                inline=False,
                fontsize=9.2,
                colors="white",
                manual=label_positions,
            )
            for label in contour_labels:
                label.set_rotation(0)
                label.set_weight("bold")
                label.set_path_effects(
                    [patheffects.withStroke(linewidth=3.0, foreground="#17202a")]
                )

        axis.set_yscale("log")
        axis.set_xlim(18, 31)
        axis.set_ylim(0.01, 0.60)
        axis.set_yticks([0.01, 0.03, 0.10, 0.30, 0.60])
        axis.set_yticklabels(["0.01", "0.03", "0.10", "0.30", "0.60"])
        axis.set_title(
            rf"$\Delta={observer_distance_au:g}$ au",
            fontsize=13,
        )
        axis.grid(False)

    if diameter_map is None:
        raise RuntimeError("No diameter map was generated")

    axes[0].set_ylabel(r"Visible geometric albedo $p_V$")
    figure.supxlabel(
        r"Observed V-band magnitude $m_V$ (larger is fainter)",
        y=0.035,
    )
    figure.suptitle(
        "Detectable NEO size increases with telescope-target distance",
        y=0.98,
        fontsize=17,
        weight="bold",
    )
    figure.text(
        0.075,
        0.885,
        r"Reference geometry: $r=1$ au and $\Phi(\alpha)=1$",
        color=MUTED,
        fontsize=10.5,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.885,
        bottom=0.17,
        top=0.82,
        wspace=0.08,
    )
    colorbar_axis = figure.add_axes([0.905, 0.17, 0.016, 0.65])
    colorbar = figure.colorbar(
        diameter_map,
        cax=colorbar_axis,
    )
    colorbar.set_label("Diameter D [m]")
    colorbar.set_ticks([0.1, 1, 10, 100, 1000, 10000])
    colorbar.set_ticklabels(["0.1", "1", "10", "100", "1,000", "10,000"])
    save(figure, "neo_h_diameter.png")


def astrometric_limit() -> None:
    snr = np.geomspace(5, 300, 500)
    figure, axis = plt.subplots(figsize=(11.8, 6.8))
    for fwhm, color in [(0.06, TEAL), (0.11, GOLD), (0.20, ORANGE)]:
        sigma_mas = 1000.0 * fwhm / (2.355 * snr)
        axis.plot(snr, sigma_mas, color=color, lw=2.4, label=f"FWHM = {fwhm:.2f} arcsec")
    axis.axhspan(1, 5, color=RED, alpha=0.14)
    axis.text(7, 1.35, "1-5 mas systematic floor must be demonstrated", color=RED, weight="bold")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(5, 300)
    axis.set_ylim(0.05, 30)
    axis.set_xlabel("Point-source signal-to-noise ratio")
    axis.set_ylabel("Photon-limited centroid error [mas]")
    axis.set_title("Centroid precision is not the same as absolute astrometric accuracy")
    axis.grid(True, which="both")
    axis.legend(labelcolor=TEXT)
    save(figure, "neo_astrometry.png")


def l2_observability() -> None:
    figure = plt.figure(figsize=(14.0, 9.0))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.02, 1.0), hspace=0.28, wspace=0.22)

    geometry = figure.add_subplot(grid[0, 0])
    geometry.set_aspect("equal")
    geometry.set_xlim(-1.30, 1.35)
    geometry.set_ylim(-1.2, 1.2)
    geometry.axis("off")
    l2 = np.array([0.0, 0.0])
    sun_direction = np.array([-1.0, 0.0])
    geometry.scatter([-1.0], [0], s=650, color=GOLD, edgecolor=ORANGE, linewidth=2)
    geometry.text(-1.0, -0.20, "Sun", ha="center", weight="bold", color=GOLD)
    geometry.scatter([-0.26], [0], s=120, color=BLUE)
    geometry.text(-0.26, -0.16, "Earth", ha="center", color=BLUE)
    geometry.scatter([0], [0], s=80, color=TEAL)
    geometry.text(0, 0.14, "Sun-Earth L2", ha="center", color=TEAL, weight="bold")
    angles = np.linspace(-np.deg2rad(60), np.deg2rad(60), 200)
    cone_x = -1.12 * np.cos(angles)
    cone_y = 1.12 * np.sin(angles)
    geometry.fill(np.r_[0, cone_x, 0], np.r_[0, cone_y, 0], color=RED, alpha=0.17)
    geometry.plot(cone_x, cone_y, color=RED, lw=2)
    geometry.annotate("", xy=(1.08, 0), xytext=(0.06, 0), arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 2})
    geometry.text(0.62, 0.09, "anti-solar", ha="center", color=TEAL)
    geometry.text(-0.66, 0.79, "baseline keep-out\nsolar elongation < 60 deg", ha="center", color=RED, weight="bold")
    geometry.text(-0.65, -0.92, "Earth and Moon remain near the same forbidden direction", ha="center", color=MUTED, fontsize=10)
    geometry.set_title("A. L2 viewing geometry", loc="left")

    latitude = figure.add_subplot(grid[0, 1])
    beta = np.linspace(-90, 90, 721)
    theta = np.deg2rad(60)
    visible = np.ones_like(beta)
    mask = np.abs(beta) < 60
    visible[mask] = 1 - np.arccos(np.cos(theta) / np.cos(np.deg2rad(beta[mask]))) / np.pi
    days = 365.25 * visible
    latitude.plot(beta, days, color=TEAL, lw=3)
    latitude.fill_between(beta, 0, days, color=TEAL, alpha=0.12)
    latitude.axhline(365.25, color=GOLD, lw=1)
    latitude.scatter([0], [365.25 * 2 / 3], color=ORANGE, s=65, zorder=5)
    latitude.text(4, 246, "ecliptic plane: about 243.5 days yr$^{-1}$", color=ORANGE)
    latitude.set(xlim=(-90, 90), ylim=(0, 385), xlabel="Ecliptic latitude [deg]", ylabel="Observable days per year")
    latitude.set_title("B. Annual access for a 60 deg Sun-avoidance angle", loc="left")
    latitude.grid(True)

    sky = figure.add_subplot(grid[1, :], projection="mollweide")
    lon = np.linspace(-np.pi, np.pi, 721)
    lat = np.linspace(-np.pi / 2, np.pi / 2, 361)
    longitude, lat_grid = np.meshgrid(lon, lat)
    elongation = np.arccos(np.clip(np.cos(lat_grid) * np.cos(longitude), -1, 1))
    classes = np.zeros_like(elongation)
    classes[(elongation >= np.deg2rad(45)) & (elongation < np.deg2rad(60))] = 1
    classes[elongation >= np.deg2rad(60)] = 2
    cmap = mpl.colors.ListedColormap([RED, ORANGE, VISIBLE_SKY])
    sky.pcolormesh(longitude, lat_grid, classes, cmap=cmap, shading="auto", alpha=0.92)
    sky.scatter([0], [0], color=GOLD, s=90, marker="*", zorder=5)
    sky.annotate(
        "Sun + Earth direction",
        xy=(0, 0),
        xytext=(0.72, 0.43),
        color=GOLD,
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": GOLD, "lw": 1.2},
    )
    sky.grid(color=MUTED, alpha=0.22)
    sky.set_title("C. Instantaneous field of regard in Sun-centered ecliptic coordinates", loc="left", pad=18)
    figure.legend(
        handles=[
            Patch(facecolor=RED, label="forbidden: elongation < 45 deg"),
            Patch(facecolor=ORANGE, label="45-60 deg: dedicated forward baffle and thermal design"),
            Patch(facecolor=VISIBLE_SKY, label="baseline field: elongation > 60 deg"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=3,
        fontsize=10,
        labelcolor=TEXT,
    )
    figure.subplots_adjust(bottom=0.08)
    save(figure, "l2_observability.png")


def force_hierarchy() -> None:
    labels = [
        "Major-body Newtonian gravity",
        "16 main-belt perturbers",
        "Orbit covariance / close encounters",
        "Solar 1PN relativity",
        "Yarkovsky A2",
        "Solar radiation pressure",
        "PR + solar-wind drag",
        "Planet J2 during close flyby",
    ]
    categories = ["required", "required", "required", "required", "fit", "size-dependent", "dust", "encounter"]
    colors = [TEAL, TEAL, RED, GOLD, ORANGE, BLUE, BLUE, ORANGE]
    figure, axis = plt.subplots(figsize=(12.5, 7.0))
    y = np.arange(len(labels))[::-1]
    axis.barh(y, np.ones_like(y), color=colors, alpha=0.82, height=0.64)
    for yi, label, category in zip(y, labels, categories):
        axis.text(0.03, yi, label, va="center", ha="left", color=BG, weight="bold")
        axis.text(1.03, yi, category, va="center", color=TEXT)
    axis.set_xlim(0, 1.45)
    axis.set_yticks([])
    axis.set_xticks([])
    axis.set_title("Force terms are selected by object scale and encounter geometry", loc="left")
    for spine in axis.spines.values():
        spine.set_visible(False)
    save(figure, "neo_force_hierarchy.png")


def main() -> None:
    thermal_windows()
    diameter_proxy()
    astrometric_limit()
    l2_observability()
    force_hierarchy()


if __name__ == "__main__":
    main()
