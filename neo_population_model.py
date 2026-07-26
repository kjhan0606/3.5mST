#!/usr/bin/env python3
"""NEOMOD3 reference population and optical/MIR flux calculations."""

from pathlib import Path
import subprocess
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import slitless_etc as etc


ROOT = Path(__file__).resolve().parent
NEOMOD_DIR = ROOT / "references" / "NEOMOD_Simulator"
NEOMOD_EXE = NEOMOD_DIR / "neomod3_simulator"
NEOMOD_SAMPLE = NEOMOD_DIR / "neomod3_D003_3km_seed50202002.dat"
OUTPUT_NPZ = ROOT / "neo_reference_population.npz"
OUTPUT_FIGURE = ROOT / "appendixC_assets" / "neo_reference_population.png"

AU_M = 149597870700.0
R_SUN_M = 6.957e8
SIGMA_SB = 5.670374419e-8
SOLAR_CONSTANT = 1361.0
H_SI = 6.62607015e-34
C_SI = 299792458.0
K_B_SI = 1.380649e-23
GAUSSIAN_K = 0.01720209895


def ensure_neomod_sample():
    if not NEOMOD_EXE.exists():
        subprocess.run(
            ["gfortran", "-O3", "-o", NEOMOD_EXE.name,
             "neomod3_simulator.f"],
            cwd=NEOMOD_DIR,
            check=True,
        )
    if NEOMOD_SAMPLE.exists():
        return
    parameters = "input_neomod3.dat\n-50202002\n-1\n0.03 3.0\n"
    with NEOMOD_SAMPLE.open("w") as output:
        subprocess.run(
            [str(NEOMOD_EXE)],
            cwd=NEOMOD_DIR,
            input=parameters,
            text=True,
            stdout=output,
            check=True,
        )


def load_population():
    ensure_neomod_sample()
    values = np.loadtxt(NEOMOD_SAMPLE)
    return {
        "H": values[:, 0],
        "a_au": values[:, 1],
        "e": values[:, 2],
        "inc_rad": np.radians(values[:, 3]),
        "diameter_km": values[:, 4],
        "p_v": values[:, 5],
    }


def assign_angular_elements(population, seed=35002026):
    rng = np.random.default_rng(seed)
    count = len(population["H"])
    population["node_rad"] = rng.uniform(0.0, 2.0 * np.pi, count)
    population["argperi_rad"] = rng.uniform(0.0, 2.0 * np.pi, count)
    population["mean_anomaly_rad"] = rng.uniform(0.0, 2.0 * np.pi, count)


def kepler_positions(population, days):
    a = population["a_au"]
    eccentricity = population["e"]
    mean_motion = GAUSSIAN_K / a**1.5
    mean_anomaly = np.mod(
        population["mean_anomaly_rad"] + mean_motion * days,
        2.0 * np.pi,
    )
    eccentric_anomaly = mean_anomaly.copy()
    for _ in range(8):
        eccentric_anomaly -= (
            eccentric_anomaly
            - eccentricity * np.sin(eccentric_anomaly)
            - mean_anomaly
        ) / (1.0 - eccentricity * np.cos(eccentric_anomaly))

    x_orbit = a * (np.cos(eccentric_anomaly) - eccentricity)
    y_orbit = a * np.sqrt(1.0 - eccentricity**2) * np.sin(
        eccentric_anomaly)
    omega = population["argperi_rad"]
    node = population["node_rad"]
    inc = population["inc_rad"]
    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)
    cos_node = np.cos(node)
    sin_node = np.sin(node)
    cos_inc = np.cos(inc)
    sin_inc = np.sin(inc)

    p_x = cos_node * cos_omega - sin_node * sin_omega * cos_inc
    p_y = sin_node * cos_omega + cos_node * sin_omega * cos_inc
    p_z = sin_omega * sin_inc
    q_x = -cos_node * sin_omega - sin_node * cos_omega * cos_inc
    q_y = -sin_node * sin_omega + cos_node * cos_omega * cos_inc
    q_z = cos_omega * sin_inc
    return np.column_stack((
        p_x * x_orbit + q_x * y_orbit,
        p_y * x_orbit + q_y * y_orbit,
        p_z * x_orbit + q_z * y_orbit,
    ))


def observer_position(days):
    longitude = GAUSSIAN_K * days
    return 1.01 * np.array([
        np.cos(longitude), np.sin(longitude), 0.0])


def observing_geometry(population, days):
    asteroid = kepler_positions(population, days)
    observer = observer_position(days)
    line_of_sight = asteroid - observer
    r_au = np.linalg.norm(asteroid, axis=1)
    delta_au = np.linalg.norm(line_of_sight, axis=1)
    los_unit = line_of_sight / delta_au[:, None]
    sun_from_observer = -observer / np.linalg.norm(observer)
    elongation = np.degrees(np.arccos(np.clip(
        los_unit @ sun_from_observer, -1.0, 1.0)))
    latitude = np.degrees(np.arcsin(np.clip(los_unit[:, 2], -1.0, 1.0)))
    sun_from_asteroid = -asteroid / r_au[:, None]
    observer_from_asteroid = -line_of_sight / delta_au[:, None]
    phase = np.degrees(np.arccos(np.clip(np.sum(
        sun_from_asteroid * observer_from_asteroid, axis=1), -1.0, 1.0)))
    return {
        "r_au": r_au,
        "delta_au": delta_au,
        "elongation_deg": elongation,
        "latitude_deg": latitude,
        "phase_deg": phase,
        "los_unit": los_unit,
    }


def hg_phase_function(phase_deg, slope_g=0.15):
    tangent = np.tan(0.5 * np.radians(np.clip(phase_deg, 0.0, 179.0)))
    phi_1 = np.exp(-3.33 * tangent**0.63)
    phi_2 = np.exp(-1.87 * tangent**1.22)
    return (1.0 - slope_g) * phi_1 + slope_g * phi_2


def optical_v_magnitude(population, geometry):
    phase = hg_phase_function(geometry["phase_deg"])
    return (
        population["H"]
        + 5.0 * np.log10(geometry["r_au"] * geometry["delta_au"])
        - 2.5 * np.log10(np.maximum(phase, 1e-12))
    )


def planck_nu(wavelength_m, temperature_k):
    wavelength_m = np.asarray(wavelength_m)
    temperature_k = np.asarray(temperature_k)
    exponent = H_SI * C_SI / (
        wavelength_m * K_B_SI * np.maximum(temperature_k, 1e-6))
    return (
        2.0 * H_SI * C_SI / wavelength_m**3
        / np.expm1(np.minimum(exponent, 700.0))
    )


def band_averaged_bnu(temperature_grid, band_um):
    wavelengths = np.linspace(band_um[0], band_um[1], 180) * 1e-6
    weights = 1.0 / wavelengths
    bnu = planck_nu(
        wavelengths[None, :], temperature_grid[:, None])
    return np.trapezoid(
        bnu * weights[None, :], wavelengths, axis=1
    ) / np.trapezoid(weights, wavelengths)


@lru_cache(maxsize=4)
def thermal_disk_grid(band_um):
    temperature_grid = np.linspace(1.0, 800.0, 1600)
    bnu_grid = band_averaged_bnu(temperature_grid, band_um)
    subsolar_grid = np.linspace(150.0, 700.0, 111)
    phase_grid = np.linspace(0.0, 150.0, 76)
    mu_nodes, mu_weights = np.polynomial.legendre.leggauss(26)
    mu = 0.5 * (mu_nodes + 1.0)
    mu_weights = 0.5 * mu_weights
    azimuth = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    dphi = 2.0 * np.pi / len(azimuth)
    normal_x = np.sqrt(1.0 - mu[:, None]**2) * np.cos(
        azimuth[None, :])
    normal_z = np.broadcast_to(mu[:, None], normal_x.shape)
    projected_weight = mu_weights[:, None] * normal_z * dphi
    disk = np.empty((len(subsolar_grid), len(phase_grid)))

    for phase_index, phase_deg in enumerate(phase_grid):
        phase_rad = np.radians(phase_deg)
        solar_mu = (
            normal_x * np.sin(phase_rad)
            + normal_z * np.cos(phase_rad)
        )
        illuminated = np.maximum(solar_mu, 0.0)
        temperature_factor = illuminated**0.25
        for temperature_index, subsolar in enumerate(subsolar_grid):
            surface_temperature = subsolar * temperature_factor
            surface_bnu = np.interp(
                surface_temperature, temperature_grid, bnu_grid)
            surface_bnu[illuminated <= 0.0] = 0.0
            disk[temperature_index, phase_index] = np.sum(
                surface_bnu * projected_weight)
    return subsolar_grid, phase_grid, disk


@lru_cache(maxsize=4)
def solar_band_flux_jy(band_um):
    wavelengths = np.linspace(band_um[0], band_um[1], 180) * 1e-6
    weights = 1.0 / wavelengths
    solar_bnu = planck_nu(wavelengths, 5772.0)
    mean_bnu = np.trapezoid(
        solar_bnu * weights, wavelengths
    ) / np.trapezoid(weights, wavelengths)
    flux = np.pi * mean_bnu * (R_SUN_M / AU_M)**2
    return flux / 1e-26


def mir_flux_jy(population, geometry, channel, beaming=1.4,
                emissivity=0.9, pir_over_pv=1.6):
    band_um = etc.MIR_CHANNELS[channel]["band_um"]
    subsolar_grid, phase_grid, disk_grid = thermal_disk_grid(band_um)
    phase_integral = RegularGridInterpolator(
        (subsolar_grid, phase_grid), disk_grid,
        bounds_error=False, fill_value=None)
    slope_g = 0.15
    phase_integral_q = 0.290 + 0.684 * slope_g
    bond_albedo = np.clip(
        phase_integral_q * population["p_v"], 0.0, 0.9)
    subsolar_temperature = (
        (1.0 - bond_albedo) * SOLAR_CONSTANT
        / (beaming * emissivity * SIGMA_SB * geometry["r_au"]**2)
    )**0.25
    points = np.column_stack((
        np.clip(subsolar_temperature, subsolar_grid[0], subsolar_grid[-1]),
        np.clip(geometry["phase_deg"], phase_grid[0], phase_grid[-1]),
    ))
    disk_bnu = phase_integral(points)
    radius_m = population["diameter_km"] * 500.0
    distance_m = geometry["delta_au"] * AU_M
    thermal = emissivity * radius_m**2 / distance_m**2 * disk_bnu / 1e-26

    if channel == "NC1":
        p_ir = np.clip(pir_over_pv * population["p_v"], 0.015, 0.70)
        phase = hg_phase_function(geometry["phase_deg"])
        reflected = (
            solar_band_flux_jy(band_um)
            * p_ir * phase
            * radius_m**2
            / ((geometry["r_au"] * AU_M)**2
               * geometry["delta_au"]**2)
        )
        thermal += reflected
    return thermal


def mir_limit_ujy(channel, elongation_deg, latitude_deg, exposure_s=145.0):
    levels = etc.MIR_CHANNELS[channel]["nuinu_nw_m2_sr"]
    proximity = np.clip((120.0 - elongation_deg) / 75.0, 0.0, 1.0)
    latitude_factor = np.clip(
        1.0 - np.abs(latitude_deg) / 40.0, 0.0, 1.0)
    weight = proximity * latitude_factor
    background = levels["low"] * (
        levels["high"] / levels["low"])**weight
    background_grid = np.array([
        levels["low"], levels["nominal"], levels["high"]])
    limit_grid = np.array([
        etc.imaging_flux_limit_jy(
            etc.mir_imaging_cfg(channel, level),
            tuple(value * 1e4 for value in etc.MIR_CHANNELS[channel]["band_um"]),
            exposure_s,
        ) * 1e6
        for level in ("low", "nominal", "high")
    ])
    return np.exp(np.interp(
        np.log(background), np.log(background_grid), np.log(limit_grid)))


def evaluate_population(population):
    geometry = observing_geometry(population, 0.0)
    v_mag = optical_v_magnitude(population, geometry)
    nc1_ujy = mir_flux_jy(population, geometry, "NC1") * 1e6
    nc2_ujy = mir_flux_jy(population, geometry, "NC2") * 1e6
    nc1_limit = mir_limit_ujy(
        "NC1", geometry["elongation_deg"], geometry["latitude_deg"])
    nc2_limit = mir_limit_ujy(
        "NC2", geometry["elongation_deg"], geometry["latitude_deg"])
    accessible = (
        (geometry["elongation_deg"] >= 45.0)
        & (geometry["elongation_deg"] <= 120.0)
        & (np.abs(geometry["latitude_deg"]) <= 40.0)
    )
    detected = accessible & (
        (nc1_ujy >= nc1_limit) | (nc2_ujy >= nc2_limit))
    return geometry, v_mag, nc1_ujy, nc2_ujy, detected


def make_figure(population, nc2_ujy, detected):
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "text.color": "#151B23",
        "axes.labelcolor": "#151B23",
        "axes.edgecolor": "#5C6875",
        "xtick.color": "#151B23",
        "ytick.color": "#151B23",
    })
    figure, axes = plt.subplots(
        1, 2, figsize=(12.8, 5.0), constrained_layout=True)
    diameters = population["diameter_km"] * 1000.0
    bins = np.logspace(np.log10(30.0), np.log10(3000.0), 45)
    axes[0].hist(
        diameters, bins=bins, histtype="step", lw=2.2, color="#007C77",
        label="NEOMOD3 realization")
    axes[0].axvline(140.0, color="#B86B00", lw=2.0, label="140 m")
    axes[0].axvline(1000.0, color="#B55220", lw=2.0, label="1 km")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Diameter [m]")
    axes[0].set_ylabel("Objects per bin")
    axes[0].set_title("Public NEOMOD3 sample")
    axes[0].legend()
    axes[0].grid(alpha=0.18)

    rng = np.random.default_rng(20260726)
    indices = rng.choice(
        len(diameters), size=min(30000, len(diameters)), replace=False)
    colors = np.where(detected[indices], "#B86B00", "#007C77")
    axes[1].scatter(
        diameters[indices], nc2_ujy[indices], s=4, c=colors, alpha=0.35,
        linewidths=0)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Diameter [m]")
    axes[1].set_ylabel("Instantaneous MIR-2 flux [uJy]")
    axes[1].set_title("Thermal flux over the NEOMOD3 orbit distribution")
    axes[1].grid(alpha=0.18)
    axes[1].text(
        0.03, 0.04,
        "Gold objects satisfy the 45-120 deg field of regard and ETC limit",
        transform=axes[1].transAxes, color="#5C6875", fontsize=9)

    figure.patch.set_facecolor("white")
    for axis in axes:
        axis.set_facecolor("white")
        for spine in axis.spines.values():
            spine.set_color("#5C6875")
    figure.savefig(OUTPUT_FIGURE, dpi=220, facecolor="white")
    plt.close(figure)


def main():
    population = load_population()
    assign_angular_elements(population)
    geometry, v_mag, nc1_ujy, nc2_ujy, detected = evaluate_population(
        population)
    np.savez_compressed(
        OUTPUT_NPZ,
        **population,
        v_mag=v_mag,
        nc1_ujy=nc1_ujy,
        nc2_ujy=nc2_ujy,
        detected_snapshot=detected,
        elongation_deg=geometry["elongation_deg"],
        latitude_deg=geometry["latitude_deg"],
        phase_deg=geometry["phase_deg"],
        delta_au=geometry["delta_au"],
        r_au=geometry["r_au"],
    )
    make_figure(population, nc2_ujy, detected)

    diameter_m = population["diameter_km"] * 1000.0
    print(f"NEOMOD3 objects D=30-3000 m: {len(diameter_m):,}")
    print(f"D >= 140 m: {np.count_nonzero(diameter_m >= 140):,}")
    print(f"D >= 1 km: {np.count_nonzero(diameter_m >= 1000):,}")
    print(
        "Instantaneously detectable in either MIR channel: "
        f"{np.count_nonzero(detected):,}")
    for lower, upper in ((30, 50), (50, 140), (140, 1000), (1000, 3001)):
        selected = (diameter_m >= lower) & (diameter_m < upper)
        print(
            f"{lower:4.0f}-{upper:4.0f} m: "
            f"{np.count_nonzero(selected):7,d} total, "
            f"{np.count_nonzero(selected & detected):6,d} snapshot detected")
    print(OUTPUT_NPZ)
    print(OUTPUT_FIGURE)


if __name__ == "__main__":
    main()
