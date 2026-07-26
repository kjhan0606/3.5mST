#!/usr/bin/env python3
"""Five-year 3.5ST MIR moving-object survey emulator."""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

import neo_population_model as neo


ROOT = Path(__file__).resolve().parent
OUTPUT_CSV = ROOT / "neo_survey_yield.csv"
OUTPUT_FIGURE = ROOT / "appendixC_assets" / "neo_survey_yield.png"
MISSION_DAYS = 5.0 * 365.25
REVISIT_DAYS = 13.0
VISIT_ALLOCATION_S = 180.0
NET_EXPOSURE_S = 145.0
DOWNLINK_HOURS_DAY = 2.25
FOV_DEG2 = 0.25
TRACKLET_VISITS = 4
LINK_EFFICIENCY = 0.95


def load_population():
    population = neo.load_population()
    neo.assign_angular_elements(population)
    return population


def angular_rate_deg_day(population, days, geometry):
    later = neo.observing_geometry(population, days + 0.25)
    separation = np.degrees(np.arccos(np.clip(np.sum(
        geometry["los_unit"] * later["los_unit"], axis=1), -1.0, 1.0)))
    return separation / 0.25


def detection_mask(population, days):
    geometry = neo.observing_geometry(population, days)
    nc1_ujy = neo.mir_flux_jy(population, geometry, "NC1") * 1e6
    nc2_ujy = neo.mir_flux_jy(population, geometry, "NC2") * 1e6
    nc1_limit = neo.mir_limit_ujy(
        "NC1", geometry["elongation_deg"], geometry["latitude_deg"],
        NET_EXPOSURE_S)
    nc2_limit = neo.mir_limit_ujy(
        "NC2", geometry["elongation_deg"], geometry["latitude_deg"],
        NET_EXPOSURE_S)
    rate = angular_rate_deg_day(population, days, geometry)
    common = (
        (geometry["elongation_deg"] >= 45.0)
        & (geometry["elongation_deg"] <= 120.0)
        & (rate >= 0.008)
        & (rate <= 8.0)
        & ((nc1_ujy >= nc1_limit) | (nc2_ujy >= nc2_limit))
    )
    return common, np.abs(geometry["latitude_deg"])


def pair_counts(population):
    times = np.arange(0.0, MISSION_DAYS + REVISIT_DAYS, REVISIT_DAYS)
    counts = {20: np.zeros(len(population["H"]), dtype=np.uint8),
              40: np.zeros(len(population["H"]), dtype=np.uint8)}
    previous_common = None
    previous_latitude = None
    for index, days in enumerate(times):
        common, latitude = detection_mask(population, days)
        if previous_common is not None:
            for latitude_limit in counts:
                pair = (
                    previous_common & common
                    & (previous_latitude <= latitude_limit)
                    & (latitude <= latitude_limit)
                )
                counts[latitude_limit] += pair
        previous_common = common
        previous_latitude = latitude
        if index % 20 == 0 or index == len(times) - 1:
            print(
                f"epoch {index + 1:3d}/{len(times)}  day={days:7.1f}  "
                f"eligible={np.count_nonzero(common):7,d}",
                flush=True,
            )
    return counts


def search_region_area_deg2(latitude_limit_deg):
    longitude_width_rad = np.radians(2.0 * (120.0 - 45.0))
    area_sr = (
        longitude_width_rad
        * 2.0 * np.sin(np.radians(latitude_limit_deg)))
    return area_sr * (180.0 / np.pi)**2


def coverage_fraction(fov_deg2, visits, time_fraction, latitude_limit_deg):
    available_s_day = (24.0 - DOWNLINK_HOURS_DAY) * 3600.0 * time_fraction
    unique_area_day = (
        fov_deg2 * available_s_day / (visits * VISIT_ALLOCATION_S))
    window_area = unique_area_day * REVISIT_DAYS
    return min(1.0, window_area / search_region_area_deg2(latitude_limit_deg))


def known_probability(diameter_m):
    probability = np.full_like(diameter_m, 0.03, dtype=float)
    probability[(diameter_m >= 50.0) & (diameter_m < 140.0)] = 0.10
    probability[(diameter_m >= 140.0) & (diameter_m < 1000.0)] = 0.38
    probability[diameter_m >= 1000.0] = 0.95
    return probability


def scenario_yield(population, counts, fov_deg2, visits, time_fraction,
                   latitude_limit_deg):
    coverage = coverage_fraction(
        fov_deg2, visits, time_fraction, latitude_limit_deg)
    if visits < 4:
        pair_probability = 0.0
    else:
        tracklet_efficiency = 0.99 if visits == 4 else 0.995
        pair_probability = (
            coverage**2 * LINK_EFFICIENCY * tracklet_efficiency**2)
    probability = 1.0 - (1.0 - pair_probability)**counts[latitude_limit_deg]
    diameter_m = population["diameter_km"] * 1000.0
    known = known_probability(diameter_m)
    bins = ((30.0, 50.0), (50.0, 140.0),
            (140.0, 1000.0), (1000.0, 3001.0))
    result = {
        "fov_deg2": fov_deg2,
        "visits": visits,
        "time_fraction": time_fraction,
        "latitude_limit_deg": latitude_limit_deg,
        "coverage_per_13d": coverage,
    }
    for lower, upper in bins:
        selected = (diameter_m >= lower) & (diameter_m < upper)
        label = f"d{int(lower)}_{int(upper)}"
        result[f"{label}_cataloged"] = float(np.sum(probability[selected]))
        result[f"{label}_new"] = float(
            np.sum(probability[selected] * (1.0 - known[selected])))
    selected_140 = diameter_m >= 140.0
    result["d140plus_cataloged"] = float(np.sum(probability[selected_140]))
    result["d140plus_new"] = float(np.sum(
        probability[selected_140] * (1.0 - known[selected_140])))
    result["d140plus_completeness"] = (
        result["d140plus_cataloged"] / np.count_nonzero(selected_140))
    return result


def build_scenarios(population, counts):
    scenarios = []
    for latitude_limit in (20, 40):
        for time_fraction in (0.25, 1.0):
            scenarios.append(scenario_yield(
                population, counts, 0.25, 4, time_fraction, latitude_limit))
    for fov in (1.0, 4.0, 11.9):
        scenarios.append(scenario_yield(
            population, counts, fov, 4, 1.0, 40))
    for visits in (2, 4, 6):
        scenarios.append(scenario_yield(
            population, counts, 0.25, visits, 1.0, 40))
    return scenarios


def write_csv(scenarios):
    with OUTPUT_CSV.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=scenarios[0].keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(scenarios)


def make_figure(scenarios):
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

    fov_scenarios = [
        row for row in scenarios
        if row["visits"] == 4
        and row["time_fraction"] == 1.0
        and row["latitude_limit_deg"] == 40
    ]
    fov_scenarios = sorted(
        {row["fov_deg2"]: row for row in fov_scenarios}.values(),
        key=lambda row: row["fov_deg2"],
    )
    fov = [row["fov_deg2"] for row in fov_scenarios]
    completeness = [
        100.0 * row["d140plus_completeness"] for row in fov_scenarios]
    new = [row["d140plus_new"] for row in fov_scenarios]
    axes[0].plot(fov, completeness, "o-", color="#B86B00", lw=2.2)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Instantaneous field [deg2]")
    axes[0].set_ylabel("Five-year D >= 140 m catalog fraction [%]")
    axes[0].set_title("Discovery completeness is field limited")
    axes[0].grid(alpha=0.18)
    axes[0].set_ylim(0.0, max(completeness) * 1.18)
    for index, (x, y, value) in enumerate(
        zip(fov, completeness, new, strict=True)
    ):
        if index == 0:
            alignment = "left"
        elif index == len(fov) - 1:
            alignment = "right"
        else:
            alignment = "center"
        axes[0].annotate(
            f"{value:,.0f} new", (x, y), xytext=(0, 8),
            textcoords="offset points", ha=alignment, color="#5C6875")

    visit_scenarios = [
        row for row in scenarios
        if row["fov_deg2"] == 0.25
        and row["time_fraction"] == 1.0
        and row["latitude_limit_deg"] == 40
    ]
    visit_scenarios = sorted(visit_scenarios, key=lambda row: row["visits"])
    visits = [row["visits"] for row in visit_scenarios]
    cataloged = [row["d140plus_cataloged"] for row in visit_scenarios]
    colors = ["#A7354D", "#007C77", "#B55220"]
    axes[1].bar(visits, cataloged, color=colors, width=0.7)
    axes[1].set_xticks(visits)
    axes[1].set_xlabel("Visits per tracklet")
    axes[1].set_ylabel("Five-year D >= 140 m cataloged")
    axes[1].set_title("Four visits are the minimum accepted cadence")
    axes[1].grid(axis="y", alpha=0.18)
    axes[1].set_ylim(0.0, max(cataloged) * 1.24)
    axes[1].text(
        0.03, 0.91,
        "Two visits fail the >=4-detection tracklet criterion.\n"
        "Six visits reduce survey area with little efficiency gain.",
        transform=axes[1].transAxes, va="top", color="#5C6875")

    figure.patch.set_facecolor("white")
    for axis in axes:
        axis.set_facecolor("white")
        for spine in axis.spines.values():
            spine.set_color("#5C6875")
    figure.savefig(OUTPUT_FIGURE, dpi=220, facecolor="white")
    plt.close(figure)


def main():
    population = load_population()
    counts = pair_counts(population)
    scenarios = build_scenarios(population, counts)
    write_csv(scenarios)
    make_figure(scenarios)
    print("\nReference scenarios")
    for row in scenarios:
        if (
            row["fov_deg2"] == 0.25
            and row["visits"] == 4
            and row["latitude_limit_deg"] == 40
        ):
            print(
                f"time={row['time_fraction']:.0%}, "
                f"coverage/13d={row['coverage_per_13d']:.3%}, "
                f"D>=140 cataloged={row['d140plus_cataloged']:.0f}, "
                f"new={row['d140plus_new']:.0f}, "
                f"fraction={row['d140plus_completeness']:.2%}")
    print(OUTPUT_CSV)
    print(OUTPUT_FIGURE)


if __name__ == "__main__":
    main()
