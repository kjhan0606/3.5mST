"""Validate a 150-year CODES integration of 1P/Halley against JPL Horizons."""

from __future__ import annotations

import csv
import json
import time
import urllib.error
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
from astropy.time import Time
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

from neo_orbit_calculator.core import (
    AU_KM,
    DE440Environment,
    ForceModel,
    propagate_custom,
)
from neo_orbit_calculator.jpl import horizons_elements, horizons_vectors

ROOT = Path(__file__).resolve().parent
KERNEL_DIR = ROOT / "neo_orbit_calculator" / "kernels"
FIGURE_PATH = ROOT / "appendixC_assets" / "halley_codes_longterm_validation.png"
CSV_PATH = ROOT / "halley_codes_longterm_validation.csv"
PROVENANCE_PATH = ROOT / "halley_codes_longterm_provenance.json"
CACHE_PATH = (
    ROOT
    / "neo_orbit_calculator"
    / "validation_data"
    / "halley_codes_1850_2000_full1pn.npz"
)

HALLEY_RECORD = "90000030"
START_TDB = "1850-01-02"
STOP_TDB = "2000-01-01"
A1_AU_DAY2 = 4.887055233121e-10
A2_AU_DAY2 = 1.554720290005e-10

# Seed epochs identify the two perihelia inside the DE440s-supported interval.
OFFICIAL_RETURNS = (
    (1910, "90000029", 2418781.6785),
    (1986, "90000030", 2446469.9736161465),
)
DENSE_CADENCE_DAYS = 365.25 / 2.0
DENSE_PERIHELION_WINDOW_DAYS = 2.0 * 365.25
DENSE_PERIHELION_STEP_DAYS = 30.4375


def _heliocentric_distance(
    jd_tdb: np.ndarray,
    states: np.ndarray,
    environment: DE440Environment,
) -> np.ndarray:
    sun = np.asarray(
        [
            environment.state("SUN", environment.jd_to_et(epoch))[:3]
            for epoch in jd_tdb
        ]
    )
    return np.linalg.norm(states[:, :3] - sun, axis=1)


def _osculating_elements(
    state: np.ndarray,
    jd_tdb: float,
    environment: DE440Environment,
) -> tuple[float, float]:
    et = environment.jd_to_et(jd_tdb)
    heliocentric = state - environment.state("SUN", et)
    rotation = np.asarray(
        spice.pxform("J2000", "ECLIPJ2000", et),
        dtype=float,
    )
    ecliptic_state = np.concatenate(
        (
            rotation @ heliocentric[:3],
            rotation @ heliocentric[3:],
        )
    )
    elements = spice.oscelt(
        ecliptic_state,
        et,
        environment.gm["SUN"],
    )
    perihelion_km = float(elements[0])
    eccentricity = float(elements[1])
    return (
        perihelion_km / (1.0 - eccentricity) / AU_KM,
        perihelion_km / AU_KM,
    )


def _period_years(semimajor_axis_au: np.ndarray | float, mu_sun: float):
    semimajor_axis_km = np.asarray(semimajor_axis_au) * AU_KM
    period_seconds = (
        2.0 * np.pi * np.sqrt(semimajor_axis_km**3 / mu_sun)
    )
    result = period_seconds / (365.25 * 86400.0)
    if np.ndim(result) == 0:
        return float(result)
    return result


def _refine_codes_perihelion(
    jd_tdb: np.ndarray,
    states: np.ndarray,
    distance_km: np.ndarray,
    seed_jd: float,
    environment: DE440Environment,
) -> tuple[float, np.ndarray, float]:
    candidate = np.flatnonzero(
        (jd_tdb >= seed_jd - 20.0) & (jd_tdb <= seed_jd + 20.0)
    )
    index = int(candidate[np.argmin(distance_km[candidate])])
    lower = max(0, index - 6)
    upper = min(len(jd_tdb), index + 7)
    splines = [
        CubicSpline(jd_tdb[lower:upper], states[lower:upper, component])
        for component in range(6)
    ]

    def distance(epoch: float) -> float:
        state = np.asarray([spline(epoch) for spline in splines])
        sun = environment.state(
            "SUN",
            environment.jd_to_et(epoch),
        )[:3]
        return float(np.linalg.norm(state[:3] - sun))

    solution = minimize_scalar(
        distance,
        bounds=(jd_tdb[index - 2], jd_tdb[index + 2]),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    state = np.asarray([spline(solution.x) for spline in splines])
    return float(solution.x), state, float(solution.fun)


def _official_perihelion(
    record: str,
    seed_jd: float,
) -> tuple[float, float, float]:
    epochs = np.linspace(seed_jd - 5.0, seed_jd + 5.0, 41)
    for attempt in range(4):
        try:
            vectors, _ = horizons_vectors(
                record,
                epochs,
                center="500@10",
            )
            break
        except urllib.error.HTTPError:
            if attempt == 3:
                raise
            time.sleep(2.0**attempt)
    splines = [
        CubicSpline(vectors[:, 0], vectors[:, component])
        for component in range(1, 7)
    ]

    def distance(epoch: float) -> float:
        return float(
            np.linalg.norm(
                [spline(epoch) for spline in splines[:3]]
            )
        )

    coarse = int(np.argmin(np.linalg.norm(vectors[:, 1:4], axis=1)))
    solution = minimize_scalar(
        distance,
        bounds=(epochs[coarse - 2], epochs[coarse + 2]),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    for attempt in range(4):
        try:
            elements, _ = horizons_elements(
                record,
                [float(solution.x)],
            )
            break
        except urllib.error.HTTPError:
            if attempt == 3:
                raise
            time.sleep(2.0**attempt)
    return (
        float(solution.x),
        float(elements[0, 7]),
        float(elements[0, 2]),
    )


def _dense_comparison_epochs(
    start_jd: float,
    stop_jd: float,
) -> np.ndarray:
    """Sample the full interval and resolve both perihelion neighborhoods."""
    baseline = np.arange(
        start_jd,
        stop_jd + 1.0,
        DENSE_CADENCE_DAYS,
    )
    near_perihelion = [
        np.arange(
            seed_jd - DENSE_PERIHELION_WINDOW_DAYS,
            seed_jd + DENSE_PERIHELION_WINDOW_DAYS,
            DENSE_PERIHELION_STEP_DAYS,
        )
        for _, _, seed_jd in OFFICIAL_RETURNS
    ]
    epochs = np.concatenate((baseline, *near_perihelion, [stop_jd]))
    epochs = np.unique(np.round(epochs, decimals=8))
    return epochs[(epochs >= start_jd) & (epochs <= stop_jd)]


def _dense_official_elements(epochs: np.ndarray) -> np.ndarray:
    """Fetch the JPL #75 osculating elements in bounded request batches."""
    batches = []
    for start in range(0, len(epochs), 25):
        batch = epochs[start : start + 25]
        for attempt in range(5):
            try:
                values, _ = horizons_elements(HALLEY_RECORD, batch)
                batches.append(values)
                time.sleep(0.2)
                break
            except (
                RuntimeError,
                urllib.error.HTTPError,
                urllib.error.URLError,
            ):
                if attempt == 4:
                    raise
                time.sleep(2.0**attempt)
    return np.vstack(batches)


def _codes_elements_at(
    epochs: np.ndarray,
    integration_jd: np.ndarray,
    integration_states: np.ndarray,
    environment: DE440Environment,
) -> np.ndarray:
    """Interpolate the CODES trajectory and derive osculating elements."""
    splines = [
        CubicSpline(
            integration_jd,
            integration_states[:, component],
        )
        for component in range(6)
    ]
    states = np.column_stack([spline(epochs) for spline in splines])
    return np.asarray(
        [
            _osculating_elements(state, epoch, environment)
            for state, epoch in zip(states, epochs, strict=True)
        ]
    )


def _residual_text(values: np.ndarray, unit: str) -> str:
    rms = float(np.sqrt(np.mean(values**2)))
    maximum = float(np.max(np.abs(values)))
    return (
        "CODES - JPL\n"
        f"RMS {rms:,.2f} {unit}\n"
        f"max {maximum:,.2f} {unit}"
    )


def run_validation() -> tuple[
    list[dict[str, float | int | str]],
    np.ndarray,
    np.ndarray,
    str,
    int,
]:
    start_jd = float(Time(START_TDB, scale="tdb").jd)
    stop_jd = float(Time(STOP_TDB, scale="tdb").jd)
    samples = int(np.ceil((stop_jd - start_jd) / 2.0)) + 1
    model = ForceModel(
        relativity_1pn=True,
        full_multibody_1pn=True,
        a1_au_day2=A1_AU_DAY2,
        a2_au_day2=A2_AU_DAY2,
        a3_au_day2=0.0,
        nongrav_law="marsden",
        outgassing_lag_days=0.0,
    )
    if CACHE_PATH.exists():
        cache = np.load(CACHE_PATH)
        result_jd = np.asarray(cache["jd"], dtype=float)
        result_state = np.asarray(cache["state"], dtype=float)
        function_evaluations = int(cache["nfev"])
        kernel = str(cache["kernel"])
    else:
        result = propagate_custom(
            HALLEY_RECORD,
            start_jd,
            stop_jd,
            samples=samples,
            model=model,
            kernel_dir=KERNEL_DIR,
            validate_horizons=False,
            include_large_asteroids=True,
            backend="fortran",
        )
        result_jd = result.jd_tdb
        result_state = result.state_km_kms
        function_evaluations = result.function_evaluations
        kernel = result.kernel
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            CACHE_PATH,
            jd=result_jd,
            state=result_state,
            nfev=function_evaluations,
            kernel=kernel,
        )
    if "full multi-body 1PN" not in kernel:
        kernel += " + full multi-body 1PN"
    environment = DE440Environment(
        KERNEL_DIR,
        include_large_asteroids=True,
    )
    distance_km = _heliocentric_distance(
        result_jd,
        result_state,
        environment,
    )
    rows: list[dict[str, float | int | str]] = []
    for year, record, seed_jd in OFFICIAL_RETURNS:
        codes_jd, codes_state, codes_distance = _refine_codes_perihelion(
            result_jd,
            result_state,
            distance_km,
            seed_jd,
            environment,
        )
        codes_a, codes_q = _osculating_elements(
            codes_state,
            codes_jd,
            environment,
        )
        official_jd, official_a, official_q = _official_perihelion(
            record,
            seed_jd,
        )
        rows.append(
            {
                "return_year": year,
                "jpl_record": record,
                "codes_perihelion_jd_tdb": codes_jd,
                "jpl_perihelion_jd_tdb": official_jd,
                "delta_t_s": (codes_jd - official_jd) * 86400.0,
                "codes_a_au": codes_a,
                "jpl_a_au": official_a,
                "delta_a_km": (codes_a - official_a) * AU_KM,
                "codes_period_years": _period_years(
                    codes_a,
                    environment.gm["SUN"],
                ),
                "jpl_period_years": _period_years(
                    official_a,
                    environment.gm["SUN"],
                ),
                "delta_period_days": (
                    _period_years(codes_a, environment.gm["SUN"])
                    - _period_years(official_a, environment.gm["SUN"])
                )
                * 365.25,
                "codes_q_au": codes_q,
                "jpl_q_au": official_q,
                "delta_q_km": (codes_q - official_q) * AU_KM,
                "codes_distance_at_perihelion_au": (
                    codes_distance / AU_KM
                ),
            }
        )
    return (
        rows,
        result_jd,
        result_state,
        kernel,
        function_evaluations,
    )


def write_outputs(
    rows: list[dict[str, float | int | str]],
    jd_tdb: np.ndarray,
    states: np.ndarray,
    kernel: str,
    function_evaluations: int,
) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    stride = 10
    timeline_jd = jd_tdb[::stride]
    timeline_year = Time(
        timeline_jd,
        format="jd",
        scale="tdb",
    ).decimalyear
    environment = DE440Environment(
        KERNEL_DIR,
        include_large_asteroids=True,
    )
    timeline_elements = np.asarray(
        [
            _osculating_elements(state, epoch, environment)
            for state, epoch in zip(
                states[::stride],
                timeline_jd,
                strict=True,
            )
        ]
    )
    timeline_a = timeline_elements[:, 0]
    timeline_q = timeline_elements[:, 1]
    timeline_period = _period_years(
        timeline_a,
        environment.gm["SUN"],
    )

    comparison_jd = _dense_comparison_epochs(jd_tdb[0], jd_tdb[-1])
    official_elements = _dense_official_elements(comparison_jd)
    codes_elements = _codes_elements_at(
        comparison_jd,
        jd_tdb,
        states,
        environment,
    )
    comparison_year = Time(
        comparison_jd,
        format="jd",
        scale="tdb",
    ).decimalyear
    codes_a = codes_elements[:, 0]
    codes_q = codes_elements[:, 1]
    codes_period = _period_years(codes_a, environment.gm["SUN"])
    official_q = official_elements[:, 2]
    official_a = official_elements[:, 7]
    official_period = _period_years(
        official_a,
        environment.gm["SUN"],
    )
    period_residual_days = (codes_period - official_period) * 365.25
    semimajor_residual_km = (codes_a - official_a) * AU_KM
    perihelion_residual_km = (codes_q - official_q) * AU_KM

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(12.2, 10.2),
        sharex=True,
    )
    series = (
        (
            timeline_period,
            official_period,
            "osculating period [yr]",
            _residual_text(period_residual_days, "d"),
        ),
        (
            timeline_a,
            official_a,
            "osculating semimajor axis [au]",
            _residual_text(semimajor_residual_km, "km"),
        ),
        (
            timeline_q,
            official_q,
            "osculating perihelion distance [au]",
            _residual_text(perihelion_residual_km, "km"),
        ),
    )
    for axis, (codes_curve, official_values, ylabel, summary) in zip(
        axes,
        series,
        strict=True,
    ):
        axis.plot(
            timeline_year,
            codes_curve,
            color="#007C77",
            lw=1.9,
            zorder=3,
            label="CODES forward integration",
        )
        axis.plot(
            comparison_year,
            official_values,
            color="#E69F32",
            lw=0.9,
            alpha=0.58,
            zorder=2,
        )
        axis.scatter(
            comparison_year,
            official_values,
            marker="D",
            s=20,
            facecolor="#FFD166",
            edgecolor="#713F24",
            linewidth=0.55,
            alpha=0.88,
            zorder=4,
            label=(
                "NASA/JPL Horizons #75 "
                f"({len(comparison_jd)} epochs)"
            ),
        )
        for year in (1910, 1986):
            axis.axvline(
                year,
                color="#A7354D",
                lw=0.85,
                ls="--",
                alpha=0.5,
            )
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22)
        axis.text(
            0.985,
            0.93,
            summary,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.7,
            color="#263441",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#D8DEE5",
                "alpha": 0.94,
            },
        )

    axes[0].legend(
        frameon=False,
        ncol=2,
        fontsize=9,
        loc="lower left",
    )
    axes[2].set_xlabel("year [TDB]")
    figure.suptitle(
        (
            "Dense NASA/JPL comparison for the 150-year "
            "CODES propagation of 1P/Halley"
        ),
        fontsize=15.5,
    )
    figure.text(
        0.5,
        0.014,
        (
            "JPL #75 sampled every 6 months, with monthly sampling within "
            "2 years of the 1910 and 1986 perihelia. JPL states are used "
            "only for post-propagation comparison."
        ),
        ha="center",
        color="#5C6875",
        fontsize=9.2,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.965))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=220, facecolor="white")
    plt.close(figure)

    provenance = {
        "generated_utc": time.strftime("%Y-%m-%d", time.gmtime()),
        "purpose": "Independent long-term propagation consistency test",
        "initial_state_source": (
            "NASA/JPL Horizons record 90000030 at 1850-01-02 TDB"
        ),
        "comparison_source": (
            "NASA/JPL Horizons record 90000030 at 397 comparison epochs, "
            "plus records 90000029 and 90000030 at the two perihelia"
        ),
        "dense_comparison_sampling": (
            "Six-month cadence from 1850 to 2000, augmented by monthly "
            "sampling within two years of the 1910 and 1986 perihelia"
        ),
        "integration_interval_tdb": [START_TDB, STOP_TDB],
        "force_model": {
            "A1_au_day2": A1_AU_DAY2,
            "A2_au_day2": A2_AU_DAY2,
            "nongrav_law": "Marsden standard water-ice law",
            "outgassing_lag_days": 0.0,
            "relativity": (
                "full massless-target Einstein-Infeld-Hoffmann 1PN"
            ),
            "major_bodies": "DE440s",
            "large_asteroids": "SB441-N16",
        },
        "kernel": kernel,
        "adaptive_step": (
            "embedded error control plus 5% gravity/crossing-time limiter"
        ),
        "function_evaluations": function_evaluations,
        "interpretation": (
            "This validates forward propagation from one fitted initial "
            "state. It is not a blind orbit fit because JPL solution #75 "
            "uses observations through 1994."
        ),
    }
    PROVENANCE_PATH.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    rows, jd_tdb, states, kernel, evaluations = run_validation()
    write_outputs(rows, jd_tdb, states, kernel, evaluations)
    print(json.dumps(rows, indent=2))
    print(FIGURE_PATH)
    print(CSV_PATH)
    print(PROVENANCE_PATH)


if __name__ == "__main__":
    main()
