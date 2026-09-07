#!/usr/bin/env python3
"""Build a five-year schedule from the normalized observing database.

The calculation reads all program, target, exposure, cadence, and mission
constraints from observing_schedule_inputs.sqlite. Fixed survey fields and
named targets are tested daily against the configured solar-elongation field
of regard and aggregated into weekly bins. Continuous surveys are assigned
with a linear program. Named-target visits use a mixed-integer program.

This is a reference scheduler, not a flight schedule. The Appendix tables do
not yet provide a common overhead model, a launch date, final survey polygons,
or complete exoplanet orbital orientations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json
import math
import sqlite3
import textwrap
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import lil_matrix

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from astropy.coordinates import get_sun
    from astropy.time import Time


ROOT = Path(__file__).resolve().parent
DATABASE_PATH = ROOT / "observing_schedule_inputs.sqlite"
if not DATABASE_PATH.exists():
    raise FileNotFoundError(
        f"Missing {DATABASE_PATH}. Run build_observing_schedule_database.py first."
    )
DB = sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True)
DB.row_factory = sqlite3.Row


def decode_setting(value: str, value_type: str) -> object:
    if value_type == "int":
        return int(value)
    if value_type == "float":
        return float(value)
    if value_type == "bool":
        return value.lower() == "true"
    if value_type == "json":
        return json.loads(value)
    return value


CONFIG: dict[str, dict[str, object]] = {}
for setting in DB.execute("SELECT section, key, value, value_type FROM settings"):
    CONFIG.setdefault(setting["section"], {})[setting["key"]] = decode_setting(
        setting["value"], setting["value_type"]
    )

OVERLEAF_COMMIT = DB.execute(
    "SELECT value FROM metadata WHERE key='overleaf_commit'"
).fetchone()[0]
DATABASE_SHA256 = hashlib.sha256(DATABASE_PATH.read_bytes()).hexdigest()
MISSION_START_YEAR = int(CONFIG["mission"]["start_year"])
N_YEARS = int(CONFIG["mission"]["years"])
WEEKS_PER_YEAR = int(CONFIG["mission"]["weeks_per_year"])
N_WEEKS = N_YEARS * WEEKS_PER_YEAR
WALL_HOURS_PER_WEEK = float(CONFIG["mission"]["wall_hours_per_week"])
SCIENCE_START_WEEK_OF_YEAR = int(
    CONFIG["mission"]["science_start_week_of_year"]
)
FIVE_YEAR_WALL_HOURS = 5.0 * 365.25 * 24.0
SUN_ELONGATION_MIN = float(CONFIG["visibility"]["solar_elongation_min_deg"])
SUN_ELONGATION_MAX = float(CONFIG["visibility"]["solar_elongation_max_deg"])
MIN_VISIBLE_FRACTION = (
    float(CONFIG["visibility"]["minimum_visible_days_per_week"]) / 7.0
)
# Appendix C does not state a blind-survey mission total, and Appendix X does
# not state a mission-wide alert total. These are explicit scenario inputs.
NEO_BLIND_FRACTION = float(CONFIG["appendix_c"]["blind_survey_fraction"])
NEO_RESERVE_FRACTION = float(CONFIG["appendix_c"]["recovery_too_fraction"])
COMPACT_TOO_RESERVE_FRACTION = float(CONFIG["appendix_x"]["transient_too_fraction"])
OPS = CONFIG["operations"]

OUT_SCHEDULE = ROOT / "five_year_schedule.csv"
OUT_CAPACITY = ROOT / "five_year_weekly_capacity.csv"
OUT_SUMMARY = ROOT / "five_year_schedule_summary.csv"
OUT_AUDIT = ROOT / "five_year_time_budget_audit.csv"
OUT_FIG_PNG = ROOT / "five_year_schedule_landscape.png"
OUT_FIG_PDF = ROOT / "five_year_schedule_landscape.pdf"
OUT_EXO = ROOT / "five_year_exoplanet_target_schedule.png"
OUT_DIAGNOSTICS = ROOT / "five_year_schedule_diagnostics.md"
OUT_APPENDIX_TIMELINES = {
    appendix: ROOT / f"appendix_{appendix}_five_year_timeline.png"
    for appendix in ("A", "B", "C", "X")
}

APPENDIX_TITLES = {
    "A": "COSMOLOGY AND GALAXY FORMATION",
    "B": "EXOPLANETS AND CIRCUMSTELLAR SYSTEMS",
    "C": "SOLAR-SYSTEM AND NEO OBSERVATIONS",
    "X": "COMPACT-OBJECT TIME-DOMAIN SCIENCE",
}


ROWS = [row["label"] for row in DB.execute(
    "SELECT label FROM display_rows ORDER BY display_order"
)]
FIELDS = {
    row["name"]: (float(row["ra_deg"]), float(row["dec_deg"]))
    for row in DB.execute(
        "SELECT name, ra_deg, dec_deg FROM targets "
        "WHERE target_group='field' AND active=1"
    )
}
FIRST_LIGHT = [
    (row["name"], row["host_name"], float(row["ra_deg"]), float(row["dec_deg"]))
    for row in DB.execute(
        "SELECT name, host_name, ra_deg, dec_deg FROM targets "
        "WHERE target_group='first_light' AND active=1 ORDER BY target_id"
    )
]
DIRECTOR_WEEKS = {
    (int(row["mission_year"]) - 1) * WEEKS_PER_YEAR
    + int(row["week_of_year"]) - 1
    for row in DB.execute(
        "SELECT mission_year, week_of_year FROM director_weeks"
    )
}
PROGRAM_META = {
    row["program_id"]: {
        "appendix": row["appendix"],
        "name": row["name"],
        "row": row["row_label"],
    }
    for row in DB.execute(
        """SELECT p.program_id, p.appendix, p.name, d.label AS row_label
        FROM programs p JOIN display_rows d USING(display_order)"""
    )
}


@dataclass
class ContinuousTask:
    key: str
    appendix: str
    program: str
    row: str
    target: str
    hours: float
    year_start: int
    year_end: int
    max_hours_week: float
    ra: float | None = None
    dec: float | None = None
    priority: float = 1.0
    week_of_year_start: int | None = None
    week_of_year_end: int | None = None


@dataclass
class Visit:
    key: str
    appendix: str
    program: str
    row: str
    target: str
    duration_h: float
    allowed_weeks: np.ndarray
    ra: float
    dec: float
    priority: float = 1.0


def mission_week_dates() -> list[list[datetime]]:
    weeks: list[list[datetime]] = []
    for year_index in range(N_YEARS):
        start = datetime(MISSION_START_YEAR + year_index, 1, 1, tzinfo=timezone.utc)
        for week_index in range(WEEKS_PER_YEAR):
            weeks.append([start + timedelta(days=7 * week_index + day) for day in range(7)])
    return weeks


WEEK_DATES = mission_week_dates()
SUN_BY_DAY = []
for dates in WEEK_DATES:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        SUN_BY_DAY.append(get_sun(Time(dates)))
SUN_UNIT_BY_WEEK = []
for suns in SUN_BY_DAY:
    vectors = suns.cartesian.xyz.value.T
    SUN_UNIT_BY_WEEK.append(vectors / np.linalg.norm(vectors, axis=1)[:, None])


_VIS_CACHE: dict[tuple[float, float], np.ndarray] = {}


def weekly_visibility(ra: float | None, dec: float | None) -> np.ndarray:
    if ra is None or dec is None or not np.isfinite(ra) or not np.isfinite(dec):
        return np.ones(N_WEEKS)
    cache_key = (round(float(ra), 6), round(float(dec), 6))
    if cache_key in _VIS_CACHE:
        return _VIS_CACHE[cache_key].copy()
    ra_rad = np.radians(float(ra))
    dec_rad = np.radians(float(dec))
    target_unit = np.array([
        np.cos(dec_rad) * np.cos(ra_rad),
        np.cos(dec_rad) * np.sin(ra_rad),
        np.sin(dec_rad),
    ])
    fractions = np.zeros(N_WEEKS)
    for week_index, sun_units in enumerate(SUN_UNIT_BY_WEEK):
        # GCRS axes are non-rotating and aligned to ICRS. A direction-vector
        # dot product avoids mixing the barycentric and geocentric origins.
        sep = np.degrees(np.arccos(np.clip(sun_units @ target_unit, -1.0, 1.0)))
        fractions[week_index] = np.mean(
            (sep >= SUN_ELONGATION_MIN) & (sep <= SUN_ELONGATION_MAX)
        )
    _VIS_CACHE[cache_key] = fractions
    return fractions.copy()


def year_mask(
    year_start: int,
    year_end: int,
    allow_pre_science: bool = False,
) -> np.ndarray:
    result = np.zeros(N_WEEKS, dtype=bool)
    result[
        (year_start - 1) * WEEKS_PER_YEAR : year_end * WEEKS_PER_YEAR
    ] = True
    if not allow_pre_science:
        result[:SCIENCE_START_WEEK_OF_YEAR - 1] = False
    if DIRECTOR_WEEKS:
        result[list(DIRECTOR_WEEKS)] = False
    return result


def apply_week_of_year_window(
    mask: np.ndarray,
    week_of_year_start: int | None,
    week_of_year_end: int | None,
) -> np.ndarray:
    result = mask.copy()
    week_of_year = np.arange(N_WEEKS) % WEEKS_PER_YEAR + 1
    if week_of_year_start is not None:
        result &= week_of_year >= week_of_year_start
    if week_of_year_end is not None:
        result &= week_of_year <= week_of_year_end
    return result


def allowed_for_target(
    ra: float,
    dec: float,
    year_start: int,
    year_end: int,
    desired_week: int | None = None,
    half_window: int = 8,
    week_of_year_start: int | None = None,
    week_of_year_end: int | None = None,
) -> np.ndarray:
    visible = weekly_visibility(ra, dec) >= MIN_VISIBLE_FRACTION
    allowed = visible & year_mask(year_start, year_end)
    allowed = apply_week_of_year_window(
        allowed, week_of_year_start, week_of_year_end
    )
    if desired_week is not None:
        local = np.zeros(N_WEEKS, dtype=bool)
        lo = max(0, desired_week - half_window)
        hi = min(N_WEEKS, desired_week + half_window + 1)
        local[lo:hi] = True
        narrowed = allowed & local
        if np.any(narrowed):
            allowed = narrowed
    weeks = np.flatnonzero(allowed)
    if len(weeks) == 0:
        raise RuntimeError(
            f"No visible week for target at ({ra:.4f}, {dec:.4f}) "
            f"in mission years {year_start}-{year_end}"
        )
    return weeks


def append_record(
    records: list[dict],
    week: int,
    hours: float,
    appendix: str,
    program: str,
    row: str,
    target: str,
    allocation_type: str,
    visibility_fraction: float = 1.0,
) -> None:
    records.append(
        {
            "mission_week": week + 1,
            "mission_year": week // 52 + 1,
            "week_of_year": week % 52 + 1,
            "calendar_start": WEEK_DATES[week][0].date().isoformat(),
            "appendix": appendix,
            "program": program,
            "key_science_row": row,
            "target": target,
            "hours": float(hours),
            "weekly_wall_fraction": float(hours / WALL_HOURS_PER_WEEK),
            "allocation_type": allocation_type,
            "visibility_fraction": float(visibility_fraction),
        }
    )


def build_continuous_tasks() -> list[ContinuousTask]:
    query = """
        SELECT c.requirement_id, p.appendix, p.name AS program, d.label AS row_label,
               COALESCE(t.name, c.target_label) AS target, c.total_hours,
               c.year_start, c.year_end, c.max_hours_per_week, c.priority,
               c.week_of_year_start, c.week_of_year_end,
               t.ra_deg, t.dec_deg
        FROM continuous_requirements c
        JOIN programs p USING (program_id)
        JOIN display_rows d
          ON d.display_order = COALESCE(c.display_order, p.display_order)
        LEFT JOIN targets t USING (target_id)
        ORDER BY c.requirement_id
    """
    tasks = []
    for record in DB.execute(query):
        tasks.append(
            ContinuousTask(
                record["requirement_id"], record["appendix"], record["program"],
                record["row_label"], record["target"], float(record["total_hours"]),
                int(record["year_start"]), int(record["year_end"]),
                float(record["max_hours_per_week"]),
                None if record["ra_deg"] is None else float(record["ra_deg"]),
                None if record["dec_deg"] is None else float(record["dec_deg"]),
                float(record["priority"]),
                (None if record["week_of_year_start"] is None
                 else int(record["week_of_year_start"])),
                (None if record["week_of_year_end"] is None
                 else int(record["week_of_year_end"])),
            )
        )
    return tasks


def make_visit_chunks(
    visits: list[Visit],
    key_prefix: str,
    appendix: str,
    program: str,
    row: str,
    target: str,
    total_hours: float,
    ra: float,
    dec: float,
    year_start: int,
    year_end: int,
    max_chunk: float = 20.0,
    desired_week: int | None = None,
    half_window: int = 8,
    priority: float = 1.0,
    week_of_year_start: int | None = None,
    week_of_year_end: int | None = None,
) -> None:
    n_chunks = max(1, math.ceil(total_hours / max_chunk))
    chunk = total_hours / n_chunks
    allowed = allowed_for_target(
        ra, dec, year_start, year_end, desired_week=desired_week,
        half_window=half_window,
        week_of_year_start=week_of_year_start,
        week_of_year_end=week_of_year_end,
    )
    for index in range(n_chunks):
        visits.append(
            Visit(
                f"{key_prefix}-{index + 1}", appendix, program, row, target,
                chunk, allowed.copy(), ra, dec, priority,
            )
        )


def build_named_visits() -> tuple[list[Visit], pd.DataFrame]:
    visits: list[Visit] = []
    query = """
        SELECT v.requirement_id, p.appendix, p.name AS program, d.label AS row_label,
               t.name AS target, t.ra_deg, t.dec_deg, v.total_hours,
               v.year_start, v.year_end, v.max_chunk_hours,
               v.desired_mission_week, v.half_window_weeks, v.priority,
               v.week_of_year_start, v.week_of_year_end
        FROM visit_requirements v
        JOIN programs p USING (program_id)
        JOIN display_rows d USING (display_order)
        JOIN targets t USING (target_id)
        WHERE t.active=1
        ORDER BY v.requirement_id
    """
    for record in DB.execute(query):
        make_visit_chunks(
            visits, record["requirement_id"], record["appendix"], record["program"],
            record["row_label"], record["target"], float(record["total_hours"]),
            float(record["ra_deg"]), float(record["dec_deg"]),
            int(record["year_start"]), int(record["year_end"]),
            max_chunk=float(record["max_chunk_hours"]),
            desired_week=(None if record["desired_mission_week"] is None
                          else int(record["desired_mission_week"])),
            half_window=int(record["half_window_weeks"]),
            priority=float(record["priority"]),
            week_of_year_start=(
                None if record["week_of_year_start"] is None
                else int(record["week_of_year_start"])
            ),
            week_of_year_end=(
                None if record["week_of_year_end"] is None
                else int(record["week_of_year_end"])
            ),
        )

    hz = pd.read_sql_query(
        """SELECT name, ra_deg AS ra, dec_deg AS dec, weight AS t_char_h
        FROM targets WHERE hz_enabled=1 AND active=1 ORDER BY target_id""",
        DB,
    )
    return visits, hz


def fixed_allocations(records: list[dict]) -> np.ndarray:
    load = np.zeros(N_WEEKS)
    operations_target = float(OPS["indirect_overhead_fraction"]) * FIVE_YEAR_WALL_HOURS
    wavefront_h = float(OPS["wavefront_fraction"]) * WALL_HOURS_PER_WEEK
    momentum_interval = int(OPS["momentum_unload_interval_weeks"])
    momentum_h = float(OPS["momentum_unload_hours"])
    station_interval = int(OPS["station_keeping_interval_weeks"])
    station_h = float(OPS["station_keeping_hours"])
    quarterly_interval = int(OPS["quarterly_calibration_interval_weeks"])
    quarterly_h = float(OPS["quarterly_calibration_hours"])
    momentum_events = N_WEEKS // momentum_interval
    station_events = N_WEEKS // station_interval
    quarterly_events = N_WEEKS // quarterly_interval
    recurring_total = (
        wavefront_h * N_WEEKS
        + momentum_h * momentum_events
        + station_h * station_events
        + quarterly_h * quarterly_events
    )
    operations_baseline_h = (operations_target - recurring_total) / N_WEEKS
    if operations_baseline_h < 0:
        raise RuntimeError("Operations event model exceeds the configured overhead total")

    fraction_rules = list(DB.execute(
        """SELECT w.*, p.appendix, p.name AS program, d.label AS row_label
        FROM weekly_fraction_rules w
        JOIN programs p USING(program_id)
        JOIN display_rows d USING(display_order)
        ORDER BY w.rule_id"""
    ))
    rule_hours: dict[str, float] = {}
    science_week_mask = year_mask(1, N_YEARS)
    for rule in fraction_rules:
        eligible = int(np.count_nonzero(science_week_mask))
        if eligible <= 0:
            raise RuntimeError(f"No eligible weeks for fixed rule {rule['rule_id']}")
        if rule["preserve_mission_total"]:
            rule_hours[rule["rule_id"]] = (
                float(rule["fraction_of_wall_clock"])
                * WALL_HOURS_PER_WEEK * N_WEEKS / eligible
            )
        else:
            rule_hours[rule["rule_id"]] = (
                float(rule["fraction_of_wall_clock"]) * WALL_HOURS_PER_WEEK
            )

    for week in range(N_WEEKS):
        general = PROGRAM_META["ops_general"]
        append_record(records, week, operations_baseline_h, "OPS",
                      general["name"], general["row"],
                      "Operations and calibration", "indirect overhead")
        wavefront = PROGRAM_META["ops_wfs"]
        append_record(records, week, wavefront_h, "OPS",
                      wavefront["name"], wavefront["row"],
                      "Segmented primary mirror", "maintenance")
        load[week] += operations_baseline_h + wavefront_h
        if (week + 1) % station_interval == 0:
            program = PROGRAM_META["ops_station"]
            append_record(records, week, station_h, program["appendix"],
                          program["name"], program["row"], "Spacecraft orbit",
                          "maintenance")
            load[week] += station_h
        if (week + 1) % momentum_interval == 0:
            program = PROGRAM_META["ops_momentum"]
            append_record(records, week, momentum_h, program["appendix"],
                          program["name"], program["row"], "Reaction wheels",
                          "maintenance")
            load[week] += momentum_h
        if (week + 1) % quarterly_interval == 0:
            program = PROGRAM_META["ops_quarterly"]
            append_record(records, week, quarterly_h, program["appendix"],
                          program["name"], program["row"],
                          "Instruments and reference fields", "maintenance")
            load[week] += quarterly_h

        if week in DIRECTOR_WEEKS:
            director = PROGRAM_META["director"]
            hours = WALL_HOURS_PER_WEEK - load[week]
            append_record(
                records, week, hours, director["appendix"], director["name"],
                director["row"], "Director-controlled observations",
                "director reserve",
            )
            load[week] += hours
            continue

        if not science_week_mask[week]:
            continue

        for rule in fraction_rules:
            if rule["skip_director_weeks"] and week in DIRECTOR_WEEKS:
                continue
            hours = rule_hours[rule["rule_id"]]
            append_record(
                records, week, hours, rule["appendix"], rule["program"],
                rule["row_label"], rule["target_label"], rule["allocation_type"],
            )
            load[week] += hours

    # Recurring visible-field requirements are read from the database. Their
    # hours remain part of the corresponding continuous-program total.
    recurring_query = """
        SELECT r.*, p.appendix, p.name AS program, d.label AS row_label,
               t.name AS target, t.ra_deg, t.dec_deg
        FROM recurring_requirements r
        JOIN programs p USING(program_id)
        JOIN display_rows d USING(display_order)
        JOIN targets t USING(target_id)
        WHERE t.active=1
        ORDER BY r.requirement_id
    """
    for rule in DB.execute(recurring_query):
        ra, dec = float(rule["ra_deg"]), float(rule["dec_deg"])
        vis = weekly_visibility(ra, dec)
        mask = year_mask(int(rule["year_start"]), int(rule["year_end"]))
        mask &= vis >= MIN_VISIBLE_FRACTION
        for week in np.flatnonzero(mask):
            hours = float(rule["hours_per_visible_week"])
            append_record(
                records, week, hours, rule["appendix"], rule["program"],
                rule["row_label"], rule["target"], rule["allocation_type"],
                vis[week],
            )
            load[week] += hours
    return load


def solve_continuous(
    tasks: list[ContinuousTask],
    fixed_load: np.ndarray,
    records: list[dict],
) -> tuple[np.ndarray, str]:
    # Remove recurring hours already charged against each continuous target.
    recurring_programs = {
        row[0] for row in DB.execute(
            """SELECT p.name FROM recurring_requirements r
            JOIN programs p USING(program_id)"""
        )
    }
    fixed_by_target: dict[str, float] = {}
    for record in records:
        if record["program"] in recurring_programs:
            fixed_by_target[record["target"]] = (
                fixed_by_target.get(record["target"], 0.0) + record["hours"]
            )
    adjusted = []
    for task in tasks:
        hours = task.hours - fixed_by_target.get(task.target, 0.0)
        adjusted.append((task, max(0.0, hours)))

    pairs: list[tuple[int, int, float, float]] = []
    task_pair_indices: list[list[int]] = [[] for _ in adjusted]
    week_pair_indices: list[list[int]] = [[] for _ in range(N_WEEKS)]
    for task_index, (task, hours) in enumerate(adjusted):
        vis = weekly_visibility(task.ra, task.dec)
        allow_pre_science = bool(
            task.week_of_year_start is not None
            and task.week_of_year_end is not None
            and task.week_of_year_end < SCIENCE_START_WEEK_OF_YEAR
        )
        active = year_mask(
            task.year_start,
            task.year_end,
            allow_pre_science=allow_pre_science,
        )
        active = apply_week_of_year_window(
            active, task.week_of_year_start, task.week_of_year_end
        )
        for week in np.flatnonzero(active & (vis >= MIN_VISIBLE_FRACTION)):
            ub = min(task.max_hours_week, task.max_hours_week * vis[week])
            pair_index = len(pairs)
            pairs.append((task_index, int(week), float(ub), float(vis[week])))
            task_pair_indices[task_index].append(pair_index)
            week_pair_indices[week].append(pair_index)

    n_pair = len(pairs)
    peak_index = n_pair
    n_var = n_pair + 1
    c = np.zeros(n_var)
    for pair_index, (task_index, week, ub, vis) in enumerate(pairs):
        task = adjusted[task_index][0]
        c[pair_index] = 2e-4 * (1.0 - vis) - 1e-7 * task.priority
    c[peak_index] = 1.0

    a_eq = lil_matrix((len(adjusted), n_var))
    b_eq = np.zeros(len(adjusted))
    for task_index, (task, hours) in enumerate(adjusted):
        if not task_pair_indices[task_index] and hours > 1e-9:
            raise RuntimeError(f"No allowed weeks for continuous task {task.key}")
        for pair_index in task_pair_indices[task_index]:
            a_eq[task_index, pair_index] = 1.0
        b_eq[task_index] = hours

    a_ub = lil_matrix((2 * N_WEEKS, n_var))
    b_ub = np.zeros(2 * N_WEEKS)
    for week in range(N_WEEKS):
        for pair_index in week_pair_indices[week]:
            a_ub[week, pair_index] = 1.0
            a_ub[N_WEEKS + week, pair_index] = 1.0
        a_ub[N_WEEKS + week, peak_index] = -1.0
        b_ub[week] = WALL_HOURS_PER_WEEK - fixed_load[week]
        b_ub[N_WEEKS + week] = -fixed_load[week]

    bounds = [(0.0, p[2]) for p in pairs] + [(0.0, WALL_HOURS_PER_WEEK)]
    result = linprog(
        c, A_ub=a_ub.tocsr(), b_ub=b_ub, A_eq=a_eq.tocsr(), b_eq=b_eq,
        bounds=bounds, method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Continuous schedule failed: {result.message}")

    load = fixed_load.copy()
    for pair_index, value in enumerate(result.x[:n_pair]):
        if value <= 1e-6:
            continue
        task_index, week, ub, vis = pairs[pair_index]
        task = adjusted[task_index][0]
        append_record(records, week, value, task.appendix, task.program, task.row,
                      task.target, "continuous LP", vis)
        load[week] += value
    return load, result.message


def solve_visits(
    visits: list[Visit],
    existing_load: np.ndarray,
    records: list[dict],
) -> tuple[np.ndarray, str]:
    pairs: list[tuple[int, int]] = []
    visit_pairs: list[list[int]] = [[] for _ in visits]
    week_pairs: list[list[int]] = [[] for _ in range(N_WEEKS)]
    for visit_index, visit in enumerate(visits):
        for week in visit.allowed_weeks:
            pair_index = len(pairs)
            pairs.append((visit_index, int(week)))
            visit_pairs[visit_index].append(pair_index)
            week_pairs[int(week)].append(pair_index)

    n_pair = len(pairs)
    peak_index = n_pair
    n_var = n_pair + 1
    c = np.zeros(n_var)
    for pair_index, (visit_index, week) in enumerate(pairs):
        visit = visits[visit_index]
        vis = weekly_visibility(visit.ra, visit.dec)[week]
        c[pair_index] = (
            1e-4 * existing_load[week] / WALL_HOURS_PER_WEEK
            + 2e-4 * (1.0 - vis)
            - 1e-7 * visit.priority
        )
    c[peak_index] = 1.0

    a_eq = lil_matrix((len(visits), n_var))
    for visit_index, indices in enumerate(visit_pairs):
        for pair_index in indices:
            a_eq[visit_index, pair_index] = 1.0
    b_eq = np.ones(len(visits))

    a_ub = lil_matrix((2 * N_WEEKS, n_var))
    b_ub = np.zeros(2 * N_WEEKS)
    for week in range(N_WEEKS):
        for pair_index in week_pairs[week]:
            visit_index, _ = pairs[pair_index]
            duration = visits[visit_index].duration_h
            a_ub[week, pair_index] = duration
            a_ub[N_WEEKS + week, pair_index] = duration
        a_ub[N_WEEKS + week, peak_index] = -1.0
        b_ub[week] = WALL_HOURS_PER_WEEK - existing_load[week]
        b_ub[N_WEEKS + week] = -existing_load[week]

    integrality = np.ones(n_var, dtype=int)
    integrality[peak_index] = 0
    lower = np.zeros(n_var)
    upper = np.ones(n_var)
    upper[peak_index] = WALL_HOURS_PER_WEEK
    result = milp(
        c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=[
            LinearConstraint(a_eq.tocsr(), b_eq, b_eq),
            LinearConstraint(a_ub.tocsr(), -np.inf, b_ub),
        ],
        options={"time_limit": 30.0, "mip_rel_gap": 0.01},
    )

    def valid_incumbent(values: np.ndarray | None) -> bool:
        if values is None:
            return False
        chosen = values[:n_pair] >= 0.5
        visit_count = np.zeros(len(visits), dtype=int)
        added = np.zeros(N_WEEKS)
        for pair_index in np.flatnonzero(chosen):
            visit_index, week = pairs[pair_index]
            visit_count[visit_index] += 1
            added[week] += visits[visit_index].duration_h
        return bool(
            np.all(visit_count == 1)
            and np.all(existing_load + added <= WALL_HOURS_PER_WEEK + 1e-6)
        )

    if valid_incumbent(result.x):
        solution = result.x
        status = result.message
        if not result.success:
            status += "; accepted verified feasible incumbent"
    else:
        # Capacity is deliberately loose in the reference schedule. This
        # deterministic fallback preserves every hard visibility and capacity
        # constraint if HiGHS has not emitted an incumbent before its limit.
        solution = np.zeros(n_var)
        greedy_load = existing_load.copy()
        order = sorted(
            range(len(visits)),
            key=lambda i: (len(visits[i].allowed_weeks), -visits[i].priority,
                           -visits[i].duration_h, visits[i].key),
        )
        for visit_index in order:
            visit = visits[visit_index]
            visibility = weekly_visibility(visit.ra, visit.dec)
            feasible = [
                int(week) for week in visit.allowed_weeks
                if greedy_load[int(week)] + visit.duration_h
                <= WALL_HOURS_PER_WEEK + 1e-9
            ]
            if not feasible:
                raise RuntimeError(f"No capacity for named visit {visit.key}")
            week = min(
                feasible,
                key=lambda w: (greedy_load[w] - 0.05 * visibility[w], w),
            )
            pair_index = next(
                p for p in visit_pairs[visit_index] if pairs[p][1] == week
            )
            solution[pair_index] = 1.0
            greedy_load[week] += visit.duration_h
        status = f"{result.message}; deterministic feasible fallback"

    load = existing_load.copy()
    for pair_index, value in enumerate(solution[:n_pair]):
        if value < 0.5:
            continue
        visit_index, week = pairs[pair_index]
        visit = visits[visit_index]
        vis = weekly_visibility(visit.ra, visit.dec)[week]
        allocation_type = (
            "target-visit MILP" if "fallback" not in status
            else "target-visit deterministic fallback"
        )
        append_record(records, week, visit.duration_h, visit.appendix,
                      visit.program, visit.row, visit.target,
                      allocation_type, vis)
        load[week] += visit.duration_h
    return load, status


def summarize_ranges(values: np.ndarray) -> str:
    weeks = np.flatnonzero(values >= MIN_VISIBLE_FRACTION) % 52 + 1
    weeks = np.unique(weeks)
    if len(weeks) == 0:
        return "none"
    runs = []
    start = prev = int(weeks[0])
    for value in weeks[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
        else:
            runs.append((start, prev))
            start = prev = value
    runs.append((start, prev))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def month_boundaries(year: int = MISSION_START_YEAR) -> tuple[list[float], list[str]]:
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    boundaries = []
    for month in range(1, 13):
        day = datetime(year, month, 1, tzinfo=timezone.utc)
        boundaries.append((day - start).days / 7.0)
    boundaries.append(52.0)
    centers = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(12)]
    return centers, labels


def make_landscape(schedule: pd.DataFrame, capacity: pd.DataFrame) -> None:
    display_rows = ROWS
    display_schedule = schedule.copy()
    display_schedule["display_row"] = display_schedule.key_science_row
    grouped = display_schedule.groupby(
        ["mission_year", "week_of_year", "display_row"], as_index=False
    ).hours.sum()
    matrix = np.zeros((N_YEARS, len(display_rows), WEEKS_PER_YEAR))
    for _, record in grouped.iterrows():
        matrix[
            int(record.mission_year) - 1,
            display_rows.index(record.display_row),
            int(record.week_of_year) - 1,
        ] = record.hours
    percent = 100.0 * matrix / WALL_HOURS_PER_WEEK
    scale_percent = percent.copy()
    director_row = display_rows.index("Director discretionary time")
    scale_percent[:, director_row, :] = 0.0
    for mission_week in DIRECTOR_WEEKS:
        year_index = mission_week // WEEKS_PER_YEAR
        week_index = mission_week % WEEKS_PER_YEAR
        scale_percent[year_index, :, week_index] = 0.0
    global_max_percent = float(np.max(scale_percent))
    if global_max_percent <= 0:
        global_max_percent = 1.0

    cmap = LinearSegmentedColormap.from_list(
        "allocation",
        [
            "#275dad", "#1f9ed1", "#20bfbb", "#52c569",
            "#d7df3f", "#f6c445", "#f28e2b", "#d73027",
        ],
    )
    norm = Normalize(0, global_max_percent)
    fig, axes = plt.subplots(N_YEARS, 1, figsize=(16, 9), sharex=True,
                             gridspec_kw={"hspace": 0.12})
    fig.patch.set_facecolor("white")
    for year_index, ax in enumerate(axes):
        ax.set_facecolor("white")
        for row_index in range(len(display_rows)):
            for week_index in range(52):
                mission_week = year_index * WEEKS_PER_YEAR + week_index
                if row_index == director_row or mission_week in DIRECTOR_WEEKS:
                    continue
                value = percent[year_index, row_index, week_index]
                if value <= 0.02:
                    continue
                ax.add_patch(
                    Rectangle(
                        (week_index, row_index - 0.39), 0.94, 0.78,
                        facecolor=cmap(norm(value)), edgecolor="#ffffff",
                        linewidth=0.22,
                    )
                )
        for week in range(53):
            ax.axvline(week, color="#e7ebf0", lw=0.28, zorder=0)
        for row in range(len(display_rows) + 1):
            ax.axhline(row - 0.5, color="#d5dbe3", lw=0.40, zorder=0)
        for week_index in range(WEEKS_PER_YEAR):
            mission_week = year_index * WEEKS_PER_YEAR + week_index
            if mission_week not in DIRECTOR_WEEKS:
                continue
            ax.add_patch(
                Rectangle(
                    (week_index, -0.5), 1.0, len(display_rows),
                    facecolor="#d1d5db", edgecolor="#6b7280", linewidth=0.55,
                    hatch="////", alpha=0.78, zorder=4,
                )
            )
            ax.add_patch(
                Rectangle(
                    (week_index, director_row - 0.39), 0.94, 0.78,
                    facecolor="#d73027", edgecolor="#ffffff",
                    linewidth=0.35, zorder=6,
                )
            )
        ax.set_ylim(len(display_rows) - 0.5, -0.5)
        ax.set_yticks(range(len(display_rows)))
        ax.set_yticklabels(display_rows, fontsize=7.8, color="#172033")
        ax.tick_params(axis="y", length=0, pad=4)
        for spine in ax.spines.values():
            spine.set_color("#aeb8c5")
            spine.set_linewidth(0.6)

    centers, month_labels = month_boundaries()
    top = axes[0].secondary_xaxis("top")
    top.set_xticks(centers)
    top.set_xticklabels(month_labels, fontsize=9, color="#172033")
    top.tick_params(length=0, pad=4)
    for ax in axes:
        for month_center in centers[1:]:
            boundary = (centers[centers.index(month_center) - 1] + month_center) / 2
            ax.axvline(boundary, color="#8d99a8", lw=0.75, zorder=0)
    axes[-1].set_xlim(0, 52)
    tick_weeks = np.arange(0, 52, 4)
    axes[-1].set_xticks(tick_weeks + 0.47)
    axes[-1].set_xticklabels([str(x + 1) for x in tick_weeks], fontsize=8, color="#465363")
    axes[-1].set_xlabel("week of mission year", fontsize=9, color="#465363")
    axes[-1].tick_params(axis="x", length=0)

    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar_axis = fig.add_axes([0.940, 0.17, 0.012, 0.68])
    cbar = fig.colorbar(scalar, cax=cbar_axis)
    cbar.set_ticks(np.linspace(0.0, global_max_percent, 6))
    cbar.set_ticklabels([f"{value:.1f}" for value in np.linspace(0.0, global_max_percent, 6)])
    cbar.set_label(
        f"weekly row allocation [%]  |  auto max = {global_max_percent:.1f}%",
        color="#172033", fontsize=9,
    )
    cbar.ax.tick_params(colors="#172033", labelsize=8)
    cbar.outline.set_edgecolor("#8d99a8")

    total = schedule.hours.sum()
    a_total = schedule.loc[schedule.appendix == "A", "hours"].sum()
    b_total = schedule.loc[schedule.appendix == "B", "hours"].sum()
    director_total = schedule.loc[schedule.appendix == "DDT", "hours"].sum()
    peak = capacity.used_hours.max()
    fig.suptitle("5-YEAR OBSERVING SCHEDULE", x=0.035, y=0.996, ha="left",
                 fontsize=18, color="#c64b1a", fontweight="bold")
    fig.text(0.035, 0.960,
             "Weekly allocation after solar-visibility and shared-capacity optimization",
             ha="left", fontsize=11.5, color="#334155")
    fig.text(
        0.035, 0.007,
        f"FoR {SUN_ELONGATION_MIN:.0f}°–{SUN_ELONGATION_MAX:.0f}°  |  "
        f"scheduled {total:,.0f} h ({100 * total / FIVE_YEAR_WALL_HOURS:.1f}% of 5-year wall clock)  |  "
        f"Appendix A {a_total:,.0f} h  |  Appendix B {b_total:,.0f} h  |  "
        f"Director {director_total:,.0f} h  |  peak week {peak:.1f} h "
        f"({100 * peak / WALL_HOURS_PER_WEEK:.0f}%)  |  "
        "red DDT cells and gray hatch = Director weeks",
        ha="left", fontsize=8.8, color="#526174",
    )
    axes[-1].set_xlabel("")
    fig.subplots_adjust(left=0.285, right=0.925, top=0.925, bottom=0.042)
    for year_index, ax in enumerate(axes):
        position = ax.get_position()
        year_axis = fig.add_axes([0.035, position.y0, 0.050, position.height])
        year_axis.set_facecolor("#eef2f6")
        year_axis.text(
            0.5, 0.5, f"YEAR\n{year_index + 1}", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#c64b1a",
        )
        year_axis.set_xticks([])
        year_axis.set_yticks([])
        for spine in year_axis.spines.values():
            spine.set_color("#c8d0da")
            spine.set_linewidth(0.7)
    fig.savefig(OUT_FIG_PNG, dpi=220, facecolor=fig.get_facecolor())
    fig.savefig(OUT_FIG_PDF, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_appendix_timelines(schedule: pd.DataFrame) -> None:
    """Render one database-driven weekly timeline for each science appendix."""

    cmap = LinearSegmentedColormap.from_list(
        "appendix_allocation",
        [
            "#275dad", "#1f9ed1", "#20bfbb", "#52c569",
            "#d7df3f", "#f6c445", "#f28e2b", "#d73027",
        ],
    )
    month_centers, month_labels = month_boundaries()

    for appendix, output_path in OUT_APPENDIX_TIMELINES.items():
        programs = [
            row["name"]
            for row in DB.execute(
                """SELECT name FROM programs WHERE appendix=?
                ORDER BY display_order, program_id""",
                (appendix,),
            )
        ]
        appendix_schedule = schedule[schedule.appendix == appendix]
        grouped = appendix_schedule.groupby(
            ["mission_year", "week_of_year", "program"], as_index=False
        ).hours.sum()
        matrix = np.zeros((N_YEARS, len(programs), WEEKS_PER_YEAR))
        for _, record in grouped.iterrows():
            matrix[
                int(record.mission_year) - 1,
                programs.index(record.program),
                int(record.week_of_year) - 1,
            ] = record.hours

        percent = 100.0 * matrix / WALL_HOURS_PER_WEEK
        maximum_percent = float(np.max(percent))
        if maximum_percent <= 0:
            maximum_percent = 1.0
        norm = Normalize(0.0, maximum_percent)
        figure_height = max(6.8, 5.8 + 0.38 * len(programs))
        fig, axes = plt.subplots(
            N_YEARS,
            1,
            figsize=(16, figure_height),
            sharex=True,
            gridspec_kw={"hspace": 0.12},
        )
        fig.patch.set_facecolor("white")
        wrapped_labels = [
            "\n".join(textwrap.wrap(program, width=31, break_long_words=False))
            for program in programs
        ]

        for year_index, ax in enumerate(axes):
            ax.set_facecolor("white")
            for row_index in range(len(programs)):
                for week_index in range(WEEKS_PER_YEAR):
                    value = percent[year_index, row_index, week_index]
                    if value <= 0.02:
                        continue
                    ax.add_patch(
                        Rectangle(
                            (week_index, row_index - 0.39),
                            0.94,
                            0.78,
                            facecolor=cmap(norm(value)),
                            edgecolor="#ffffff",
                            linewidth=0.22,
                        )
                    )
            for week in range(53):
                ax.axvline(week, color="#e7ebf0", lw=0.28, zorder=0)
            for row in range(len(programs) + 1):
                ax.axhline(row - 0.5, color="#d5dbe3", lw=0.40, zorder=0)
            for week_index in range(WEEKS_PER_YEAR):
                mission_week = year_index * WEEKS_PER_YEAR + week_index
                if mission_week not in DIRECTOR_WEEKS:
                    continue
                ax.add_patch(
                    Rectangle(
                        (week_index, -0.5),
                        1.0,
                        len(programs),
                        facecolor="#d1d5db",
                        edgecolor="#6b7280",
                        linewidth=0.55,
                        hatch="////",
                        alpha=0.78,
                        zorder=4,
                    )
                )
            ax.set_ylim(len(programs) - 0.5, -0.5)
            ax.set_yticks(range(len(programs)))
            ax.set_yticklabels(wrapped_labels, fontsize=7.6, color="#172033")
            ax.tick_params(axis="y", length=0, pad=4)
            for spine in ax.spines.values():
                spine.set_color("#aeb8c5")
                spine.set_linewidth(0.6)

        top_axis = axes[0].secondary_xaxis("top")
        top_axis.set_xticks(month_centers)
        top_axis.set_xticklabels(month_labels, fontsize=9, color="#172033")
        top_axis.tick_params(length=0, pad=4)
        for ax in axes:
            for month_index in range(1, len(month_centers)):
                boundary = (month_centers[month_index - 1] + month_centers[month_index]) / 2
                ax.axvline(boundary, color="#8d99a8", lw=0.75, zorder=0)
        axes[-1].set_xlim(0, WEEKS_PER_YEAR)
        tick_weeks = np.arange(0, WEEKS_PER_YEAR, 4)
        axes[-1].set_xticks(tick_weeks + 0.47)
        axes[-1].set_xticklabels(
            [str(value + 1) for value in tick_weeks],
            fontsize=8,
            color="#465363",
        )
        axes[-1].tick_params(axis="x", length=0)

        scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar_axis = fig.add_axes([0.940, 0.17, 0.012, 0.68])
        cbar = fig.colorbar(scalar, cax=cbar_axis)
        ticks = np.linspace(0.0, maximum_percent, 6)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{value:.1f}" for value in ticks])
        cbar.set_label(
            f"weekly program allocation [%]  |  auto max = {maximum_percent:.1f}%",
            color="#172033",
            fontsize=9,
        )
        cbar.ax.tick_params(colors="#172033", labelsize=8)
        cbar.outline.set_edgecolor("#8d99a8")

        total = float(appendix_schedule.hours.sum())
        fig.suptitle(
            f"APPENDIX {appendix} / {APPENDIX_TITLES[appendix]}",
            x=0.035,
            y=0.985,
            ha="left",
            fontsize=17,
            color="#c64b1a",
            fontweight="bold",
        )
        fig.text(
            0.035,
            0.915,
            "Weekly allocations from the shared five-year observing database",
            ha="left",
            fontsize=11,
            color="#334155",
        )
        fig.text(
            0.035,
            0.008,
            f"FoR {SUN_ELONGATION_MIN:.0f}°–{SUN_ELONGATION_MAX:.0f}°  |  "
            f"Appendix {appendix} total {total:,.1f} h "
            f"({100 * total / FIVE_YEAR_WALL_HOURS:.2f}% of the five-year wall clock)  |  "
            "gray hatch = protected Director weeks",
            ha="left",
            fontsize=8.8,
            color="#526174",
        )
        fig.subplots_adjust(left=0.285, right=0.925, top=0.865, bottom=0.042)
        for year_index, ax in enumerate(axes):
            position = ax.get_position()
            year_axis = fig.add_axes([0.035, position.y0, 0.050, position.height])
            year_axis.set_facecolor("#eef2f6")
            year_axis.text(
                0.5,
                0.5,
                f"YEAR\n{year_index + 1}",
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold",
                color="#c64b1a",
            )
            year_axis.set_xticks([])
            year_axis.set_yticks([])
            for spine in year_axis.spines.values():
                spine.set_color("#c8d0da")
                spine.set_linewidth(0.7)
        fig.savefig(output_path, dpi=220, facecolor=fig.get_facecolor())
        plt.close(fig)


def make_exoplanet_detail(schedule: pd.DataFrame, hz: pd.DataFrame) -> None:
    key_names = list(hz["name"])
    planet_names = []
    for _, host, _, _ in FIRST_LIGHT:
        if host not in key_names:
            key_names.append(host)
    for planet, _, _, _ in FIRST_LIGHT:
        planet_names.append(planet)
    sub = schedule[
        (schedule.appendix == "B")
        & (schedule.target.isin(key_names + planet_names))
    ].copy()
    fig, ax = plt.subplots(figsize=(16, 7.4))
    fig.patch.set_facecolor("#040848")
    ax.set_facecolor("#040848")
    colors = {
        "Nearest FGK HZ Systems Survey": "#51d1a8",
        "First-light known giants": "#ffc107",
        "Giant orbit determination": "#e97a4d",
        "Nearest Stellar Systems Survey": "#45a3ff",
    }
    ymap = {name: i for i, name in enumerate(key_names)}
    for _, record in sub.iterrows():
        host_name = record.target
        if record.program in ("First-light known giants", "Giant orbit determination"):
            for planet, host, _, _ in FIRST_LIGHT:
                if record.target == planet:
                    host_name = host
                    break
        if host_name not in ymap:
            continue
        y = ymap[host_name]
        color = colors.get(record.program, "#45a3ff")
        ax.add_patch(Rectangle((record.mission_week - 1, y - 0.34), 0.9, 0.68,
                               facecolor=color, edgecolor="none", alpha=0.9))
    for year in range(6):
        ax.axvline(year * 52, color="#aab0d6", lw=0.8)
    ax.set_xlim(0, N_WEEKS)
    ax.set_ylim(len(key_names) - 0.5, -0.5)
    ax.set_yticks(range(len(key_names)))
    ax.set_yticklabels(key_names, fontsize=9, color="#f8f7f1")
    ax.set_xticks([26 + 52 * i for i in range(5)])
    ax.set_xticklabels([f"Year {i + 1}" for i in range(5)], color="#f8f7f1")
    ax.grid(axis="y", color="#30397a", lw=0.35)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_color("#46508c")
    ax.set_title("NAMED EXOPLANET TARGET ALLOCATIONS", loc="left", fontsize=17,
                 color="#ff8a50", fontweight="bold", pad=14)
    handles = [Rectangle((0, 0), 1, 1, color=color, label=label)
               for label, color in colors.items()]
    ax.legend(handles=handles, loc="upper right", ncol=2, frameon=False,
              fontsize=8, labelcolor="#f8f7f1")
    fig.tight_layout()
    fig.savefig(OUT_EXO, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_outputs(
    records: list[dict],
    hz: pd.DataFrame,
    continuous_status: str,
    visit_status: str,
) -> None:
    schedule = pd.DataFrame(records).sort_values(
        ["mission_week", "key_science_row", "program", "target"]
    ).reset_index(drop=True)
    schedule.to_csv(OUT_SCHEDULE, index=False)

    weekly = schedule.groupby("mission_week", as_index=False).hours.sum()
    weekly = weekly.rename(columns={"hours": "used_hours"})
    weekly["mission_year"] = (weekly.mission_week - 1) // 52 + 1
    weekly["week_of_year"] = (weekly.mission_week - 1) % 52 + 1
    weekly["available_wall_hours"] = WALL_HOURS_PER_WEEK
    weekly["free_hours"] = weekly.available_wall_hours - weekly.used_hours
    weekly["used_fraction"] = weekly.used_hours / weekly.available_wall_hours
    weekly.to_csv(OUT_CAPACITY, index=False)

    summary = schedule.groupby(["appendix", "program"], as_index=False).hours.sum()
    summary["five_year_wall_fraction"] = summary.hours / FIVE_YEAR_WALL_HOURS
    summary.to_csv(OUT_SUMMARY, index=False)

    budget_rows = {
        row["appendix"]: row
        for row in DB.execute(
            """SELECT appendix, minimum_hours, nominal_hours, maximum_hours
            FROM budget_ranges WHERE appendix IN ('A', 'B')"""
        )
    }
    audit_rows = []
    for appendix in ("A", "B"):
        budget = budget_rows[appendix]
        audit_rows.append({
            "appendix": appendix,
            "minimum_hours": float(budget["minimum_hours"]),
            "nominal_hours": float(budget["nominal_hours"]),
            "maximum_hours": float(budget["maximum_hours"]),
        })
    a_budget = budget_rows["A"]
    b_budget = budget_rows["B"]
    audit_rows.append({
        "appendix": "A+B",
        "minimum_hours": float(a_budget["minimum_hours"] + b_budget["minimum_hours"]),
        "nominal_hours": float(a_budget["nominal_hours"] + b_budget["nominal_hours"]),
        "maximum_hours": float(a_budget["maximum_hours"] + b_budget["maximum_hours"]),
    })
    audit = pd.DataFrame(audit_rows)
    for column in ("minimum", "nominal", "maximum"):
        audit[f"{column}_fraction"] = audit[f"{column}_hours"] / FIVE_YEAR_WALL_HOURS
        audit[f"{column}_percent"] = 100.0 * audit[f"{column}_fraction"]
    audit.to_csv(OUT_AUDIT, index=False)

    make_landscape(schedule, weekly)
    make_appendix_timelines(schedule)
    make_exoplanet_detail(schedule, hz)

    vis_violations = schedule[
        (schedule.allocation_type != "reserved")
        & (schedule.visibility_fraction + 1e-9 < MIN_VISIBLE_FRACTION)
        & (~schedule.target.isin(["Reference fields", "Flexible target list", "Coronagraph reference stars", "Target list not fixed in Appendix B"]))
    ]
    appendix_totals = schedule.groupby("appendix").hours.sum().to_dict()
    total = schedule.hours.sum()
    peak_row = weekly.loc[weekly.used_hours.idxmax()]

    hz_rows = []
    for _, target in hz.iterrows():
        target_schedule = schedule[
            (schedule.target == target["name"])
            & (schedule.program == "Nearest FGK HZ Systems Survey")
        ]
        visibility = weekly_visibility(float(target.ra), float(target.dec))
        hz_rows.append(
            f"| {target['name']} | {summarize_ranges(visibility)} | "
            f"{target_schedule.hours.sum():.1f} | {int(target_schedule.mission_week.nunique())} |"
        )

    text = f"""# Five-Year Schedule Diagnostics

## Provenance and assumptions

- Latest fetched Overleaf revision: `{OVERLEAF_COMMIT}`.
- Authoritative input database: `{DATABASE_PATH.name}`.
- Input database SHA-256: `{DATABASE_SHA256}`.
- Mission-year template: calendar years {MISSION_START_YEAR}-{MISSION_START_YEAR + N_YEARS - 1}.
- Weekly grid: 52 seven-day bins per mission year.
- Year 1 commissioning and performance acceptance: W01-W{SCIENCE_START_WEEK_OF_YEAR - 1:02d}.
- Routine science begins in Year 1 W{SCIENCE_START_WEEK_OF_YEAR:02d}.
- Civil five-year wall clock: {FIVE_YEAR_WALL_HOURS:,.1f} h.
- Plotted 260-week grid: {N_WEEKS * WALL_HOURS_PER_WEEK:,.1f} h. The civil-calendar difference combines six omitted calendar days with the Julian-year convention used by the wall-clock denominator.
- Solar-elongation field of regard: {SUN_ELONGATION_MIN:.0f}-{SUN_ELONGATION_MAX:.0f} degrees.
- Weekly capacity: {WALL_HOURS_PER_WEEK:.0f} wall-clock hours.
- Director discretionary weeks: {', '.join(f'W{week + 1:02d}' for week in sorted({index % WEEKS_PER_YEAR for index in DIRECTOR_WEEKS}))} in every mission year.
- Appendix A nominal request: 6,650 h, the midpoint of 4,700-8,600 h.
- Appendix B nominal request: 6,350 h, the midpoint of 4,950-7,750 h.
- Appendix C optical blind-survey scenario: {100 * NEO_BLIND_FRACTION:.1f}% of the full mission wall clock, redistributed outside director weeks.
- Appendix C NEO recovery reserve: {100 * NEO_RESERVE_FRACTION:.1f}% of the full mission wall clock, redistributed outside director weeks.
- Appendix X transient reserve: {100 * COMPACT_TOO_RESERVE_FRACTION:.1f}% of the full mission wall clock, redistributed outside director weeks.
- Indirect observatory overhead: {100 * float(OPS['indirect_overhead_fraction']):.1f}% of the five-year wall clock.
- The Appendix C text explicitly separates the 5.0% optical blind-survey capacity scenario from the 2.0% external-alert recovery reserve. Neither line is a completeness or discovery-yield claim.
- Appendix X planned monitoring is calculated from the stated visit durations and cadence rather than from a mission-total number.
- HZ `t_char` values are used as relative weights and normalized to the 2,000 h midpoint. The Appendix states that they are photon-noise lower bounds, not validated exposure requests.
- Known-planet orbital orientation and phase are not sufficiently specified in the tracked cache. The first run applies target-specific solar visibility but not a claimed planet-phase optimum.

## Budget calculation

The reported mission fraction is

`fraction = requested hours / (5 x 365.25 x 24 h)`.

| Program | Minimum h | Nominal h | Maximum h | Minimum % | Nominal % | Maximum % |
|---|---:|---:|---:|---:|---:|---:|
| Appendix A | 4,700 | 6,650 | 8,600 | {100 * 4700 / FIVE_YEAR_WALL_HOURS:.2f} | {100 * 6650 / FIVE_YEAR_WALL_HOURS:.2f} | {100 * 8600 / FIVE_YEAR_WALL_HOURS:.2f} |
| Appendix B | 4,950 | 6,350 | 7,750 | {100 * 4950 / FIVE_YEAR_WALL_HOURS:.2f} | {100 * 6350 / FIVE_YEAR_WALL_HOURS:.2f} | {100 * 7750 / FIVE_YEAR_WALL_HOURS:.2f} |
| A plus B | 9,650 | 13,000 | 16,350 | {100 * 9650 / FIVE_YEAR_WALL_HOURS:.2f} | {100 * 13000 / FIVE_YEAR_WALL_HOURS:.2f} | {100 * 16350 / FIVE_YEAR_WALL_HOURS:.2f} |

Appendix A reaches approximately 20% only at its upper envelope. Appendix B
does not receive 30% of the mission. The older 30% value was a science
integration efficiency used by an illustrative exoplanet timeline. The
nominal combined Appendix A and B allocation is 29.66%.

## Visibility and capacity calculation

For target unit vector `n` and geocentric Sun unit vector `s(d)` on day `d`,
the scheduler evaluates `theta(d) = arccos[n dot s(d)]`. A day is visible when
{SUN_ELONGATION_MIN:.0f} degrees <= `theta` <= {SUN_ELONGATION_MAX:.0f} degrees.
A week is usable when at least {7 * MIN_VISIBLE_FRACTION:.0f} of its seven days
are visible. The continuous linear program conserves every requested program
total and enforces

`fixed overhead(t) + reserved time(t) + sum_p x(p,t) <= 168 h`

in every mission week. Named visits then enforce one allowed week per visit
under the remaining shared capacity.

## Maintenance model and sources

- Total indirect observatory overhead is fixed at {100 * float(OPS['indirect_overhead_fraction']):.1f}%, following the current JWST statistical accounting reference of about 16% for calibrations, momentum management, wavefront sensing and control, and other maintenance.
- Wavefront sensing and control is charged at {100 * float(OPS['wavefront_fraction']):.1f}% every week. STScI reports that routine JWST wavefront sensing and control takes less than 1.5% of total observatory time and that sensing currently occurs every four days.
- A {float(OPS['momentum_unload_hours']):.0f} h momentum unload is inserted every {int(OPS['momentum_unload_interval_weeks'])} weeks. STScI reports an operational cadence of roughly six weeks.
- A {float(OPS['station_keeping_hours']):.0f} h station-keeping block is inserted every {int(OPS['station_keeping_interval_weeks'])} weeks. STScI describes L2 station keeping roughly every two or three weeks.
- A {float(OPS['quarterly_calibration_hours']):.0f} h calibration block is inserted in weeks 13, 26, 39, and 52 of each mission year. This quarterly concentration is a 3.5ST scheduling assumption. HST likewise maintains cycle calibration programs for detector reference files and performance monitoring.
- JWST overhead accounting: https://jwst-docs.stsci.edu/jwst-general-support/jwst-observing-overheads-and-time-accounting-overview
- JWST optics maintenance: https://jwst-docs.stsci.edu/jwst-observatory-characteristics/optics-performance-stability
- JWST momentum management: https://jwst-docs.stsci.edu/jwst-observatory-hardware/jwst-attitude-control-subsystem/jwst-momentum-management
- JWST moving-target and station-keeping operations: https://jwst-docs.stsci.edu/methods-and-roadmaps/jwst-moving-target-observations/jwst-moving-target-supporting-technical-information/moving-target-ephemerides
- HST calibration-plan example: https://hst-docs.stsci.edu/wfc3ihb/appendix-e-reduction-and-calibration-of-wfc3-data/e-19-the-cycle-32-calibration-plan

## Time ledger

- Total scheduled: {total:,.1f} h, or {100 * total / FIVE_YEAR_WALL_HOURS:.2f}% of the five-year wall clock.
- Appendix A: {appendix_totals.get('A', 0):,.1f} h.
- Appendix B: {appendix_totals.get('B', 0):,.1f} h.
- Appendix C blind survey plus recovery reserve: {appendix_totals.get('C', 0):,.1f} h.
- Appendix X planned plus reserve: {appendix_totals.get('X', 0):,.1f} h.
- Observatory maintenance and indirect overhead: {appendix_totals.get('OPS', 0):,.1f} h.
- Director discretionary allocation: {appendix_totals.get('DDT', 0):,.1f} h across {len(DIRECTOR_WEEKS)} protected weeks.
- Peak week: mission week {int(peak_row.mission_week)}, {peak_row.used_hours:.1f} h, or {100 * peak_row.used_fraction:.1f}%.
- Minimum free time in any week: {weekly.free_hours.min():.1f} h.

## Optimization status

- Continuous linear program: {continuous_status}
- Named-target mixed-integer program: {visit_status}
- Visibility violations: {len(vis_violations)}
- Capacity violations: {int((weekly.used_hours > WALL_HOURS_PER_WEEK + 1e-6).sum())}

## Representative field seasons

- Boötes north visible weeks: {summarize_ranges(weekly_visibility(*FIELDS['Boötes north']))}.
- SXDS south visible weeks: {summarize_ranges(weekly_visibility(*FIELDS['SXDS south']))}.
- SN north visible weeks: {summarize_ranges(weekly_visibility(*FIELDS['SN north']))}.
- SN south visible weeks: {summarize_ranges(weekly_visibility(*FIELDS['SN south']))}.

## HZ target allocations

| Target | Visible weeks each year | Scheduled h | Scheduled weeks |
|---|---:|---:|---:|
{chr(10).join(hz_rows)}

## Interpretation limits

The result is the first shared-capacity reference schedule. It demonstrates
seasonal and time-budget feasibility under the stated assumptions. It is not a
flight schedule. Finalization requires a launch date, a mission-wide field of
regard, north and south tile polygons, measured overheads, guide-star and roll
constraints, a validated coronagraph ETC, and orbital-phase posteriors for the
known planets. Alert-driven NEO and merger targets remain reserves until an
actual alert supplies coordinates and an ephemeris.

## Reproduction and controlled changes

The authoritative target, exposure, visit, cadence, program, and mission
inputs are stored in `observing_schedule_inputs.sqlite`. Recalculate every
CSV, figure, and this report with

```bash
python make_five_year_optimized_schedule.py
```

Use `python build_observing_schedule_database.py --force` only when deliberately
rebuilding the baseline database from the tracked TOML and catalogue. Routine
target or exposure changes shall be made in the database and validated before
the scheduler is rerun.

The schedule CSV is the source for every plotted cell. The time-budget audit
is stored separately in `five_year_time_budget_audit.csv`. No proposal or
Overleaf source is modified by this calculation.
"""
    OUT_DIAGNOSTICS.write_text(text, encoding="utf-8")


def main() -> None:
    records: list[dict] = []
    fixed_load = fixed_allocations(records)
    continuous_tasks = build_continuous_tasks()
    load_after_continuous, continuous_status = solve_continuous(
        continuous_tasks, fixed_load, records
    )
    visits, hz = build_named_visits()
    final_load, visit_status = solve_visits(visits, load_after_continuous, records)
    write_outputs(records, hz, continuous_status, visit_status)
    print(f"continuous tasks: {len(continuous_tasks)}")
    print(f"named visits: {len(visits)}")
    print(f"peak weekly load: {final_load.max():.2f} / {WALL_HOURS_PER_WEEK:.0f} h")
    print(f"wrote {OUT_FIG_PNG}")
    for output_path in OUT_APPENDIX_TIMELINES.values():
        print(f"wrote {output_path}")
    print(f"wrote {OUT_EXO}")
    print(f"wrote {OUT_SCHEDULE}")
    print(f"wrote {OUT_AUDIT}")
    print(f"wrote {OUT_DIAGNOSTICS}")


if __name__ == "__main__":
    main()
