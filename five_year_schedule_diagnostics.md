# Five-Year Schedule Diagnostics

## Provenance and assumptions

- Latest fetched Overleaf revision: `ecb7dd2`.
- Authoritative input database: `observing_schedule_inputs.sqlite`.
- Input database SHA-256: `19a80774301c723140592a1e207521dce6b4c0b6caaae41877c94b7ec21af628`.
- Mission-year template: calendar years 2030-2034.
- Weekly grid: 52 seven-day bins per mission year.
- Year 1 commissioning and performance acceptance: W01-W13.
- Routine science begins in Year 1 W14.
- Civil five-year wall clock: 43,830.0 h.
- Plotted 260-week grid: 43,680.0 h. The civil-calendar difference combines six omitted calendar days with the Julian-year convention used by the wall-clock denominator.
- Solar-elongation field of regard: 90-180 degrees.
- Weekly capacity: 168 wall-clock hours.
- Director discretionary weeks: W43, W44 in every mission year.
- Appendix A nominal request: 6,650 h, the midpoint of 4,700-8,600 h.
- Appendix B nominal request: 6,350 h, the midpoint of 4,950-7,750 h.
- Appendix C optical blind-survey scenario: 5.0% of the full mission wall clock, redistributed outside director weeks.
- Appendix C NEO recovery reserve: 2.0% of the full mission wall clock, redistributed outside director weeks.
- Appendix X transient reserve: 1.0% of the full mission wall clock, redistributed outside director weeks.
- Indirect observatory overhead: 16.0% of the five-year wall clock.
- The Appendix C text explicitly separates the 5.0% optical blind-survey capacity scenario from the 2.0% external-alert recovery reserve. Neither line is a completeness or discovery-yield claim.
- Appendix X planned monitoring is calculated from the stated visit durations and cadence rather than from a mission-total number.
- HZ `t_char` values are used as relative weights and normalized to the 2,000 h midpoint. The Appendix states that they are photon-noise lower bounds, not validated exposure requests.
- Known-planet orbital orientation and phase are not sufficiently specified in the tracked cache. The first run applies target-specific solar visibility but not a claimed planet-phase optimum.

## Budget calculation

The reported mission fraction is

`fraction = requested hours / (5 x 365.25 x 24 h)`.

| Program | Minimum h | Nominal h | Maximum h | Minimum % | Nominal % | Maximum % |
|---|---:|---:|---:|---:|---:|---:|
| Appendix A | 4,700 | 6,650 | 8,600 | 10.72 | 15.17 | 19.62 |
| Appendix B | 4,950 | 6,350 | 7,750 | 11.29 | 14.49 | 17.68 |
| A plus B | 9,650 | 13,000 | 16,350 | 22.02 | 29.66 | 37.30 |

Appendix A reaches approximately 20% only at its upper envelope. Appendix B
does not receive 30% of the mission. The older 30% value was a science
integration efficiency used by an illustrative exoplanet timeline. The
nominal combined Appendix A and B allocation is 29.66%.

## Visibility and capacity calculation

For target unit vector `n` and geocentric Sun unit vector `s(d)` on day `d`,
the scheduler evaluates `theta(d) = arccos[n dot s(d)]`. A day is visible when
90 degrees <= `theta` <= 180 degrees.
A week is usable when at least 3 of its seven days
are visible. The continuous linear program conserves every requested program
total and enforces

`fixed overhead(t) + reserved time(t) + sum_p x(p,t) <= 168 h`

in every mission week. Named visits then enforce one allowed week per visit
under the remaining shared capacity.

## Maintenance model and sources

- Total indirect observatory overhead is fixed at 16.0%, following the current JWST statistical accounting reference of about 16% for calibrations, momentum management, wavefront sensing and control, and other maintenance.
- Wavefront sensing and control is charged at 1.5% every week. STScI reports that routine JWST wavefront sensing and control takes less than 1.5% of total observatory time and that sensing currently occurs every four days.
- A 3 h momentum unload is inserted every 6 weeks. STScI reports an operational cadence of roughly six weeks.
- A 2 h station-keeping block is inserted every 3 weeks. STScI describes L2 station keeping roughly every two or three weeks.
- A 24 h calibration block is inserted in weeks 13, 26, 39, and 52 of each mission year. This quarterly concentration is a 3.5ST scheduling assumption. HST likewise maintains cycle calibration programs for detector reference files and performance monitoring.
- JWST overhead accounting: https://jwst-docs.stsci.edu/jwst-general-support/jwst-observing-overheads-and-time-accounting-overview
- JWST optics maintenance: https://jwst-docs.stsci.edu/jwst-observatory-characteristics/optics-performance-stability
- JWST momentum management: https://jwst-docs.stsci.edu/jwst-observatory-hardware/jwst-attitude-control-subsystem/jwst-momentum-management
- JWST moving-target and station-keeping operations: https://jwst-docs.stsci.edu/methods-and-roadmaps/jwst-moving-target-observations/jwst-moving-target-supporting-technical-information/moving-target-ephemerides
- HST calibration-plan example: https://hst-docs.stsci.edu/wfc3ihb/appendix-e-reduction-and-calibration-of-wfc3-data/e-19-the-cycle-32-calibration-plan

## Time ledger

- Total scheduled: 25,117.0 h, or 57.31% of the five-year wall clock.
- Appendix A: 6,650.0 h.
- Appendix B: 6,350.0 h.
- Appendix C blind survey plus recovery reserve: 3,057.6 h.
- Appendix X planned plus reserve: 618.3 h.
- Observatory maintenance and indirect overhead: 7,012.8 h.
- Director discretionary allocation: 1,428.3 h across 10 protected weeks.
- Peak week: mission week 43, 168.0 h, or 100.0%.
- Minimum free time in any week: 0.0 h.

## Optimization status

- Continuous linear program: Optimization terminated successfully. (HiGHS Status 7: Optimal)
- Named-target mixed-integer program: Optimization terminated successfully. (HiGHS Status 7: Optimal)
- Visibility violations: 0
- Capacity violations: 0

## Representative field seasons

- Boötes north visible weeks: 2-28.
- SXDS south visible weeks: 1-3, 30-52.
- SN north visible weeks: 2-28.
- SN south visible weeks: 1-3, 29-52.

## HZ target allocations

| Target | Visible weeks each year | Scheduled h | Scheduled weeks |
|---|---:|---:|---:|
| 61 Cyg A | 22-48 | 8.4 | 2 |
| eps Ind | 18-44 | 11.4 | 2 |
| 36 Oph B | 11-37 | 30.2 | 2 |
| 36 Oph A | 11-37 | 32.1 | 2 |
| ksi Boo A | 4-30 | 47.1 | 4 |
| 61 Vir | 3-29 | 107.2 | 5 |
| kap01 Cet | 1-6, 33-52 | 134.7 | 8 |
| HD 102365 | 2-27 | 126.2 | 8 |
| zet Dor | 1-6, 33-52 | 308.6 | 11 |
| lam Ser | 7-33 | 399.4 | 17 |
| lam Aur | 1-10, 38-52 | 429.1 | 17 |
| 36 UMa | 1-18, 45-52 | 365.6 | 16 |

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
