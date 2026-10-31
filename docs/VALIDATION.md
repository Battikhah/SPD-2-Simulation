# Software validation — 27 October 2026

## Cluster-head capacity clarification — 28 October 2026

The interface now separates physical/reachable population from admitted population,
reports load at each receiving head, and compares current trigger traffic with
every reachable node reporting on every recording opportunity. A single baseline
receiver retains its 4,000 reservations/round ceiling. The 48,000 daily reservations
do not mean 48,000 nodes can each report twelve times per day.

The update passed **71 tests** and Ruff. Regression coverage includes acreage
growth with fixed head resources, one overloaded head despite spare aggregate
capacity, dynamic average versus full-report demand, and current-scenario isolation.
The all-reporting summary is omitted from fast parameter-search evaluations;
reports remain compatible with those evaluations. ALOHA labels its capacity as a
reference rather than a guaranteed admission ceiling.

## Hardware-baseline update — 28 October 2026

The starting scenario now uses the user-reported 433 MHz SX1278 and a single
2,600 mAh / 3.7 V battery per node, with the exact twelve recording start hours
and 100–2,500 Hz bandpass metadata. An independent verifier ran **66 tests**,
Ruff and compilation successfully. Focused checks cover all presets, inventory
versus simulated population, legacy scenario/session migration, custom schedule
counts, filter validation, and distinct 433/915 MHz reference provenance.

The example scenario/report and 3,888-row sweep were regenerated with the corrected
baseline; zero sweep rows were invalid. Purchased hardware performance, exact board
circuits, regulator efficiency and recording/sleep currents remain unverified.
Static type checking remains unconfigured. Original build evidence follows.

Executed in the project Python 3.11 virtual environment:

- `python -m pytest -q`: **50 passed** (including 7 real Streamlit AppTests).
- `ruff check .`: passed.
- `python -m compileall -q simulator app.py tests`: passed.
- `python -m pip check`: no broken requirements.
- `git diff --check`: passed; new source/document files were also checked separately.
- Live Streamlit server on loopback port 8501: `/_stcore/health` returned `ok`.
- Rendered browser inspection: responsive main controls and explicit plantation map.
- A full factorial search of **3,888 designs** completed with zero invalid rows
  in **14.16 seconds** on this machine. This timing is one software run, not a
  hardware/ESP32 benchmark or a general performance guarantee.

The reproducible search definition, CSV output and default scenario/report are in
`examples/`. The source code is not an embedded firmware implementation.

| Requested acceptance case | Checked behavior |
|---|---|
| SF dependency | Sensitivity, airtime, range, coverage, served count and capacity change |
| Frequency dependency | Module and propagation recalculate; current does not scale arbitrarily |
| Latency 800→200 ms | 184.8→46.2 mJ at 3.3 V / 70 mA |
| Current 70→100 mA | 184.8→264 mJ at 800 ms / 3.3 V |
| Airtime | 214 bytes = 338.176 ms; 13 bytes = 46.336 ms |
| TDMA conflict | 8,000 × 900 ms = 7,200 s, exceeding one hour on one receiver |
| Fixed geography | Reduced reach reduces linked positions without creating/deleting trees |
| SF capacity tradeoff | Higher SF can reach more nodes and schedule fewer |
| CNN memory | Excess arena/contiguous demand is rejected as infeasible |
| Consistency | CNN current changes energy, not RF; density changes population, not propagation |

Additional regressions cover malformed imports, zero triggers, zero packet delivery,
server-model storage, retries without ACK, a packet longer than its reporting window,
expected daily traffic versus one seeded event round, unique trees with multiple
sensors, integer parameter sweeps, constant tornado plots, scenario reload, measured
profile invalidation, and custom inference values outside the slider range.

An independent read-only verifier checked numerical invariants, unit conversion,
Pareto extraction against exhaustive dominance, synthetic RSSI regression, the live
AppTest sweep/preset flow and the repaired cases. Synthetic regression fixtures are
not presented as field measurements.

Not checked: static type analysis (no checker configured), compiled firmware, TFLM
operator allocation, physical antennas/RF, classifier accuracy, real battery behavior,
or the missing Senior Design PDF. See `MODELS.md` for conditional model assumptions.


## Formula and presentation clarification — 29 October 2026

The full suite passed 73 tests before final explanation-only corrections. The
final formula/provenance and Streamlit AppTest checks passed all 14 tests; Ruff
and compilation passed. All 48 baseline summary values remained identical. The
input ledger covers all 130 scenario fields; the 14 formula topics were reviewed
against the numerical implementation. Live browser checks verified formula search,
expanded equations/rationale/constants, and the simplified Overview on port 8502
(port 8501 was occupied by an unrelated app).

No physical battery, RF, or firmware measurements were made. The source ledger
distinguishes manufacturer references, user-supplied baseline, numerical conventions
and unmeasured defaults; this clarification does not calibrate the simulator.
