# Palm sensing digital twin

A local Streamlit application connecting SX127x LoRa link budgets, explicit palm
coordinates, gateway schedules, ESP32 CNN memory, event energy and battery life.

The current starting configuration follows the user-confirmed purchased hardware:
**ESP32-WROOM-32/ESP-32S, SX1278 at 433 MHz, one 3.7 V / 2,600 mAh LiPo per node,
MCP602E and the WANGCL analog piezo module**. The 915 MHz RFM95W/SX1276 proposal remains
a comparison preset using different radio hardware. See the
[hardware baseline and remaining unknowns](docs/HARDWARE_BASELINE.md).

## Run

Python 3.11 or later:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

Open [localhost:8501](http://localhost:8501) for the full interactive engineering
dashboard. Inputs live in the current browser session.

```sh
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/python -m compileall -q simulator app.py
git diff --check
```

The test suite includes all ten requested acceptance categories, real Streamlit
interaction checks, zero-trigger and invalid-input cases. Automated checks do not
establish RF, firmware or hardware performance.

## Use

The optional meeting view (`meeting_app.py`) has three blocks: node operation, estimated battery life, and
the cluster-head timing limit. Open **Scenario** to change area, density, head
count, or recording duration. The default `app.py` entry point opens the engineering dashboard for detailed research and model controls.

The default cloud-classification design assumes a Pololu S7V8F3 3.3 V regulator
in the final circuit. The Pololu is planned, not purchased. The default sleep
estimate adds its 100 µA enabled/no-load allowance to the converted sleep loads:
10 µA ESP32 chip deep-sleep reference, 0.2 µA SX1278 typical sleep reference,
1 µA exploratory sensor/amplifier switch-leakage allowance, and 10 µA unsourced
battery-path leakage allowance. It assumes 80% sleep-load transfer efficiency and
zero other rail loads for a custom circuit; none of these circuit-level values is
measured or guaranteed for the purchased development boards. This gives about
122.5 µA at the battery and ~160 days under the supplied active-state estimates:
180 mA for 2 s wake;
160 mA for 10 s recording; 180 mA for 5 s processing; 300 mA for at least 0.5 s
per transmission. Sleep efficiency is applied only to converted rail loads, not added again to the active
currents. Battery life is a nominal extrapolation; cell aging and self-discharge
are not modeled.

Recording scenarios of 10, 30, 60, and 120 seconds keep recording current fixed
and linearly scale processing from the 5-second-at-10-second baseline. This is a
workload assumption, not a firmware benchmark. The model holds packet size and
trigger probability fixed and does not infer detection performance. Read
[`docs/ACOUSTIC_TEST_PROTOCOL.md`](docs/ACOUSTIC_TEST_PROTOCOL.md) for the
TreeVibes exciter-to-piezo bench-replay procedure and trial CSV template. Cluster-head report capacity reserves the selected
wake, recording, and processing interval before the hourly radio window.
The default 50% scheduling allowance gives **2,000 reservations/head/hour**, versus
a 4,000-slot theoretical ceiling at SF7. Neither figure establishes reliable field
capacity.

The separate `engineering_app.py` contains the existing detailed comparison,
measurement imports, model designer, and research tools. It uses a separate
scenario and regulated-rail energy model. Node CNN inference is an optional
engineering comparison, not part of the meeting overview's cloud-classification design.

## Scientific boundaries

**The referenced Senior Design 1 PDF was not attached.** Report-specific values
come only from the supplied prompt, retained in [docs/REQUEST.md](docs/REQUEST.md).
No radio, battery, arena or inference measurement is bundled. No configuration is
marked verified feasible. “Conditionally feasible” means it passes the selected
model and targets, subject to field and deployment verification.

The default plantation has 100 ha × 100 trees/ha = 10,000 physical trees. The
density is an assumption. Increasing radio range never creates more physical trees.
One SX1276 transceiver remains one receiver: capacity uses `min(channels, receivers)`.
Parallel receiver settings assume appropriate gateway hardware. The model is raw
LoRa, not a complete LoRaWAN implementation; legal transmission limits are excluded
as requested, while hardware configuration limits remain checked.

The baseline radio uses SF7, 125 kHz, CR4/5, explicit header, CRC and eight
preamble symbols. The 214-byte airtime is **338.176 ms** and 13-byte airtime is
**46.336 ms**. At 3.3 V / 70 mA, inference is **184.8 mJ at 800 ms** and
**46.2 mJ at 200 ms**; 800 ms / 100 mA is **264 mJ**.

The meeting estimate assumes +20 dBm; the engineering comparison starts at +17 dBm.
Both are unverified settings. Antenna gains, propagation and power inputs are
planning assumptions rather than established performance of the purchased parts.
Battery quantity is inventory, not a multiplication of per-node battery capacity.

See [formulas and modeling limits](docs/MODELS.md), [measurement import schemas](docs/MEASUREMENTS.md)
and the in-app source ledger. Cost inputs use arbitrary user currency units;
their defaults are exploratory, not quotations.

## Source layout

| File | Responsibility |
|---|---|
| `simulator/config.py` | One scenario schema, defaults and presets |
| `simulator/radio.py` | Manufacturer sensitivity tables, link budget, airtime |
| `simulator/network.py` | Seeded geometry, links, receiver allocation, schedules |
| `simulator/cnn.py` | Sequential architecture, MACs, storage and memory estimates |
| `simulator/energy.py` | Shared pipeline, A/B stages, battery and infrastructure |
| `simulator/engine.py` | Validation and deterministic dependency orchestration |
| `simulator/research.py` | Sampling, sensitivity, objective eligibility, Pareto |
| `simulator/io.py` | Calibration, JSON/CSV/report provenance |
| `simulator/prototype.py` | Battery-side prototype estimates and shared network calculations |
| `simulator/prototype_view.py` | Simple default page and settings/results download |
| `app.py` | View selection and advanced dashboard |

## Measurement gaps

For an explanation of a result, open its calculation panel or search the
**Assumptions and Sources** tab. The formula guide explains why each model equation
is used and where its constants originate. The input ledger separates the supplied
hardware baseline, manufacturer reference values, scenario overrides and assumptions.
Main scenario controls and raw output tables are collapsed to keep the Overview direct.

Field RSSI and delivery observations, actual antenna installation and gateway
resources, compiled TFLM arena/operator allocation, ESP32 current waveforms,
recording-buffer strategy, battery discharge and solar yield all require validation.
The simulator does not predict classifier recall, false alarms or pest-detection
accuracy from parameter count. Those metrics require a labeled acoustic evaluation.

The concise meeting overview remains available as a separate entry point:

```sh
.venv/bin/python -m streamlit run meeting_app.py
```
