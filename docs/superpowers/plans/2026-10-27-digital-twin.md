# Digital Twin Implementation Plan

> **For agentic workers:** Use isolated file ownership and test each engine before integration.

**Goal:** Build and run the interactive RF, CNN and energy plantation simulator.

**Architecture:** Flat configuration shared by pure calculation engines. One orchestration
function recomputes RF, both schedules, memory, energy and feasibility; Streamlit reruns it.

**Tech Stack:** Python, NumPy, SciPy, pandas, Streamlit, Plotly, pytest, Ruff.

**Spec:** docs/REQUEST.md and docs/superpowers/specs/2026-10-27-digital-twin-design.md.

## Global Constraints
- Physical trees do not change with RF or inference settings.
- All predictive outputs retain their assumption/measurement labels.
- Default latency/current are independent: 800 ms, 70 mA, 3.3 V.
- Keep A and B delivery, energy and server workload separate.

## Review Focus
- Zero triggers, zero equipped nodes and zero successful delivery must remain finite.
- Reject invalid imports, unsupported radio settings and excessive allocation sizes.
- Never count overlapping gateway coverage twice or treat channels as extra receivers.
- Account for battery conversion and sleep only for remaining daily time.
- No verified-feasible classification without real deployment evidence.

### Task 1: Radio and network (deep worker)
- [x] Write and run failing airtime, population and scheduling tests.
- [x] Implement simulator/radio.py and simulator/network.py with exact airtime,
  bounded population, seeded spatial reliability and per-resource schedules.
- [x] Run tests/test_radio_network.py, including both frequency bands and SF tradeoff.

### Task 2: CNN, energy and research (main)
- [x] Write failing energy, memory, input validation and coupling tests.
- [x] Implement config.py, cnn.py, energy.py, engine.py, research.py, io.py.
- [x] Cover stated 184.8/46.2/264 mJ regressions and all ten acceptance cases.
- [x] Implement deterministic full factorial/random/LHS search and Pareto extraction.

### Task 3: Dashboard (build worker)
- [x] Implement app.py with twelve requested tabs, one config state and live results.
- [x] Add interactive charts, maps, controls, scenario/data import and exports.
- [x] Run Streamlit AppTest and smoke-launch the server.

### Task 4: Verification and delivery
- [x] Independently review tests, numerical boundaries, app interactions and exports.
- [x] Run pytest, Ruff, compileall, git diff --check and HTTP health check.
- [x] Record formulas, measured-data schemas, launch instructions and known limits.
