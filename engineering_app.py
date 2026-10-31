"""Interactive engineering dashboard for the palm-weevil LoRa digital twin."""

from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from simulator.config import DEFAULTS, OPTIONS, PRESETS
from simulator.explanations import FORMULA_GUIDE, input_provenance
from simulator.hardware import HARDWARE_BASELINE

st.set_page_config(page_title="Palm LoRa Digital Twin", page_icon="🌴", layout="wide")
st.title("🌴 Palm Weevil Detection · System Digital Twin")
st.caption("Calculated and modeled results are estimates unless explicitly marked measured or manufacturer specified.")

if "cfg" not in st.session_state:
    st.session_state.cfg = deepcopy(DEFAULTS)
cfg = st.session_state.cfg
if "recording_hours" not in cfg and cfg.get("recordings_per_day", 12) != len(DEFAULTS["recording_hours"]):
    cfg["recording_hours"] = []
for field, default in DEFAULTS.items():
    cfg.setdefault(field, deepcopy(default))
rendered_fields = set()


def sync_widget_state():
    for key, value in cfg.items():
        if key in st.session_state:
            st.session_state[key] = value


def apply_manual_performance(field, widget):
    cfg[field] = st.session_state[widget]
    cfg["current_profile"] = []
    cfg["performance_source"] = "Exploratory assumption"
    st.session_state[field] = cfg[field]
    st.session_state["performance_source"] = cfg["performance_source"]


def apply_scenario_upload():
    try:
        from simulator.io import loads_scenario

        upload = st.session_state.get("scenario_upload")
        loaded = loads_scenario(upload.getvalue().decode("utf-8"))
        cfg.update(DEFAULTS | loaded)
        sync_widget_state()
        st.session_state.custom_tree_text = "\n".join(f"{x},{y}" for x, y in cfg["custom_trees"])
        st.session_state.gateway_text = "\n".join(f"{x},{y}" for x, y in cfg["gateway_positions"])
        st.session_state.layers_json = json.dumps(cfg["layers"], indent=2)
        st.session_state.input_shape_text = ",".join(map(str, cfg["input_shape"]))
        st.session_state.scenario_error = ""
    except (ValueError, UnicodeDecodeError) as exc:
        st.session_state.scenario_error = str(exc)


def apply_measurement_upload():
    try:
        from simulator.io import import_measurements

        upload = st.session_state.get("measurement_upload")
        kind = st.session_state.get("measurement_kind", "Inference")
        cfg.update(import_measurements(cfg, kind, pd.read_csv(upload)))
        sync_widget_state()
        st.session_state.measurement_error = ""
    except (ValueError, KeyError) as exc:
        st.session_state.measurement_error = str(exc)


def edit(key, label=None, *, help=None, minimum=None, maximum=None, step=None, slider=False):
    """Render a control bound directly to the canonical session configuration."""
    if key in rendered_fields:
        return
    rendered_fields.add(key)
    name = label or key.replace("_", " ").title()
    if key in OPTIONS:
        opts = OPTIONS[key]
        current = cfg.get(key, DEFAULTS[key])
        index = opts.index(current) if current in opts else 0
        cfg[key] = st.selectbox(name, opts, index=index, key=key, help=help)
    elif isinstance(DEFAULTS.get(key), bool):
        cfg[key] = st.checkbox(name, value=bool(cfg.get(key)), key=key, help=help)
    elif isinstance(DEFAULTS.get(key), (int, float)):
        val = cfg.get(key, DEFAULTS[key])
        low = (
            minimum
            if minimum is not None
            else (-30 if key in {"tx_gain_dbi", "rx_gain_dbi"} else (0 if val >= 0 else -100))
        )
        high = maximum if maximum is not None else max(float(val) * 4, low + 100)
        kwargs = {
            "min_value": type(val)(low),
            "max_value": type(val)(high),
            "value": type(val)(val),
            "key": key,
            "help": help,
        }
        if step is not None:
            kwargs["step"] = type(val)(step)
        if slider:
            new_value = st.slider(name, **kwargs)
        else:
            new_value = st.number_input(name, **kwargs)
        if key in {"inference_latency_ms", "inference_current_ma"} and new_value != val:
            cfg["current_profile"] = []
            cfg["performance_source"] = "Exploratory assumption"
            st.session_state["performance_source"] = "Exploratory assumption"
        if key == "recordings_per_day" and new_value != val:
            cfg["recording_hours"] = []
        cfg[key] = new_value


def section(keys, title=None):
    if title:
        st.markdown(f"**{title}**")
    for key in keys:
        if key in DEFAULTS:
            edit(key)


def calculation_help(topic):
    guide = next(item for item in FORMULA_GUIDE if item["Topic"] == topic)
    with st.expander(f"How {topic.lower()} is calculated"):
        st.write("**Formula**")
        st.code(guide["Formula"].replace("; ", ";\n").replace(". ", ".\n"), language=None, wrap_lines=True)
        st.write("**Why this formula**")
        st.write(guide["Why it is used"])
        st.write("**Constants and evidence**")
        st.write(guide["Constants and source"])


with st.sidebar:
    st.header("Scenario")
    mode = st.radio("Mode", ["Beginner", "Engineering", "Research"], horizontal=True)
    st.caption(
        {
            "Beginner": "Start with the nine live controls; the other tabs show how the estimates are formed.",
            "Engineering": "Detailed RF, scheduling, memory, and battery inputs use SI units shown beside each field.",
            "Research": "Set a seed, document evidence, and run explicit parameter sweeps for reproducible comparisons.",
        }[mode]
    )
    preset = st.selectbox("Load preset", list(PRESETS))
    if st.button("Apply preset", width="stretch"):
        cfg.update(DEFAULTS | PRESETS[preset])
        sync_widget_state()
        st.session_state.custom_tree_text = ""
        st.session_state.gateway_text = ""
        st.session_state.layers_json = json.dumps(cfg["layers"], indent=2)
        st.session_state.input_shape_text = ",".join(map(str, cfg["input_shape"]))
        st.rerun()
    st.caption("Preset changes are applied to this session only.")
    if mode != "Beginner":
        st.divider()
        st.subheader("Scenario file")
        scenario_upload = st.file_uploader("Load scenario JSON", type="json", key="scenario_upload")
        if scenario_upload is not None:
            st.button("Load uploaded scenario", on_click=apply_scenario_upload)
        if st.session_state.get("scenario_error"):
            st.error(f"Could not load scenario: {st.session_state.scenario_error}")
        if st.button("Download current scenario", width="stretch"):
            from simulator.io import dumps_scenario

            st.download_button("Save scenario JSON", dumps_scenario(cfg), "scenario.json", "application/json")

with st.expander("Change scenario inputs", expanded=False):
    st.caption("These nine controls update all results below. Detailed settings are in the tabs.")
    priority_rows = [st.columns(3) for _ in range(3)]
    with priority_rows[0][0]:
        edit("frequency_mhz", "Frequency", help="Carrier frequency in MHz.")
    with priority_rows[0][1]:
        edit("sf", "Spreading factor", minimum=6, maximum=12, step=1)
    with priority_rows[0][2]:
        edit("tx_dbm", "TX power (dBm)", minimum=2, maximum=20, step=1, slider=True)
    with priority_rows[1][0]:
        edit("inference_latency_ms", "CNN latency (ms)", minimum=1, maximum=max(10000, cfg["inference_latency_ms"]), step=1, slider=True)
    with priority_rows[1][1]:
        edit("inference_current_ma", "CNN current (mA)", minimum=1, maximum=max(500, cfg["inference_current_ma"]), step=1, slider=True)
    with priority_rows[1][2]:
        edit("area_ha", "Plantation (ha)", minimum=0.1, maximum=max(1000, cfg["area_ha"]), step=0.1, slider=True)
    with priority_rows[2][0]:
        edit("trees_per_ha", "Trees / ha", minimum=1, maximum=1000, step=1, slider=True)
    with priority_rows[2][1]:
        edit("gateways", "Gateways", minimum=1, maximum=20, step=1, slider=True)
    with priority_rows[2][2]:
        edit("trigger_probability", "SNR trigger probability", minimum=0, maximum=1, step=0.05, slider=True)

tabs = st.tabs(
    [
        "Overview",
        "Radio Settings",
        "Plantation Map",
        "Gateway and Network Capacity",
        "ESP32 Settings",
        "CNN Model Designer",
        "Architecture Comparison",
        "Battery Simulator",
        "Sensitivity Analysis",
        "Optimization Laboratory",
        "Assumptions and Sources",
        "Export Results",
    ]
)

# Controls are all rendered before simulation, so every result uses this rerun's values.
with tabs[1]:
    st.subheader("Radio and propagation")
    calculation_help("Radio range")
    calculation_help("LoRa airtime")
    cols = st.columns(3)
    with cols[0]:
        section(["bandwidth_khz", "coding_rate", "preamble_symbols", "implicit_header", "crc", "ldro"], "Modem")
        section(["payload_a_bytes", "payload_b_bytes", "metadata_bytes", "protocol_bytes"], "Payload")
    with cols[1]:
        section(
            [
                "propagation",
                "path_loss_exponent",
                "reference_distance_m",
                "vegetation_loss_db",
                "shadowing_db",
                "obstruction_loss_db",
                "obstruction_probability",
                "interference_loss_db",
            ],
            "Channel model",
        )
        section(
            [
                "tx_gain_dbi",
                "rx_gain_dbi",
                "tx_height_m",
                "rx_height_m",
                "cable_loss_db",
                "misc_loss_db",
                "required_margin_db",
            ],
            "Link budget",
        )
    with cols[2]:
        section(
            ["tx_current_override_ma", "radio_standby_ma", "radio_sleep_ma", "radio_rx_ma", "mcu_tx_ma"],
            "Radio current",
        )
        if "distance_m" in cfg:
            edit("distance_m")
    distance_check_m = st.number_input(
        "Distance for path-loss check (m)", min_value=1.0, value=1000.0, key="distance_check_m"
    )

with tabs[2]:
    st.subheader("Plantation geometry")
    calculation_help("Plantation and coverage")
    cols = st.columns(3)
    with cols[0]:
        section(["area_ha", "width_m", "length_m", "dimensions_override"], "Dimensions")
    with cols[1]:
        section(
            ["trees_per_ha", "tree_spacing_m", "layout", "nodes_per_tree", "equipped_fraction", "deployed_limit"],
            "Tree population",
        )
    with cols[2]:
        section(["gateways", "channels", "receivers"], "Gateway layout")
    gateway_text = st.text_area(
        "Optional gateway positions (CSV: x,y per row)", help="Leave blank for automatic placement.", key="gateway_text"
    )
    if gateway_text.strip():
        try:
            cfg["gateway_positions"] = pd.read_csv(pd.io.common.StringIO("x,y\n" + gateway_text)).values.tolist()
        except (ValueError, pd.errors.ParserError):
            st.warning("Enter one numeric x,y coordinate pair per line.")
    elif cfg["gateway_positions"] and st.button("Use automatic gateway placement"):
        cfg["gateway_positions"] = []
    if cfg["gateway_positions"] and len(cfg["gateway_positions"]) != cfg["gateways"]:
        st.warning("Custom gateway position count must match the selected gateway count.")
    coords = st.text_area(
        "Custom tree positions (CSV: x,y per row)", help="Used only with Custom layout.", key="custom_tree_text"
    )
    if coords.strip():
        try:
            cfg["custom_trees"] = pd.read_csv(pd.io.common.StringIO("x,y\n" + coords)).values.tolist()
        except (ValueError, pd.errors.ParserError):
            st.warning("Enter one numeric x,y coordinate pair per line.")

with tabs[3]:
    st.subheader("Network scheduling")
    calculation_help("Reporting capacity")
    calculation_help("Delivery probability")
    cols = st.columns(3)
    with cols[0]:
        section(["scheduling", "slot_ms", "guard_ms", "clock_error_ms", "window_s"], "Schedule")
    with cols[1]:
        section(
            ["retries", "ack", "ack_bytes", "ack_wait_ms", "packet_success", "min_reliability", "delivery_target"],
            "Reliability",
        )
    with cols[2]:
        section(["trigger_probability", "snr_gate_db", "seed"], "Event gating")

with tabs[4]:
    st.subheader("ESP32 execution")
    calculation_help("Recording and SNR gate")
    calculation_help("Inference timing and energy")
    st.caption("Enter a precise value or a custom value beyond the main slider range. These overrides are exploratory.")
    for field, widget, label in [
        ("inference_latency_ms", "custom_latency_ms", "CNN Inference Latency (ms)"),
        ("inference_current_ma", "custom_current_ma", "CNN Inference Current (mA)"),
    ]:
        st.session_state[widget] = float(cfg[field])
        st.number_input(
            label,
            min_value=1.0,
            value=float(cfg[field]),
            key=widget,
            on_change=apply_manual_performance,
            args=(field, widget),
        )
    cols = st.columns(3)
    with cols[0]:
        section(["cpu_mhz", "cpu_utilization", "cycles_per_mac", "backend"], "Compute")
    with cols[1]:
        section(
            [
                "inference_latency_ms",
                "inference_current_ma",
                "preprocessing_ms",
                "preprocessing_current_ma",
                "performance_source",
            ],
            "Measured or estimated performance",
        )
    with cols[2]:
        section(
            [
                "recording_s",
                "sample_rate_hz",
                "filter_low_hz",
                "filter_high_hz",
                "recording_current_ma",
                "wake_ms",
                "wake_current_ma",
                "afe_ms",
                "afe_current_ma",
                "mfcc_ms",
                "mfcc_current_ma",
                "radio_startup_ms",
                "sleep_current_ma",
                "voltage_v",
            ],
            "Sensing and voltage",
        )
    cfg["measurement_note"] = st.text_input(
        "Measurement/source note", value=cfg.get("measurement_note", ""), key="measurement_note"
    )
    if cfg["performance_source"] == "Published benchmark":
        st.info(
            "Add a citation in the source note. This app does not transfer published results to this model unless the scenario inputs match."
        )
    if cfg.get("current_profile"):
        st.caption(
            "An imported measured current profile replaces the inference latency/current inputs until either live control is edited."
        )
    measurement = st.file_uploader("Import measured CSV", type="csv", key="measurement_upload")
    if measurement:
        st.selectbox(
            "Measurement kind",
            ["Inference", "Current profile", "RSSI", "Packet success", "Trees", "Gateways"],
            key="measurement_kind",
        )
        st.button("Import measurements", on_click=apply_measurement_upload)
        if st.session_state.get("measurement_error"):
            st.error(f"Measurement import failed: {st.session_state.measurement_error}")

with tabs[5]:
    st.subheader("CNN model and memory")
    calculation_help("CNN workload and memory")
    workload_preset = st.selectbox(
        "Quick workload parameter count",
        [50_000, 100_000, 250_000, 350_000, 500_000, 1_000_000],
        format_func=lambda value: f"{value:,} parameters",
        key="workload_preset",
    )
    if st.button("Apply workload preset", key="apply_workload_preset"):
        cfg["model_mode"] = "Workload"
        cfg["parameters"] = workload_preset
        st.session_state["model_mode"] = "Workload"
        st.session_state["parameters"] = workload_preset
        st.rerun()
    cols = st.columns(3)
    with cols[0]:
        section(["model_mode", "parameters", "macs", "precision", "input_shape"], "Workload")
    with cols[1]:
        section(
            [
                "flash_kb",
                "firmware_flash_kb",
                "dram_kb",
                "firmware_ram_kb",
                "stack_kb",
                "audio_buffer_kb",
                "dma_buffer_kb",
            ],
            "ESP32 memory",
        )
    with cols[2]:
        section(
            ["tensor_arena_kb", "scratch_kb", "contiguous_kb", "activation_kb", "operators_supported", "deadline_ms"],
            "Model feasibility",
        )
    st.session_state.setdefault("layers_json", json.dumps(cfg.get("layers", []), indent=2))
    layers_text = st.text_area(
        "CNN layer designer (JSON)",
        key="layers_json",
        help="Optional ordered layer definitions consumed by the CNN engine.",
    )
    try:
        layers = json.loads(layers_text)
        if not isinstance(layers, list):
            raise TypeError("Expected a JSON array of layer definitions.")
        cfg["layers"] = layers
    except (json.JSONDecodeError, TypeError) as exc:
        st.warning(f"Layer configuration not applied: {exc}")
    st.session_state.setdefault("input_shape_text", ",".join(map(str, cfg.get("input_shape", []))))
    shape = st.text_input("Input shape (comma-separated)", key="input_shape_text")
    try:
        parsed_shape = [int(part.strip()) for part in shape.split(",") if part.strip()]
        if len(parsed_shape) == 3:
            cfg["input_shape"] = parsed_shape
        else:
            st.caption("Enter three dimensions to apply the input shape.")
    except ValueError:
        st.caption("Input shape changes apply when all three integer dimensions are entered.")
    st.caption("The reported arena and latency are scenario inputs unless a measured or published source is provided.")
    st.caption("The bandpass is applied to recorded samples before the SNR gate; verify its limits against the final analog circuit and sampling implementation.")

with tabs[7]:
    st.subheader("Battery and deployment assumptions")
    calculation_help("Daily energy and battery life")
    calculation_help("Deployment totals and optimization")
    cols = st.columns(3)
    with cols[0]:
        section(
            [
                "recordings_per_day",
                "battery_v",
                "battery_mah",
                "regulator_efficiency",
                "quiescent_ma",
                "usable_fraction",
            ],
            "Node battery",
        )
        hours = cfg["recording_hours"]
        if hours:
            if hours == DEFAULTS["recording_hours"]:
                night = [h for h in hours if h >= 22 or h < 6]
                day = [h for h in hours if 6 <= h < 22]
                st.caption(
                    f"Configured local schedule (Asia/Dubai): nighttime 22:00–06:00 hourly ({len(night)} rounds at "
                    f"{', '.join(f'{h:02}:00' for h in night)}); daytime 06:00–22:00 every 4 h "
                    f"({len(day)} rounds at {', '.join(f'{h:02}:00' for h in day)})."
                )
                st.caption("The energy model counts 12 rounds/day; packet times depend on the scheduler. The hourly night rounds leave no timing slack if one round's active work exceeds an hour.")
            else:
                st.caption(f"Configured local schedule (Asia/Dubai): {len(hours)} recording start hours/day at {', '.join(f'{h:02}:00' for h in hours)}. Packet times depend on the scheduler.")
        else:
            st.caption("Custom rounds/day is set, but exact recording timestamps are unknown; the energy model counts rounds without assigning clock times.")
    with cols[1]:
        section(
            ["temperature_factor", "aging_factor", "self_discharge_monthly", "solar_wh_day", "battery_target_days"],
            "Lifetime adjustments",
        )
    with cols[2]:
        section(
            [
                "gateway_idle_w",
                "gateway_process_w",
                "backhaul_j_packet",
                "server_idle_w",
                "server_inference_w",
                "server_inference_ms",
                "server_model_capacity_mb",
                "node_cost",
                "gateway_cost",
                "electricity_cost_kwh",
            ],
            "Infrastructure",
        )
    if cfg.get("current_profile"):
        st.dataframe(pd.DataFrame(cfg["current_profile"]), width="stretch")

with tabs[8]:
    st.subheader("Sensitivity analysis")
    st.caption("Choose a parameter and inspect how the connected engine responds while all other settings stay fixed.")
    sensitivity_parameter = st.selectbox(
        "Parameter",
        [
            "sf",
            "frequency_mhz",
            "tx_dbm",
            "inference_latency_ms",
            "inference_current_ma",
            "area_ha",
            "trees_per_ha",
            "gateways",
            "trigger_probability",
        ],
        key="sensitivity_parameter",
    )
    st.caption("Use the Optimization Laboratory for configurable multidimensional sweeps.")

with tabs[9]:
    st.subheader("Optimization laboratory")
    sweep_columns = st.columns(3)
    numeric_fields = [
        key for key, value in DEFAULTS.items() if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    with sweep_columns[0]:
        sweep_parameters = st.multiselect("Sweep parameters", numeric_fields, default=["sf"])
    with sweep_columns[1]:
        method = st.selectbox("Method", ["Full factorial", "Random", "Latin hypercube", "Monte Carlo"])
    with sweep_columns[2]:
        samples = st.number_input("Samples", min_value=10, max_value=20000, value=1000, step=100)
    objective = st.selectbox(
        "Optimization objective",
        [
            "Max monitored trees under targets",
            "Max monitored nodes under targets",
            "Max battery life",
            "Min node energy",
            "Min deployment energy",
            "Min gateways",
            "Max coverage",
            "Min airtime",
            "Max CNN parameters",
            "Min energy per monitored tree",
        ],
    )
    objective_arch = st.selectbox("Architecture objective", ["A", "B"], index=1)
    sweep_values = {}
    for key in sweep_parameters:
        common_values = {
            "sf": "7,9,12",
            "frequency_mhz": "433,915",
            "tx_dbm": "8,14,17,20",
            "inference_latency_ms": "200,800,2000",
            "inference_current_ma": "30,70,150",
            "area_ha": "10,100,1000",
            "trees_per_ha": "50,100,200",
            "gateways": "1,2,4",
            "trigger_probability": "0.1,0.25,0.5,1",
        }
        current = cfg[key]
        default_values = common_values.get(key, f"{current},{current * 2 if current else 1}")
        raw = st.text_input(f"{key} values (comma separated)", default_values, key=f"sweep_{key}")
        try:
            values = [float(v.strip()) for v in raw.split(",") if v.strip()]
            if isinstance(DEFAULTS[key], int):
                if any(not v.is_integer() for v in values):
                    raise ValueError("Integer field requires whole numbers.")
                values = [int(v) for v in values]
            sweep_values[key] = values
        except ValueError:
            st.error(f"Invalid numeric values for {key}.")
    run_sweep = st.button("Run sweep", type="primary")

with tabs[11]:
    st.subheader("Import, report, and export")
    st.caption("The current scenario is recalculated automatically on each input change.")

try:
    from simulator.engine import simulate

    result = simulate(cfg, detail=True)
except ImportError as exc:
    st.error(f"Simulation engine is not available yet: {exc}")
    st.stop()
except ValueError as exc:
    st.error(f"Invalid configuration: {exc}")
    st.stop()

# Overview metrics are deliberately drawn after every tab has collected its controls.
with tabs[0]:
    s = result["summary"]
    rounds = cfg["recordings_per_day"]
    expected_reports = rounds * cfg["trigger_probability"]
    max_copies = cfg["retries"] + 1
    network_a, network_b = result["network_a"], result["network_b"]
    radio = result["radio"]
    st.subheader("How each sensor node operates")
    st.markdown(
        f"**Nodes transmit intermittently, not continuously.** Each deployed node wakes for {rounds} opportunities/day. "
        f"It records {cfg['recording_s']:g} s at {cfg['sample_rate_hz']:,} Hz, filters {cfg['filter_low_hz']:g}–{cfg['filter_high_hz']:g} Hz, then applies the {cfg['snr_gate_db']:g} dB SNR gate. "
        f"At the assumed {cfg['trigger_probability']:.0%} trigger rate, that is **{expected_reports:.1f} expected initial reports/day per admitted node**. "
        "A rejected recording ends before MFCC, CNN, and radio transmission."
    )
    st.markdown(
        f"- **A · remote CNN:** sends {cfg['payload_a_bytes'] + cfg['metadata_bytes'] + cfg['protocol_bytes']} bytes for server processing.\n"
        f"- **B · local CNN:** processes on the ESP32 and sends {cfg['payload_b_bytes'] + cfg['metadata_bytes'] + cfg['protocol_bytes']} bytes."
    )
    calculation_help("Recording and SNR gate")
    copies_a = network_a["attempts_per_event"]
    copies_b = network_b["attempts_per_event"]
    operating_rows = []
    for label, net, copies in [("A · remote CNN", network_a, copies_a), ("B · local CNN", network_b, copies_b)]:
        attempts_day = expected_reports * copies
        tx_s = attempts_day * radio[f"airtime_{label[0].lower()}_ms"] / 1000
        operating_rows.append(
            {
                "Architecture": label,
                "Expected TX airtime / admitted node / day": f"{tx_s:.3f} s",
                "Modeled active / sleep time per day": f"{result['energy'][label[0].lower()]['active_s_day'] / 3600:.2f} h / {result['energy'][label[0].lower()]['sleep_s_day'] / 3600:.2f} h",
            }
        )
    st.dataframe(pd.DataFrame(operating_rows), hide_index=True, width="stretch")
    retry_note = (
        f"At most {max_copies} copies/report (retries={cfg['retries']}); "
        + ("ACKs can stop retries after success." if cfg["ack"] else "ACKs are off, so each triggered report uses the configured copies.")
    )
    st.caption(
        "These are averages/expected values, not guaranteed transmissions at fixed times. "
        + ("Recording start hours follow the configured local schedule; packet times depend on the scheduler. " if cfg["recording_hours"] else f"{rounds} opportunities/day has no configured timestamps. ")
        + f"{retry_note} Capacity-rejected nodes and link failures are separate from the SNR gate. "
        + "Fixed TDMA reserves a slot even when there is no event; an empty reservation is not a transmission."
    )
    st.subheader("Cluster-head reporting capacity")
    capacity_kind = "scheduled capacity" if network_a["gateway_load"].capacity_is_hard_limit.all() else "probabilistic ALOHA reference capacity"
    st.write(
        f"A modeled head is one gateway site. Current {capacity_kind}: A {network_a['capacity_nodes'] // max(cfg['gateways'], 1):,} reports/head/round; "
        f"B capacity is {network_b['capacity_nodes'] // max(cfg['gateways'], 1):,}. "
        f"The round lasts {cfg['window_s']:g} s, with {cfg['channels']} channel(s), {cfg['receivers']} receiver(s), and a {network_a['slot_s']:.3f} s A slot."
    )
    calculation_help("Reporting capacity")
    head_rows = []
    for label, current, worst in [
        ("A · remote CNN", network_a, result["worst_case"]["network_a"]),
        ("B · local CNN", network_b, result["worst_case"]["network_b"]),
    ]:
        head_rows.append(
            {
                "Architecture": label,
                "Reachable nodes": current["reachable_nodes"],
                "Outside coverage": current["outside_nodes"],
                "Current admitted population": f"{current['admitted_nodes']:,.1f}",
                "Worst-case admitted population": f"{worst['admitted_nodes']:,.1f}",
                "Worst-case rejected nodes/round": f"{worst['capacity_rejected']:,}",
            }
        )
    st.dataframe(pd.DataFrame(head_rows), hide_index=True, width="stretch")
    st.caption("Worst case: every reachable node requests a report at every opportunity. The daily total is report slots, not nodes; it does not mean one report per node per day. Nodes outside coverage are separate. This model does not add relay hops or head buffers.")
    nominal_wh = cfg["battery_v"] * cfg["battery_mah"] / 1000
    st.caption(
        f"Node battery currently modeled: **{cfg['battery_v']:g} V × {cfg['battery_mah']:g} mAh = {nominal_wh:.2f} Wh nominal**; "
        f"regulated sensor rail: **{cfg['voltage_v']:g} V**. Battery is per sensor node; gateway/server power is modeled separately."
    )
    with st.expander("Battery assumptions and source"):
        st.write(
            f"Current scenario: usable fraction {cfg['usable_fraction']:.0%}, aging factor {cfg['aging_factor']:.0%}, "
            f"temperature factor {cfg['temperature_factor']:.0%}, regulator efficiency {cfg['regulator_efficiency']:.0%}, "
            f"quiescent current {cfg['quiescent_ma']:g} mA, monthly self-discharge {cfg['self_discharge_monthly']:.1%}, "
            f"solar {cfg['solar_wh_day']:g} Wh/day. The 3.7 V / 2600 mAh per-node value is the user-reported 505573 baseline. "
            "Three purchased batteries do not multiply each node's capacity; wiring and regulator efficiency are unknown. "
            f"The configured {cfg['recording_current_ma']:g} mA recording current represents the whole active recording chain, not just the piezo or amplifier. "
            f"TX power {cfg['tx_dbm']:g} dBm, antenna gains {cfg['tx_gain_dbi']:g}/{cfg['rx_gain_dbi']:g} dBi, regulator efficiency "
            f"{cfg['regulator_efficiency']:.0%}, ESP32 sleep {cfg['sleep_current_ma']:g} mA, and converter quiescent "
            f"{cfg['quiescent_ma']:g} mA are editable exploratory placeholders, not measurements of the purchased circuit."
        )
    with st.expander("Purchased hardware and unresolved details"):
        st.caption("Inventory below is user-reported. Purchased quantities are not deployment limits; batteries are not assumed to be paralleled.")
        st.dataframe(pd.DataFrame([{"Part": part, **details} for part, details in HARDWARE_BASELINE.items() if isinstance(details, dict)]), hide_index=True, width="stretch")
        st.markdown("**Still unknown:** " + "; ".join(HARDWARE_BASELINE["unknowns"]))
        if cfg["frequency_mhz"] == 915:
            st.warning("915 MHz is retained for comparison with the original SX1276/RFM95W report. The purchased SX1278 is the 433 MHz baseline and cannot operate at 915 MHz.")
        st.caption("The MCP602E was purchased; the report specified MCP6002. Final schematic, gain, wiring, and actual current are not established.")
    summary_metrics = [
        ("Reachable nodes", "reachable_nodes", "nodes"),
        ("Served · A", "served_a", "nodes"),
        ("Served · B", "served_b", "nodes"),
        ("Battery · A", "battery_a_days", "days"),
        ("Battery · B", "battery_b_days", "days"),
    ]
    for offset in range(0, len(summary_metrics), 3):
        cards = st.columns(3)
        for col, (label, key, unit) in zip(cards, summary_metrics[offset : offset + 3]):
            col.metric(label, f"{s.get(key, 0):,.1f} {unit}")
    status = result["feasibility"]["status"]
    (st.success if status.lower().startswith(("feasible", "verified")) else st.warning)(f"Feasibility: {status}")
    if result["feasibility"].get("issues"):
        for issue in result["feasibility"]["issues"]:
            st.warning(issue)
    energy = result["energy"]
    st.plotly_chart(
        px.bar(
            x=["Architecture A", "Architecture B"],
            y=[energy["a"]["daily_j"], energy["b"]["daily_j"]],
            labels={"x": "Architecture", "y": "Node energy (J/day)"},
            title="Daily node energy",
        ),
        width="stretch",
    )
    with st.expander("All calculated outputs"):
        st.dataframe(pd.DataFrame([s]), width="stretch")
    with st.expander("Dependency inspector · what changed and why"):
        previous = st.session_state.get("previous_result_snapshot")
        if previous:
            changed_inputs = [key for key in cfg if cfg[key] != previous["config"].get(key)]
            changed_outputs = [
                {"Output": key, "Before": previous["summary"].get(key), "Now": value}
                for key, value in s.items()
                if value != previous["summary"].get(key)
            ]
            st.write("Changed inputs:", ", ".join(changed_inputs) or "None since the previous rerun")
            if changed_outputs:
                st.dataframe(pd.DataFrame(changed_outputs), hide_index=True, width="stretch")
        st.code(
            "RF → sensitivity / airtime → geographic links → scheduling → reports → node and server energy\nCNN → storage / arena / independent latency and current → inference energy → battery / break-even\nArea and density → fixed physical trees → deployed population; CNN inputs do not change RF"
        )
    st.session_state.previous_result_snapshot = {"config": deepcopy(cfg), "summary": deepcopy(s)}

with tabs[1]:
    radio = result["radio"]
    cols = st.columns(3)
    cols[0].metric("Modeled range", f"{radio['range_m']:,.0f} m")
    cols[1].metric("Receiver sensitivity", f"{radio['sensitivity_dbm']:.0f} dBm")
    cols[2].metric("Selected module", radio["module"])
    with st.expander("All radio outputs"):
        st.json(radio)
    st.plotly_chart(
        px.bar(
            x=["A", "B"],
            y=[radio["airtime_a_ms"], radio["airtime_b_ms"]],
            labels={"x": "Payload", "y": "Airtime (ms)"},
            title="Packet airtime",
        ),
        width="stretch",
    )
    st.metric("Airtime · payload A", f"{radio['airtime_a_ms']:.3f} ms")
    st.metric("Airtime · payload B", f"{radio['airtime_b_ms']:.3f} ms")
    from simulator.radio import airtime_ms, link_budget_db, path_loss_db

    distance_axis = np.geomspace(1, max(distance_check_m * 2, 2), 80)
    margin_axis = link_budget_db(cfg) - path_loss_db(cfg, distance_axis)
    st.plotly_chart(
        px.line(
            x=distance_axis,
            y=margin_axis,
            labels={"x": "Distance (m)", "y": "Modeled link margin (dB)"},
            title="Distance versus link margin",
        ),
        width="stretch",
    )
    frequencies = [433, 915]
    frequency_losses = [
        float(path_loss_db(cfg | {"frequency_mhz": frequency, "propagation": "Free-space"}, distance_check_m))
        for frequency in frequencies
    ]
    st.plotly_chart(
        px.bar(
            x=frequencies,
            y=frequency_losses,
            labels={"x": "Frequency (MHz)", "y": "Free-space path loss (dB)"},
            title=f"Frequency versus path loss at {distance_check_m:g} m",
        ),
        width="stretch",
    )
    payload_limit = 255 - cfg["metadata_bytes"] - cfg["protocol_bytes"]
    payload_rows = [
        {
            "Payload (bytes)": size,
            "Airtime (ms)": airtime_ms(
                size,
                sf=cfg["sf"],
                bandwidth_khz=cfg["bandwidth_khz"],
                coding_rate=cfg["coding_rate"],
                preamble_symbols=cfg["preamble_symbols"],
                implicit_header=cfg["implicit_header"],
                crc=cfg["crc"],
                ldro=cfg["ldro"],
            ),
        }
        for size in range(1, payload_limit + 1)
    ]
    st.plotly_chart(
        px.line(pd.DataFrame(payload_rows), x="Payload (bytes)", y="Airtime (ms)", title="Payload size versus airtime"),
        width="stretch",
    )
    sf_rows = []
    for sf in [7, 9, 12]:
        test_cfg = cfg | {"sf": sf}
        comparison = simulate(test_cfg, detail=False)
        rs, ns, es = comparison["radio"], comparison["network_a"], comparison["energy"]
        sf_rows.append(
            {
                "SF": sf,
                "Sensitivity (dBm)": rs["sensitivity_dbm"],
                "Range (m)": rs["range_m"],
                "Bounded area (ha)": ns["bounded_coverage_ha"],
                "Long-run reachable": ns["long_run_reachable_nodes"],
                "Reachable nodes": ns["reachable_nodes"],
                "Capacity": ns["capacity_nodes"],
                "Admitted candidates": ns["admitted_nodes"],
                "Expected monitored": ns["served_nodes"],
                "Expected monitored trees": ns["served_trees"],
                "Required gateways": ns["required_gateways"],
                "Airtime A (ms)": rs["airtime_a_ms"],
                "TX energy A (mJ)": es["tx_a_mj"],
                "Battery A (days)": es["a"]["battery_days"],
                "Deployment A (kWh/day)": es["deployment_a"]["total_kwh_day"],
            }
        )
    sf_df = pd.DataFrame(sf_rows)
    sf_metric = st.selectbox("SF comparison metric", [col for col in sf_df if col != "SF"], key="sf_comparison_metric")
    st.plotly_chart(
        px.line(sf_df, x="SF", y=sf_metric, markers=True, title=f"SF comparison · {sf_metric}"),
        width="stretch",
    )
    st.dataframe(sf_df, hide_index=True, width="stretch")

with tabs[2]:
    map_arch = st.radio("Map architecture", ["A", "B"], horizontal=True, key="map_arch")
    coverage_method = st.radio(
        "Coverage calculation",
        ["Explicit coordinates", "Analytical area-density"],
        horizontal=True,
        key="coverage_method",
        help="Analytical coverage assumes uniform tree density. Scheduling and node status always use explicit modeled tree positions.",
    )
    mapped = result[f"network_{map_arch.lower()}"]
    nodes = mapped["nodes"]
    gateways = mapped["gateways"]
    fig = px.scatter(
        nodes,
        x="x",
        y="y",
        color="status",
        color_discrete_map={
            "Served": "#16834a",
            "Outside coverage": "#c64747",
            "Capacity limited": "#dc8a13",
            "Inactive candidate": "#4d84b8",
            "Unequipped": "#a5adb4",
        },
        hover_data=[c for c in ["tree_id", "gateway", "margin_db", "probability"] if c in nodes],
        title=f"Orchard reachability and service · Architecture {map_arch}",
    )
    if len(gateways):
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in gateways],
                y=[p[1] for p in gateways],
                mode="markers",
                name="Gateway",
                marker={"symbol": "star", "size": 14, "color": "black"},
            )
        )
        radius = result["radio"]["range_m"]
        for x, y in gateways:
            fig.add_shape(
                type="circle",
                x0=x - radius,
                y0=y - radius,
                x1=x + radius,
                y1=y + radius,
                line={"color": "#167a55", "dash": "dash"},
                fillcolor="rgba(22,122,85,0.08)",
            )
    from simulator.network import dimensions

    orchard_width, orchard_length = dimensions(cfg)
    fig.update_xaxes(range=[0, orchard_width], title="Orchard x (m)", constrain="domain")
    fig.update_yaxes(
        range=[0, orchard_length], title="Orchard y (m)", scaleanchor="x", scaleratio=1, constrain="domain"
    )
    st.plotly_chart(fig, width="stretch")
    coverage_ha = mapped.get(
        "analytical_coverage_ha" if coverage_method == "Analytical area-density" else "bounded_coverage_ha", 0.0
    )
    covered_trees = mapped.get(
        "analytical_covered_trees" if coverage_method == "Analytical area-density" else "covered_trees", 0
    )
    st.metric(f"{coverage_method} area", f"{coverage_ha:.2f} ha")
    st.metric(f"{coverage_method} tree estimate", f"{covered_trees:,.1f}")
    st.caption(
        "Analytical coverage assumes uniform tree density. Scheduling and node status always use explicit modeled tree positions."
    )

with tabs[3]:
    architecture = st.radio("Network architecture", ["A · remote CNN", "B · local CNN"], horizontal=True, key="network_architecture")
    scenario_load = st.radio("Request load", ["Current expected trigger rate", "Worst case · all reachable report"], horizontal=True, key="network_request_load")
    arch = "network_a" if architecture.startswith("A") else "network_b"
    network = result[arch] if scenario_load.startswith("Current") else result["worst_case"][arch]
    labels = [
        "Total trees",
        "Deployed nodes",
        "Long-run reachable",
        "Reachable",
        "Admitted",
        "Expected monitored",
        "Capacity rejected",
        "Outside coverage",
    ]
    vals = [
        network["total_trees"],
        network["deployed_nodes"],
        network["long_run_reachable_nodes"],
        network["reachable_nodes"],
        network["admitted_nodes"],
        network["served_nodes"],
        network["capacity_rejected"],
        network["outside_nodes"],
    ]
    st.plotly_chart(
        px.bar(
            x=labels,
            y=vals,
            title="Physical population, reachability, and capacity",
            labels={"x": "Population stage", "y": "Nodes"},
        ),
        width="stretch",
    )
    cols = st.columns(3)
    cols[0].metric("Bounded coverage proxy", f"{network['bounded_coverage_ha']:.2f} ha")
    cols[1].metric("Total physical trees", f"{network['total_trees']:,}")
    cols[2].metric("Gateway capacity", f"{network['capacity_nodes']:,} nodes")
    cols = st.columns(3)
    cols[0].metric("Admitted candidates", f"{network['admitted_nodes']:,.1f}")
    cols[1].metric("Expected successful transmissions", f"{network['expected_successful_transmissions']:.1f}")
    cols[2].metric("Expected monitored nodes", f"{network['served_nodes']:,.1f}")
    st.metric("Worst modeled latency", f"{network['latency_worst_s']:.2f} s")
    st.subheader("Load by cluster head")
    st.dataframe(network["gateway_load"], hide_index=True, width="stretch")
    st.caption(
        "Admitted population is the expected/current modeled node population after capacity. Sampled offered and scheduled requests are per reporting round. "
        + ("Capacity is an airtime reference in ALOHA; collisions reduce delivery probabilistically." if not network["gateway_load"].capacity_is_hard_limit.all() else "Capacity is a hard scheduled limit per head and round.")
    )
    if network["gateway_load"].capacity_rejected.sum():
        st.warning(
            f"{int(network['gateway_load'].capacity_rejected.sum()):,} offered node requests exceed their assigned head's reporting capacity. "
            "A spare head's unused capacity does not help when nodes associate with another head."
        )
    st.caption(
        "Admitted candidates are scheduled before delivery loss; expected monitored nodes include modeled link and packet success. All deployed nodes still wake and process, including inactive candidates."
    )
    dedicated_round_s = network["deployed_nodes"] * cfg["slot_ms"] / 1000
    st.caption(
        f"Dedicated TDMA round for all deployed nodes at the configured slot: {dedicated_round_s:,.1f} s of {cfg['window_s']:,.0f} s available."
    )
    if dedicated_round_s > cfg["window_s"]:
        st.warning("The full dedicated TDMA round cannot fit in the configured reporting window.")
    schedule = network.get("schedule", pd.DataFrame())
    if "schedule" not in network:
        st.caption("Worst-case load is a summary scenario without a generated timeline.")
    elif len(schedule) <= 10000:
        timeline = schedule.assign(
            duration_s=schedule.end_s - schedule.start_s,
            resource=schedule.gateway.astype(str) + " / " + schedule.channel.astype(str),
        )
        st.plotly_chart(
            go.Figure(
                go.Bar(
                    x=timeline.duration_s,
                    base=timeline.start_s,
                    y=timeline.resource,
                    orientation="h",
                    customdata=timeline.node,
                    hovertemplate="Node %{customdata}<br>Start %{base:.3f}s<br>Duration %{x:.3f}s<extra></extra>",
                )
            ).update_layout(
                title="Packet / reservation schedule",
                xaxis_title="Time in reporting round (s)",
                yaxis_title="Gateway / receiver channel",
                barmode="overlay",
            ),
            width="stretch",
        )
    else:
        st.caption(f"Timeline omitted for {len(schedule):,} slots; download the schedule CSV in Export Results.")
    st.dataframe(schedule.head(1000), width="stretch")

with tabs[4]:
    st.json(result["cnn"])
    st.caption(f"Performance input source: {cfg.get('performance_source', 'Exploratory assumption')}")

with tabs[5]:
    cnn = result["cnn"]
    st.metric("CNN inference energy", f"{cnn['inference_energy_mj']:.2f} mJ")
    if len(cnn["layers"]):
        st.plotly_chart(
            px.bar(
                cnn["layers"],
                x="layer",
                y=[col for col in ["parameters", "macs"] if col in cnn["layers"]],
                barmode="group",
                title="CNN layer workload",
            ),
            width="stretch",
        )
    else:
        st.info(
            "Layer workload is not expanded in Workload mode. Choose Layer designer and add layer definitions to chart per-layer costs."
        )
    st.metric(
        "Required arena", f"{cnn['arena_required_kb']:.1f} KB", f"Available contiguous {cnn['contiguous_kb']:.1f} KB"
    )

with tabs[6]:
    calculation_help("Deployment totals and optimization")
    rows = [
        {
            "Architecture": name,
            "Daily node energy (J)": result["energy"][name.lower()]["daily_j"],
            "Battery life (days)": result["energy"][name.lower()]["battery_days"],
            "Deployment energy (kWh/day)": result["energy"][f"deployment_{name.lower()}"]["total_kwh_day"],
        }
        for name in ["A", "B"]
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.metric("Net energy saved by local inference", f"{result['energy']['net_saving_mj']:.2f} mJ/event")
    cols = st.columns(3)
    cols[0].metric("TX energy saved", f"{energy['tx_saving_mj']:.2f} mJ/event")
    cols[1].metric("Break-even latency", f"{energy['break_even_ms']:.1f} ms")
    cols[2].metric("Break-even current", f"{energy['break_even_ma']:.1f} mA")
    components = [
        {"Architecture": arch.upper(), "Component": label, "kWh/day": energy[f"deployment_{arch}"][field]}
        for arch in ("a", "b")
        for field, label in [
            ("all_deployed_kwh_day", "All sensor batteries"),
            ("gateway_kwh_day", "Gateways"),
            ("server_kwh_day", "Central server"),
        ]
    ]
    st.plotly_chart(
        px.bar(
            pd.DataFrame(components), x="Architecture", y="kWh/day", color="Component", title="Entire deployment energy"
        ),
        width="stretch",
    )
    st.caption("Server inference scaling at the selected trigger rate (hypothetical admitted populations):")
    st.dataframe(energy["server_scale"], hide_index=True, width="stretch")

with tabs[7]:
    e = result["energy"]
    st.metric("CNN inference energy", f"{result['cnn']['inference_energy_mj']:.2f} mJ")
    st.plotly_chart(
        px.bar(
            e["b"]["stages"],
            x="stage",
            y="energy_mj",
            title="Architecture B event-energy stages",
            labels={"x": "Stage", "y": "Energy (mJ)"},
        ),
        width="stretch",
    )
    stages = e["b"]["stages"].copy()
    st.plotly_chart(
        go.Figure(
            go.Waterfall(
                x=list(stages.stage) + ["Total"],
                y=list(stages.energy_mj) + [0],
                measure=["relative"] * len(stages) + ["total"],
            )
        ).update_layout(title="Accepted-event energy waterfall · B", yaxis_title="Energy at rail (mJ)"),
        width="stretch",
    )
    stage_ends = stages["duration_ms"].cumsum()
    waveform_x, waveform_y = [], []
    for index, stage in stages.iterrows():
        start = 0 if index == 0 else stage_ends.iloc[index - 1]
        end = stage_ends.iloc[index]
        waveform_x.extend([start, end])
        waveform_y.extend([stage["current_ma"], stage["current_ma"]])
    st.plotly_chart(
        go.Figure(
            go.Scatter(x=waveform_x, y=waveform_y, mode="lines", line_shape="hv", name="Regulated-rail current")
        ).update_layout(
            title="Per-event current waveform",
            xaxis_title="Cumulative stage time (ms)",
            yaxis_title="Current (mA at configured rail)",
        ),
        width="stretch",
    )
    st.json(
        {
            "Architecture A": e["a"],
            "Architecture B": e["b"],
            "break_even_ms": e["break_even_ms"],
            "break_even_ma": e["break_even_ma"],
        }
    )

with tabs[8]:
    calculation_help("Deployment totals and optimization")
    values = {
        "sf": [7, 8, 9, 10, 11, 12],
        "frequency_mhz": [433, 915],
        "inference_latency_ms": [100, 200, 400, 800, 1500, 3000],
        "inference_current_ma": [30, 50, 70, 100, 150],
        "tx_dbm": [5, 8, 11, 14, 17, 20],
        "area_ha": [1, 10, 100, 500, 1000],
        "trees_per_ha": [25, 50, 100, 200, 400],
        "gateways": [1, 2, 4, 8],
        "trigger_probability": [0, 0.1, 0.25, 0.5, 0.75, 1],
    }[sensitivity_parameter]
    from simulator.research import sweep

    sensitivity_df = sweep(
        cfg, {sensitivity_parameter: values}, method="Full factorial", samples=len(values), seed=int(cfg["seed"])
    )
    output_columns = [
        col
        for col in [
            "range_m",
            "coverage_ha",
            "reachable_nodes",
            "served_a",
            "served_b",
            "capacity_a",
            "capacity_b",
            "battery_a_days",
            "battery_b_days",
            "system_a_kwh_day",
            "system_b_kwh_day",
            "airtime_a_ms",
            "airtime_b_ms",
            "cnn_mj",
            "break_even_ms",
        ]
        if col in sensitivity_df
    ]
    sensitivity_metric = st.selectbox(
        "Response metric",
        output_columns,
        index=output_columns.index(
            "served_b"
            if sensitivity_parameter.startswith("inference_") and "served_b" in output_columns
            else "served_a"
        )
        if ("served_b" if sensitivity_parameter.startswith("inference_") else "served_a") in output_columns
        else 0,
        key="sensitivity_metric",
    )
    st.plotly_chart(
        px.line(
            sensitivity_df,
            x=sensitivity_parameter,
            y=sensitivity_metric,
            markers=True,
            title=f"{sensitivity_metric} by {sensitivity_parameter}",
        ),
        width="stretch",
    )
    if "error" in sensitivity_df and (sensitivity_df["error"] != "").any():
        st.warning("Some points are invalid under the active configuration; they are retained with validation errors.")
        st.dataframe(
            sensitivity_df.loc[sensitivity_df["error"] != "", [sensitivity_parameter, "error"]], hide_index=True
        )

with tabs[9]:
    calculation_help("Deployment totals and optimization")
    if run_sweep:
        try:
            from simulator.research import sweep

            st.session_state.sweep_result = sweep(
                cfg, sweep_values, method=method, samples=int(samples), seed=int(cfg.get("seed", 42))
            )
            st.session_state.pop("pareto", None)
            st.session_state.sweep_signature = json.dumps(
                {"config": cfg, "method": method, "samples": int(samples), "space": sweep_values},
                sort_keys=True,
                default=str,
            )
        except (ValueError, ImportError) as exc:
            st.error(f"Sweep could not run: {exc}")
    if "sweep_result" in st.session_state:
        current_signature = json.dumps(
            {"config": cfg, "method": method, "samples": int(samples), "space": sweep_values},
            sort_keys=True,
            default=str,
        )
        if current_signature != st.session_state.get("sweep_signature"):
            st.warning("Sweep results are stale because the active configuration changed. Rerun the sweep.")
        df = st.session_state.sweep_result
        if not df.empty:
            arch = objective_arch.lower()
            objective_columns = {
                "Max monitored trees under targets": f"eligible_trees_{arch}",
                "Max monitored nodes under targets": f"eligible_served_{arch}",
                "Max battery life": f"battery_{arch}_days",
                "Min node energy": f"node_{arch}_daily_j",
                "Min deployment energy": f"system_{arch}_kwh_day",
                "Min gateways": "gateways",
                "Max coverage": "coverage_ha",
                "Min airtime": f"airtime_{arch}_ms",
                "Max CNN parameters": "parameters",
                "Min energy per monitored tree": f"energy_per_served_tree_{arch}_j",
            }
            objective_column = objective_columns[objective]
            maximize = objective.startswith("Max")
            valid = df.loc[df["error"].eq("")].copy() if "error" in df else df.copy()
            feasible_column = f"feasible_{arch}"
            if "under targets" not in objective and feasible_column in valid:
                feasible_rows = valid.loc[valid[feasible_column]]
                if not feasible_rows.empty:
                    valid = feasible_rows
                else:
                    st.warning(
                        "No design meets the selected architecture feasibility targets; showing the best valid modeled candidates."
                    )
            if objective_column in valid and not valid.empty:
                best = valid.sort_values(objective_column, ascending=not maximize).head(10)
                st.markdown(f"**Best candidates · {objective} · Architecture {objective_arch}**")
                st.dataframe(best, hide_index=True, width="stretch")
            numeric = list(df.select_dtypes("number").columns)
            metrics = [n for n in numeric if n not in sweep_parameters and n != "search_index"]
            x_options = [n for n in numeric if n != "search_index"]
            default_x = sweep_parameters[0] if sweep_parameters and sweep_parameters[0] in x_options else x_options[0]
            x = st.selectbox("X axis", x_options, index=x_options.index(default_x), key="sweep_x")
            y = st.selectbox(
                "Response metric",
                metrics,
                index=metrics.index(f"eligible_served_{arch}") if f"eligible_served_{arch}" in metrics else 0,
                key="sweep_y",
            )
            varying = [p for p in sweep_parameters if p in df and df[p].nunique() > 1]
            chart_options = ["Scatter", "Distribution", "Tornado"] + (
                ["Heatmap", "3D surface"] if len(varying) >= 2 else []
            )
            chart = st.selectbox("Graph", chart_options, key="sweep_chart")
            if chart == "Scatter":
                fig = px.scatter(
                    df,
                    x=x,
                    y=y,
                    color=feasible_column if feasible_column in df else None,
                    title="Configuration response explorer",
                )
            elif chart == "Heatmap":
                second = st.selectbox("Second input axis", [p for p in varying if p != x], key="sweep_heatmap_y")
                fig = px.imshow(
                    df.pivot_table(index=second, columns=x, values=y, aggfunc="mean"),
                    aspect="auto",
                    labels={"color": y},
                    title=f"Mean {y} heatmap",
                )
            elif chart == "3D surface":
                second = st.selectbox("Second input axis", [p for p in varying if p != x], key="sweep_surface_y")
                grid = df.pivot_table(index=second, columns=x, values=y, aggfunc="mean")
                fig = go.Figure(go.Surface(x=grid.columns, y=grid.index, z=grid.to_numpy()))
                fig.update_layout(
                    title=f"Three-dimensional response · {y}",
                    scene={"xaxis_title": x, "yaxis_title": second, "zaxis_title": y},
                )
            elif chart == "Distribution":
                metric = st.selectbox("Distribution metric", metrics, key="sweep_distribution_metric")
                fig = px.histogram(df, x=metric, marginal="box", title=f"Distribution of {metric}")
            else:
                metric = st.selectbox(
                    "Tornado response",
                    metrics,
                    index=metrics.index(f"eligible_served_{arch}") if f"eligible_served_{arch}" in metrics else 0,
                    key="sweep_tornado_metric",
                )
                impacts = []
                for parameter in sweep_parameters:
                    if parameter in df and df[parameter].nunique() > 1:
                        response = df.groupby(parameter)[metric].mean()
                        impacts.append({"Parameter": parameter, "Impact range": response.max() - response.min()})
                fig = px.bar(
                    pd.DataFrame(impacts, columns=["Parameter", "Impact range"]).sort_values("Impact range"),
                    x="Impact range",
                    y="Parameter",
                    orientation="h",
                    title=f"Tornado sensitivity: {metric}",
                )
            st.plotly_chart(fig, width="stretch")
            st.dataframe(df, width="stretch")
            if st.button("Extract Pareto front"):
                from simulator.research import pareto_front

                st.session_state.pareto = pareto_front(df, architecture=arch)
            if "pareto" in st.session_state:
                st.plotly_chart(px.scatter(st.session_state.pareto, x=x, y=y, title="Pareto front"), width="stretch")
            st.download_button("Download sweep CSV", df.to_csv(index=False), "sweep.csv", "text/csv")

with tabs[10]:
    from simulator.io import LIMITATIONS, SOURCES

    st.subheader("How the estimates are calculated")
    st.markdown("**Evidence labels**")
    st.write(
        "Formula results are estimates. “Measured” is a scenario source label; provide the imported data or a measurement/source note to substantiate it. Defaults are not evidence, and a published benchmark is not applied unless you enter matching values and cite it."
    )
    st.write(f"Performance source: {cfg.get('performance_source', 'Exploratory assumption')}. Source note: {cfg.get('measurement_note') or 'None supplied.'}")
    formula_search = st.text_input("Find a formula or constant", key="formula_search")
    visible_guides = [
        guide
        for guide in FORMULA_GUIDE
        if formula_search.lower() in " ".join(str(value) for value in guide.values()).lower()
    ]
    examples = {
        "Radio range": f"Modeled radius: {result['radio']['range_m']:,.1f} m at {cfg['frequency_mhz']} MHz.",
        "LoRa airtime": f"This scenario's packets: A {result['radio']['airtime_a_ms']:.2f} ms; B {result['radio']['airtime_b_ms']:.2f} ms.",
        "Plantation and coverage": f"{result['network_a']['total_trees']:,} physical trees; {result['network_a']['reachable_nodes']:,} nodes pass the configured link requirement.",
        "Reporting capacity": f"A capacity: {result['network_a']['capacity_nodes'] // max(cfg['gateways'], 1):,} reports/head/round; current slot {result['network_a']['slot_s']:.3f} s.",
        "Delivery probability": f"Expected A delivery probability after the configured retry policy: {result['network_a']['delivery_probability']:.1%}.",
        "CNN workload and memory": f"Current workload: {result['cnn']['parameters']:,} parameters, {result['cnn']['macs']:,} MACs; estimated arena {result['cnn']['arena_required_kb']:.1f} KB.",
        "Inference timing and energy": f"Current effective latency/current: {result['cnn']['effective_latency_ms']:g} ms / {result['cnn']['effective_current_ma']:g} mA; modeled inference energy {result['cnn']['inference_energy_mj']:.2f} mJ.",
        "Radio current": f"Modeled TX radio current: {result['radio']['tx_current_ma']:.1f} mA; manufacturer-family reference interpolation for the uncharacterized SX1278 board.",
        "Recording and SNR gate": f"Expected initial reports: {cfg['recordings_per_day']} opportunities/day × {cfg['trigger_probability']:.0%} trigger = {cfg['recordings_per_day'] * cfg['trigger_probability']:.1f} per admitted node/day.",
        "Daily energy and battery life": f"Nominal battery: {cfg['battery_v']:g} V × {cfg['battery_mah']:g} mAh / 1000 = {cfg['battery_v'] * cfg['battery_mah'] / 1000:.2f} Wh; estimated A life {result['energy']['a']['battery_days']:.1f} days.",
        "Deployment totals and optimization": f"A deployment energy: {result['energy']['deployment_a']['total_kwh_day']:.4f} kWh/day; modeled deployment cost {result['energy']['deployment_a']['cost']:,.2f} currency units.",
        "Shadowing and reliability": f"Selected model: {cfg['propagation']}; exponent {cfg['path_loss_exponent']:g}, shadowing {cfg['shadowing_db']:g} dB, reachability target {cfg['min_reliability']:.0%}.",
        "Schedules, retries, and monitoring": f"Expected traffic: {cfg['recordings_per_day'] * cfg['trigger_probability']:.1f} initial reports/node/day; admitted A population {result['network_a']['admitted_nodes']:,.1f}.",
    }
    for guide in visible_guides:
        with st.expander(guide["Topic"]):
            st.write("**Formula**")
            st.code(guide["Formula"].replace("; ", ";\n").replace(". ", ".\n"), language=None, wrap_lines=True)
            st.caption(examples.get(guide["Topic"], ""))
            st.write("**Why it is used**")
            st.write(guide["Why it is used"])
            st.write("**Constants, units, and source boundary**")
            st.write(guide["Constants and source"])
    with st.expander("Current value, app default, and evidence for every input"):
        st.caption("Changed values are scenario overrides. They are not measured unless a measurement or source note documents them.")
        st.dataframe(pd.DataFrame(input_provenance(cfg)), hide_index=True, width="stretch")
    st.markdown("**Sources**")
    for source in SOURCES:
        if source["url"]:
            st.markdown(f"- [{source['name']}]({source['url']}) — {source['basis']}: {source['use']}")
        else:
            st.markdown(f"- {source['name']} — {source['use']}")
    st.markdown(
        "- [ITU-R P.525 PDF](https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.525-5-202411-I!!PDF-E.pdf) — physical free-space loss formula and the standard's rounded 32.4 dB constant."
    )
    st.markdown("**Limitations**")
    for limitation in LIMITATIONS:
        st.markdown(f"- {limitation}")

with tabs[11]:
    from simulator.io import export_json, report_markdown

    st.download_button("Download full result JSON", export_json(result), "simulation.json", "application/json")
    st.download_button("Download research report", report_markdown(result), "simulation_report.md", "text/markdown")
    st.download_button(
        "Download network nodes CSV", result["network_a"]["nodes"].to_csv(index=False), "nodes.csv", "text/csv"
    )
    st.download_button(
        "Download schedule CSV", result["network_a"]["schedule"].to_csv(index=False), "schedule.csv", "text/csv"
    )
