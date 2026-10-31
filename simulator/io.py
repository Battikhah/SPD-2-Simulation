"""Strict scenario/measurement import and reproducible, source-labeled exports."""

import json
from copy import deepcopy
from datetime import datetime, timezone
from math import isfinite

import numpy as np
import pandas as pd

from .config import DEFAULTS
from .engine import validate_config
from .hardware import HARDWARE_BASELINE
from .radio import fspl_db

SOURCES = [
    {
        "name": "Semtech SX1278 product documentation",
        "url": "https://www.semtech.com/products/wireless-rf/lora-connect/sx1278",
        "basis": "Manufacturer product reference",
        "use": "Purchased radio family and low-frequency band reference; module-board configuration and RF performance remain unverified.",
    },
    {
        "name": "Microchip MCP602 product documentation",
        "url": "https://www.microchip.com/en-us/product/mcp602",
        "basis": "Manufacturer product reference",
        "use": "Purchased amplifier family; not a substitute for the report's MCP6002 circuit or an amplifier-current measurement.",
    },
    {
        "name": "Semtech SX1276 datasheet and product documents",
        "url": "https://www.semtech.com/products/wireless-rf/lora-connect/sx1276",
        "basis": "Manufacturer specified",
        "use": "Airtime equation, SF6 implicit header, low-data-rate optimization and raw-LoRa payload limit.",
    },
    {
        "name": "HopeRF RFM95W/96W/98W v2.0",
        "url": "https://www.hoperf.com/uploads/RFM95W-V2.0_1695351437.pdf",
        "basis": "Manufacturer specified",
        "use": "Legacy low-band sensitivity table and TX current anchors used as family-reference proxies for the uncharacterized purchased SX1278 board; current interpolation is estimated, not board measured.",
    },
    {
        "name": "Espressif ESP32 memory types",
        "url": "https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-guides/memory-types.html",
        "basis": "Manufacturer specified",
        "use": "320 KiB DRAM / 200 KiB IRAM distinction and static allocation limits.",
    },
    {
        "name": "ESP-NN implementation and benchmarks",
        "url": "https://github.com/espressif/esp-nn",
        "basis": "Published benchmark reference",
        "use": "Backend reference only; no benchmark transferred to this CNN or original ESP32.",
    },
    {
        "name": "TFLite Micro memory management",
        "url": "https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/memory_management.md",
        "basis": "Implementation documentation",
        "use": "Arena head/tail, planning and operator scratch require actual allocation measurements.",
    },
    {
        "name": "Senior Design 1 Report.pdf",
        "url": "",
        "basis": "Unavailable",
        "use": "Not attached. 15×13 input, 10 s / 8 kHz, 12 rounds/day and 900 ms slot are supplied-prompt assumptions, not verified report facts.",
    },
]

LIMITATIONS = [
    "No hardware, RF field, classifier accuracy or battery measurement is bundled. No design is marked verified feasible.",
    "Log-distance exponent, shadowing, foliage and independent retry outcomes are exploratory; no plantation-specific propagation calibration is bundled.",
    "Coverage circle is theoretical and unbounded; explicit tree counts respect orchard boundaries and gateway overlap.",
    "Strongest-link gateway association is heuristic. Dynamic monitored capacity uses expected event load; seeded actual events and schedules are shown separately.",
    "ALOHA uses exp(-2G), with no capture, adjacent-channel rejection or inter-SF interference simulation.",
    "Sequential CNN activation and serialized-size estimates do not replace TFLM arena instrumentation, an operator audit or a compiled ESP32 test.",
    "MAC-based latency is an independent exploratory estimate. Manual latency/current controls are not predictions from model parameter count.",
    "Battery model assumes constant usable capacity factors and load, constant solar yield and exponential self discharge; voltage sag and discharge curves are unmodeled.",
    "All deployed nodes wake and process. Scheduled nodes add communication energy; no battery savings are claimed for silently dropping unserved nodes.",
    "This models acoustic gating and workload, not pest classification accuracy. Trigger probability must be calibrated against actual acoustic windows.",
]


def _clean(value):
    if isinstance(value, pd.DataFrame):
        return _clean(value.to_dict(orient="records"))
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not isfinite(value):
        return None
    return value


def dumps_scenario(c):
    validate_config(c)
    return json.dumps({"schema_version": 1, "config": c}, indent=2, allow_nan=False)


def loads_scenario(text):
    if len(text) > 25_000_000:
        raise ValueError("Scenario is larger than 25 MB.")
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid scenario JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Scenario must be a JSON object.")
    if "config" in data:
        if data.get("schema_version", 1) != 1:
            raise ValueError("Unsupported scenario schema version.")
        data = data["config"]
    if not isinstance(data, dict):
        raise ValueError("Scenario config must be an object.")
    config = deepcopy(DEFAULTS)
    config.update(data)
    if "recording_hours" not in data and config["recordings_per_day"] != len(DEFAULTS["recording_hours"]):
        config["recording_hours"] = []
    return validate_config(config)


def export_json(result):
    output = dict(
        result,
        schema_version=1,
        provenance={
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "hardware_verified": False,
            "sources": SOURCES,
            "limitations": LIMITATIONS,
            "nonfinite_values": "null represents undefined / unbounded modeled outputs.",
            "measurement_note": result["config"]["measurement_note"],
            "user_reported_hardware": HARDWARE_BASELINE,
        },
    )
    return json.dumps(_clean(output), indent=2, allow_nan=False)


def report_markdown(r):
    c, a, b = r["config"], r["network_a"], r["network_b"]
    e = r["energy"]
    lines = [
        "# Palm sensing simulation research report",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "**All results are calculated estimates; no hardware verification is implied.**",
        "",
        f"Configuration: {c['frequency_mhz']} MHz, SF{c['sf']}, {c['bandwidth_khz']} kHz, {c['tx_dbm']} dBm.",
        "",
        "## User-reported hardware baseline",
        "",
        "```json",
        json.dumps(HARDWARE_BASELINE, indent=2),
        "```",
        "",
        "Hardware quantities describe reported purchases; they do not scale per-node battery capacity or limit modeled deployment counts.",
        f"Channel model: {c['propagation']}. Seed: {c['seed']}.",
        f"Inference profile: {c['performance_source']}. {c['measurement_note']}",
        "",
        "| Metric | Architecture A | Architecture B |",
        "|---|---:|---:|",
        f"| Physical trees | {a['total_trees']} | {b['total_trees']} |",
        f"| Reachable nodes | {a['reachable_nodes']} | {b['reachable_nodes']} |",
        f"| Expected monitored nodes | {a['served_nodes']:.2f} | {b['served_nodes']:.2f} |",
        f"| Expected unique monitored trees | {a['served_trees']:.2f} | {b['served_trees']:.2f} |",
        f"| Airtime (ms) | {r['radio']['airtime_a_ms']:.3f} | {r['radio']['airtime_b_ms']:.3f} |",
        f"| Node battery energy (J/day) | {e['a']['battery_daily_j']:.3f} | {e['b']['battery_daily_j']:.3f} |",
        f"| Battery life (days) | {e['a']['battery_days']:.1f} | {e['b']['battery_days']:.1f} |",
        f"| Entire deployment (kWh/day) | {e['deployment_a']['total_kwh_day']:.6f} | {e['deployment_b']['total_kwh_day']:.6f} |",
        "",
        "## Cluster-head reporting capacity",
        "",
        f"Each modeled gateway site has a per-round {'airtime reference' if not a['gateway_load'].capacity_is_hard_limit.all() else 'scheduled capacity'} of {a['gateway_load'].capacity_nodes.iloc[0] if len(a['gateway_load']) else 0} nodes for A and {b['gateway_load'].capacity_nodes.iloc[0] if len(b['gateway_load']) else 0} for B, across {c['channels']} RF channel(s) and {c['receivers']} receiver(s). TDMA capacity applies at each assigned head; ALOHA collisions are modeled probabilistically. Unused capacity at another head cannot serve overloaded assignments.",
        "The modeled head is a gateway receiving site; no additional relay hop or buffer is assumed.",
        "",
        f"CNN inference energy: {e['cnn_mj']:.3f} mJ. Single-attempt transmission saving: {e['tx_saving_mj']:.3f} mJ.",
        f"Core break-even inference time: {e['break_even_ms']:.3f} ms.",
        "",
        "## Feasibility",
        "",
        r["feasibility"]["status"],
        "",
    ]
    if "worst_case" in r:
        lines.insert(
            lines.index("The modeled head is a gateway receiving site; no additional relay hop or buffer is assumed."),
            f"At a 100% trigger rate, A admits {r['worst_case']['network_a']['admitted_nodes']:.1f} of {r['worst_case']['network_a']['reachable_nodes']} reachable nodes per recording opportunity and rejects {r['worst_case']['network_a']['capacity_rejected']:,}; B admits {r['worst_case']['network_b']['admitted_nodes']:.1f} and rejects {r['worst_case']['network_b']['capacity_rejected']:,}.",
        )
        lines.insert(
            lines.index("The modeled head is a gateway receiving site; no additional relay hop or buffer is assumed."),
            "This worst case means all reachable nodes request a report on every configured recording opportunity.",
        )
    lines += [f"- {issue}" for issue in r["feasibility"]["issues"]]
    lines += ["", "## Limitations", ""] + [f"- {line}" for line in LIMITATIONS]
    lines += ["", "## Sources", ""] + [
        f"- [{s['name']}]({s['url']}) — {s['basis']}: {s['use']}" if s["url"] else f"- {s['name']} — {s['use']}"
        for s in SOURCES
    ]
    lines += ["", "## Reproducible configuration", "", "```json", dumps_scenario(c), "```"]
    return "\n".join(lines)


def import_measurements(c, kind, df):
    """Apply supplied observations only; no hidden measurement or provenance generation."""
    config = deepcopy(c)
    columns = {
        "Inference": ["latency_ms", "current_ma"],
        "Current profile": ["time_ms", "current_ma"],
        "RSSI": ["distance_m", "rssi_dbm"],
        "Packet success": ["sent", "received"],
        "Trees": ["x", "y"],
        "Gateways": ["x", "y"],
    }
    if kind not in columns or not set(columns[kind]).issubset(df.columns):
        raise ValueError(f"{kind} import requires columns: {columns.get(kind, [])}")
    data = df[columns[kind]].apply(pd.to_numeric, errors="coerce")
    if data.empty or len(data) > 250000 or not np.isfinite(data.to_numpy()).all():
        raise ValueError("Import needs 1–250000 finite numeric rows; missing/non-numeric values are rejected.")
    if kind == "Inference":
        if (data <= 0).any().any():
            raise ValueError("Measured latency and current must be positive.")
        config["inference_latency_ms"] = float(data.latency_ms.mean())
        config["inference_current_ma"] = float(np.average(data.current_ma, weights=data.latency_ms))
        config["current_profile"] = []
        config["performance_source"] = "Measured"
    elif kind == "Current profile":
        config["current_profile"] = data.to_dict(orient="records")
        config["inference_latency_ms"] = float(data.time_ms.iloc[-1] - data.time_ms.iloc[0])
        if config["inference_latency_ms"] <= 0:
            raise ValueError("Profile duration must be positive.")
        config["inference_current_ma"] = float(
            np.trapezoid(data.current_ma, data.time_ms) / config["inference_latency_ms"]
        )
        config["performance_source"] = "Measured"
    elif kind == "RSSI":
        if len(data) < 3 or (data.distance_m <= 0).any() or data.distance_m.nunique() < 2:
            raise ValueError(
                "RSSI calibration needs at least 3 positive-distance observations at two or more distances."
            )
        d0 = config["reference_distance_m"]
        if (data.distance_m < d0).any():
            raise ValueError("RSSI distances must be at least the configured reference distance.")
        observed_loss = (
            config["tx_dbm"]
            + config["tx_gain_dbi"]
            + config["rx_gain_dbi"]
            - config["cable_loss_db"]
            - config["misc_loss_db"]
            - config["interference_loss_db"]
            - data.rssi_dbm.to_numpy()
        )
        x = np.log10(data.distance_m.to_numpy() / d0)
        slope, intercept = np.polyfit(x, observed_loss, 1)
        vegetation = float(intercept - fspl_db(d0, config["frequency_mhz"]))
        if slope <= 0 or vegetation < 0:
            raise ValueError(
                "Fit gives a nonphysical exponent or negative extra loss under current antenna/reference assumptions; review observations."
            )
        config.update(
            path_loss_exponent=float(slope / 10),
            vegetation_loss_db=vegetation,
            shadowing_db=float(np.std(observed_loss - (intercept + slope * x), ddof=1)),
            propagation="Log-distance",
        )
    elif kind == "Packet success":
        if (
            (data.sent <= 0).any()
            or (data.received < 0).any()
            or (data.received > data.sent).any()
            or not np.equal(data, np.floor(data)).all().all()
        ):
            raise ValueError("Packet counts must be integers: 0 <= received <= sent and sent > 0.")
        config["packet_success"] = float(data.received.sum() / data.sent.sum())
    else:
        if (data < 0).any().any():
            raise ValueError("Coordinates are orchard-local nonnegative metres.")
        config["custom_trees" if kind == "Trees" else "gateway_positions"] = data.to_numpy().tolist()
        if kind == "Trees":
            config["layout"] = "Custom"
        else:
            config["gateways"] = len(data)
    config["measurement_note"] = (
        config["measurement_note"]
        + "\n"
        + f"User CSV import: {kind}, {len(data)} observations. Calibration applies to the current hardware/settings; raw observations must be retained by the researcher."
    ).strip()
    return validate_config(config)
