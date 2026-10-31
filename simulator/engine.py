"""One deterministic dependency pipeline shared by the UI, tests and search."""

from copy import deepcopy
from math import isfinite

from .cnn import analyze_cnn
from .config import DEFAULTS, OPTIONS
from .energy import calculate_energy
from .network import simulate_network
from .radio import radio_summary


def validate_config(c):
    """Reject malformed or unbounded imported inputs before numerical allocation."""
    unknown = set(c) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    missing = set(DEFAULTS) - set(c)
    if missing:
        raise ValueError(f"Missing configuration fields: {sorted(missing)}")
    signed = {"tx_dbm", "tx_gain_dbi", "rx_gain_dbi", "snr_gate_db"}
    for key, default in DEFAULTS.items():
        value = c[key]
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean.")
        elif isinstance(default, (int, float)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError(f"{key} must be a finite number.")
            if key not in signed and value < 0:
                raise ValueError(f"{key} must be nonnegative.")
            if isinstance(default, int) and not isinstance(value, int):
                raise ValueError(f"{key} must be an integer.")
        elif isinstance(default, str) and not isinstance(value, str):
            raise ValueError(f"{key} must be text.")
        elif isinstance(default, list) and not isinstance(value, list):
            raise ValueError(f"{key} must be a list.")
    for key, values in OPTIONS.items():
        if c[key] not in values:
            raise ValueError(f"{key} must be one of {values}.")
    for key in (
        "area_ha",
        "width_m",
        "length_m",
        "reference_distance_m",
        "path_loss_exponent",
        "battery_v",
        "battery_mah",
        "regulator_efficiency",
        "voltage_v",
        "inference_latency_ms",
        "inference_current_ma",
        "cpu_utilization",
        "cycles_per_mac",
        "window_s",
        "tree_spacing_m",
        "deadline_ms",
        "sample_rate_hz",
    ):
        if c[key] <= 0:
            raise ValueError(f"{key} must be positive.")
    for key in (
        "trigger_probability",
        "equipped_fraction",
        "packet_success",
        "obstruction_probability",
        "usable_fraction",
        "temperature_factor",
        "aging_factor",
        "regulator_efficiency",
        "cpu_utilization",
        "delivery_target",
    ):
        if c[key] > 1:
            raise ValueError(f"{key} must be in [0, 1].")
    if not 0.5 <= c["min_reliability"] < 1:
        raise ValueError("Minimum link reliability must be at least 0.5 and below 1.")
    if not 0 <= c["self_discharge_monthly"] < 1:
        raise ValueError("Monthly self discharge must be in [0, 1).")
    if not 1 <= c["gateways"] <= 20 or not 1 <= c["channels"] <= 64 or not 1 <= c["receivers"] <= 64:
        raise ValueError("Use 1–20 gateways and 1–64 channels/receiver resources.")
    if not 1 <= c["nodes_per_tree"] <= 20 or c["retries"] > 10:
        raise ValueError("Use 1–20 nodes/tree and at most 10 retries.")
    if not 0 < c["dram_kb"] <= 320:
        raise ValueError("ESP32 has at most 320 KiB data RAM; 520 KiB is not one allocatable data pool.")
    if not 2 <= c["tx_dbm"] <= 20:
        raise ValueError("PA_BOOST output power must be in 2–20 dBm.")
    if not 1 <= c["path_loss_exponent"] <= 8:
        raise ValueError("Use a path-loss exponent between 1 and 8 for this model.")
    for key in ("tx_gain_dbi", "rx_gain_dbi"):
        if not -30 <= c[key] <= 50:
            raise ValueError(f"{key} must be between -30 and 50 dBi.")
    if c["frequency_mhz"] == 915 and c["bandwidth_khz"] < 125:
        raise ValueError("The 915 MHz RFM95W/SX1276 comparison profile supports 125–500 kHz bandwidth.")
    hours = c["recording_hours"]
    if hours and (
        len(hours) != c["recordings_per_day"]
        or any(isinstance(hour, bool) or not isinstance(hour, int) or not 0 <= hour <= 23 for hour in hours)
        or len(set(hours)) != len(hours)
    ):
        raise ValueError("Recording hours must be unique integer hours from 0–23 and match recordings/day.")
    if not 0 < c["filter_low_hz"] < c["filter_high_hz"] <= c["sample_rate_hz"] / 2:
        raise ValueError("Bandpass limits must satisfy 0 < low < high <= the sample-rate Nyquist frequency.")
    if c["sf"] == 6 and not c["implicit_header"]:
        raise ValueError("SF6 requires implicit header mode and matched receiver configuration.")
    for key in ("payload_a_bytes", "payload_b_bytes"):
        if not 1 <= c[key] + c["metadata_bytes"] + c["protocol_bytes"] <= 255:
            raise ValueError(f"{key} plus metadata/overhead must be 1–255 bytes for this raw-LoRa model.")
    if not 6 <= c["preamble_symbols"] <= 65535:
        raise ValueError("Preamble must be 6–65535 programmed symbols.")
    tsym_ms = 2 ** c["sf"] / c["bandwidth_khz"]
    if tsym_ms > 16 and c["ldro"] == "off":
        raise ValueError("Low-data-rate optimization is required for symbol times above 16 ms.")
    for key in ("custom_trees", "gateway_positions"):
        coords = c[key]
        if len(coords) > (250000 if key == "custom_trees" else 20):
            raise ValueError("Coordinate import exceeds the application resource limit.")
        for p in coords:
            if (
                not isinstance(p, (list, tuple))
                or len(p) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) for v in p)
            ):
                raise ValueError(f"{key} must contain finite [x_m, y_m] coordinates.")
    area = c["width_m"] * c["length_m"] / 10000 if c["dimensions_override"] else c["area_ha"]
    tree_count = len(c["custom_trees"]) if c["layout"] == "Custom" else int(area * c["trees_per_ha"])
    if tree_count * c["nodes_per_tree"] > 250000:
        raise ValueError("At most 250,000 sensor positions per run; lower density/area or nodes per tree.")
    if c["layout"] == "Custom" and not c["custom_trees"]:
        raise ValueError("Custom layout needs imported tree coordinates.")
    if c["gateway_positions"] and len(c["gateway_positions"]) != c["gateways"]:
        raise ValueError("Imported gateway coordinates must match gateway count.")
    if len(c["input_shape"]) != 3 or any(
        not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 4096 for v in c["input_shape"]
    ):
        raise ValueError("Input shape must contain three positive integer dimensions, each at most 4096.")
    if len(c["layers"]) > 100:
        raise ValueError("Use at most 100 sequential layers.")
    for layer in c["layers"]:
        if not isinstance(layer, dict) or layer.get("type") not in (
            "Conv2D",
            "MaxPool",
            "AveragePool",
            "Flatten",
            "Dense",
            "GlobalAveragePool",
            "Activation",
        ):
            raise ValueError("Each layer needs a supported type.")
        if layer.get("padding", "valid") not in ("valid", "same"):
            raise ValueError("Layer padding must be valid or same.")
        for key in ("kernel", "stride", "filters", "units"):
            if key in layer and (
                not isinstance(layer[key], int) or isinstance(layer[key], bool) or not 1 <= layer[key] <= 100000
            ):
                raise ValueError(f"Layer {key} must be a positive integer, at most 100000.")
    profile = c["current_profile"]
    if profile:
        if len(profile) < 2 or len(profile) > 100000:
            raise ValueError("Current profile requires 2–100000 samples.")
        prev = -1
        for point in profile:
            if not isinstance(point, dict) or set(point) != {"time_ms", "current_ma"}:
                raise ValueError("Current profile needs time_ms/current_ma columns.")
            t, i = point["time_ms"], point["current_ma"]
            if (
                not all(isinstance(v, (int, float)) and not isinstance(v, bool) and isfinite(v) for v in (t, i))
                or t <= prev
                or i < 0
            ):
                raise ValueError("Profile times must strictly increase and currents must be finite/nonnegative.")
            prev = t
        if profile[0]["time_ms"] != 0:
            raise ValueError("Current profile must start at time_ms=0.")
    if c["performance_source"] != "Exploratory assumption" and not c["measurement_note"].strip():
        raise ValueError("Measured/published performance requires an evidence note or source citation.")
    return c


def simulate(config=None, detail=True):
    c = deepcopy(DEFAULTS)
    if config:
        c.update(config)
    validate_config(c)
    radio = radio_summary(c)
    a = simulate_network(c, radio, detail=detail, payload="a")
    b = simulate_network(c, radio, detail=detail, payload="b")
    cnn = analyze_cnn(c)
    energy = calculate_energy(c, radio, a, b, cnn)
    issues = list(cnn["issues"])
    for label, n in [("A", a), ("B", b)]:
        if n["outside_nodes"]:
            issues.append(
                f"Architecture {label}: {n['outside_nodes']:,} deployed nodes fail the modeled link-reliability requirement."
            )
        if n["capacity_rejected"]:
            issues.append(
                f"Architecture {label}: {n['capacity_rejected']:,} reachable nodes exceed reporting-window capacity."
            )
        if n["delivery_probability"] < c["delivery_target"]:
            issues.append(
                f"Architecture {label}: modeled delivery probability {n['delivery_probability']:.3f} is below target {c['delivery_target']:.3f}."
            )
        if energy[label.lower()]["active_s_day"] > 86400:
            issues.append(f"Architecture {label}: requested daily processing exceeds 24 hours.")
        if energy[label.lower()]["battery_days"] < c["battery_target_days"]:
            issues.append(
                f"Architecture {label}: battery lifetime is below the {c['battery_target_days']:g}-day target."
            )
    if c["tx_dbm"] > 17:
        issues.append(
            "High-power PA_BOOST operation needs manufacturer duty/thermal checks; +20 dBm is not continuous-operation authorization."
        )
    server_model_fits = cnn["serialized_kb"] / 1024 <= c["server_model_capacity_mb"]
    if not server_model_fits:
        issues.append("Model exceeds configured server model-storage budget.")
    eligible_a = (
        server_model_fits
        and a["delivery_probability"] >= c["delivery_target"]
        and energy["a"]["active_s_day"] <= 86400
        and energy["a"]["battery_days"] >= c["battery_target_days"]
    )
    eligible_b = (
        not cnn["issues"]
        and b["delivery_probability"] >= c["delivery_target"]
        and energy["b"]["active_s_day"] <= 86400
        and energy["b"]["battery_days"] >= c["battery_target_days"]
    )
    feasible_a = eligible_a and not a["outside_nodes"] and not a["capacity_rejected"]
    feasible_b = eligible_b and not b["outside_nodes"] and not b["capacity_rejected"]
    summary = {
        "frequency_mhz": c["frequency_mhz"],
        "sf": c["sf"],
        "tx_dbm": c["tx_dbm"],
        "range_m": radio["range_m"],
        "coverage_ha": radio["coverage_ha"],
        "reachable_nodes": a["reachable_nodes"],
        "served_a": a["served_nodes"],
        "served_b": b["served_nodes"],
        "capacity_a": a["capacity_nodes"],
        "capacity_b": b["capacity_nodes"],
        "battery_a_days": energy["a"]["battery_days"],
        "battery_b_days": energy["b"]["battery_days"],
        "node_a_daily_j": energy["a"]["battery_daily_j"],
        "node_b_daily_j": energy["b"]["battery_daily_j"],
        "system_a_kwh_day": energy["deployment_a"]["total_kwh_day"],
        "system_b_kwh_day": energy["deployment_b"]["total_kwh_day"],
        "cnn_mj": cnn["inference_energy_mj"],
        "parameters": cnn["parameters"],
        "gateways": c["gateways"],
        "inference_latency_ms": c["inference_latency_ms"],
        "inference_current_ma": c["inference_current_ma"],
        "trigger_probability": c["trigger_probability"],
        "area_ha": area_ha(c),
        "trees_per_ha": c["trees_per_ha"],
        "airtime_a_ms": radio["airtime_a_ms"],
        "airtime_b_ms": radio["airtime_b_ms"],
        "sensitivity_dbm": radio["sensitivity_dbm"],
        "tx_a_mj": energy["tx_a_mj"],
        "tx_b_mj": energy["tx_b_mj"],
        "break_even_ms": energy["break_even_ms"],
        "break_even_ma": energy["break_even_ma"],
        "tensor_arena_kb": c["tensor_arena_kb"],
        "cpu_mhz": c["cpu_mhz"],
        "macs": cnn["macs"],
        "weight_kb": cnn["weight_kb"],
        "estimated_latency_ms": cnn["estimated_latency_ms"],
        "ram_used_kb": cnn["ram_used_kb"],
        "feasible": feasible_b,
        "feasible_a": feasible_a,
        "feasible_b": feasible_b,
        "eligible_served_a": a["served_nodes"] if eligible_a else 0,
        "eligible_served_b": b["served_nodes"] if eligible_b else 0,
        "served_trees_a": a["served_trees"],
        "served_trees_b": b["served_trees"],
        "eligible_trees_a": a["served_trees"] if eligible_a else 0,
        "eligible_trees_b": b["served_trees"] if eligible_b else 0,
        "energy_per_served_tree_a_j": energy["deployment_a"]["total_kwh_day"] * 3.6e6 / a["served_trees"]
        if a["served_trees"]
        else float("inf"),
        "energy_per_served_tree_b_j": energy["deployment_b"]["total_kwh_day"] * 3.6e6 / b["served_trees"]
        if b["served_trees"]
        else float("inf"),
    }
    targets_failed = not feasible_a and not feasible_b
    result = {
        "config": c,
        "radio": radio,
        "network_a": a,
        "network_b": b,
        "cnn": cnn,
        "energy": energy,
        "feasibility": {
            "status": "Infeasible for full-deployment targets" if targets_failed else "Conditionally feasible",
            "issues": issues,
            "architecture_a": feasible_a,
            "architecture_b": feasible_b,
        },
        "summary": summary,
    }
    if detail:
        worst_case_config = c | {"trigger_probability": 1.0}
        result["worst_case"] = {
            "network_a": simulate_network(worst_case_config, radio, detail=False, payload="a"),
            "network_b": simulate_network(worst_case_config, radio, detail=False, payload="b"),
        }
    return result


def area_ha(c):
    return c["width_m"] * c["length_m"] / 10000 if c["dimensions_override"] else c["area_ha"]
