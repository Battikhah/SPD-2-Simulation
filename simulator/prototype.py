"""Small, explicit battery and cluster-capacity calculations for the default view."""

from copy import deepcopy
from math import floor, isfinite

from .config import DEFAULTS
from .engine import validate_config
from .network import simulate_network
from .radio import radio_summary

BATTERY_PROFILE = {
    "wake_ma": 180.0,
    "wake_s": 2.0,
    "record_ma": 160.0,
    "record_s": 10.0,
    "process_ma": 180.0,
    "process_s": 5.0,
    "transmit_ma": 300.0,
    "transmit_s_min": 0.5,
    "cycles_per_day": 12,
    "battery_mah": 2600.0,
    "usable_fraction": 0.8,
    "voltage_v": 3.7,
}

REGULATOR_PROFILE = {
    "name": "Pololu S7V8F3",
    "status": "Planned; not listed in purchased hardware",
    "output_v": 3.3,
    "input_min_v": 2.7,
    "input_max_v": 11.8,
    "enabled_no_load_max_ma": 0.1,
    "source": "https://www.pololu.com/product/2122/specs",
}

SLEEP_COMPONENTS = {
    "esp32_chip_ua": 10.0,
    "esp32_source": "Espressif ESP32 datasheet, table 4-2 (chip deep-sleep RTC timer); not a full module or development-board guarantee: https://documentation.espressif.com/esp32_datasheet_en.html",
    "sx1278_ua": 0.2,
    "sx1278_source": "SX1278 datasheet, table 2.5.1, typical sleep current; maximum listed as 1 µA: https://uelectronics.com/wp-content/uploads/2026/01/SX1278-Datasheet.pdf",
    "regulator_source": REGULATOR_PROFILE["source"],
}


def calculate_sleep_components(efficiency=0.8, switched_leakage_ua=1.0, other_rail_ua=0.0, other_battery_ua=10.0, regulator_iq_ma=0.1):
    """Estimate sleep current from powered components plus explicit unknown allowances."""
    efficiency, switched_leakage_ua, other_rail_ua, other_battery_ua, regulator_iq_ma = map(
        float, (efficiency, switched_leakage_ua, other_rail_ua, other_battery_ua, regulator_iq_ma)
    )
    if not isfinite(efficiency) or not 0 < efficiency <= 1:
        raise ValueError("Sleep conversion efficiency must be greater than 0 and at most 1.")
    if any(not isfinite(v) or v < 0 for v in (switched_leakage_ua, other_rail_ua, other_battery_ua, regulator_iq_ma)):
        raise ValueError("Sleep component currents must be finite and nonnegative.")
    rail_components = {
        "ESP32 deep-sleep chip reference": {"current_ua": SLEEP_COMPONENTS["esp32_chip_ua"], "basis": SLEEP_COMPONENTS["esp32_source"]},
        "SX1278 sleep reference": {"current_ua": SLEEP_COMPONENTS["sx1278_ua"], "basis": SLEEP_COMPONENTS["sx1278_source"]},
        "Switched sensor/amplifier branch leakage allowance": {"current_ua": switched_leakage_ua, "basis": "Exploratory allowance; switch leakage and signal backfeed depend on the final circuit and must be measured."},
        "Other 3.3 V rail loads": {"current_ua": other_rail_ua, "basis": "Editable, unsourced rail-current allowance. Default zero assumes a custom circuit without USB, LEDs or a development-board regulator; exact board/flash loads are unknown."},
    }
    rail_ua = sum(row["current_ua"] for row in rail_components.values())
    rail_to_battery = REGULATOR_PROFILE["output_v"] / BATTERY_PROFILE["voltage_v"] / efficiency
    rest_battery_ua = rail_ua * rail_to_battery + other_battery_ua
    total_battery_ua = rest_battery_ua + regulator_iq_ma * 1000
    return {
        "rail_components": rail_components,
        "rail_total_ua": rail_ua,
        "rail_to_battery_factor": rail_to_battery,
        "other_battery_ua": other_battery_ua,
        "regulator_ua": regulator_iq_ma * 1000,
        "rest_battery_ua": rest_battery_ua,
        "total_battery_ua": total_battery_ua,
        "total_battery_ma": total_battery_ua / 1000,
        "efficiency": efficiency,
        "formula": "I_batt = regulator Iq + (I_3.3V × 3.3 V / (3.7 V × efficiency)) + other battery-side current",
        "assumptions": {
            "efficiency": f"{efficiency * 100:g}% modeling assumption for sleep rail load, excluding separately added regulator Iq; not a low-load guarantee.",
            "other_battery": f"{other_battery_ua:g} µA unsourced exploratory allowance for unknown protection, charging-board, divider and other battery-path leakage.",
        },
    }


def calculate_battery(event_probability=1.0, sleep_ma=None, airtime_s=0.338, regulator_iq_ma=0.1, *, recording_s=10.0):
    """Estimate battery-side daily use from the supplied complete-node current profile.

    Wake, record, and processing are charged on every scheduled cycle; transmission
    is charged at the event probability. All currents are estimates at battery input.
    """
    p = float(event_probability)
    sleep_ma = calculate_sleep_components(regulator_iq_ma=regulator_iq_ma)["total_battery_ma"] if sleep_ma is None else float(sleep_ma)
    airtime_s = float(airtime_s)
    regulator_iq_ma = float(regulator_iq_ma)
    recording_s = float(recording_s)
    if not isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Event probability must be between 0 and 1.")
    if not isfinite(sleep_ma) or sleep_ma < 0:
        raise ValueError("Sleep current must be a finite, nonnegative value.")
    if not isfinite(airtime_s) or airtime_s < 0:
        raise ValueError("Radio airtime must be a finite, nonnegative value.")
    if not isfinite(regulator_iq_ma) or regulator_iq_ma < 0:
        raise ValueError("Regulator idle allowance must be a finite, nonnegative value.")
    if not isfinite(recording_s) or recording_s <= 0:
        raise ValueError("Recording duration must be a finite, positive value.")
    if regulator_iq_ma > sleep_ma:
        raise ValueError("Regulator idle allowance cannot exceed the whole-node sleep-current budget.")
    profile = BATTERY_PROFILE
    tx_s = max(profile["transmit_s_min"], airtime_s)
    processing_s = profile["process_s"] * recording_s / profile["record_s"]
    active_s = profile["cycles_per_day"] * (profile["wake_s"] + recording_s + processing_s + p * tx_s)
    if active_s > 86400:
        raise ValueError("The selected daily active work exceeds 24 hours.")
    stages = [
        ("Sleep", 86400 - active_s, sleep_ma),
        ("Wake and initialize", profile["cycles_per_day"] * profile["wake_s"], profile["wake_ma"]),
        ("Record vibration", profile["cycles_per_day"] * recording_s, profile["record_ma"]),
        ("Filter and extract features", profile["cycles_per_day"] * processing_s, profile["process_ma"]),
        ("LoRa transmit", profile["cycles_per_day"] * p * tx_s, profile["transmit_ma"]),
    ]
    daily_mah = sum(seconds * current / 3600 for _, seconds, current in stages)
    usable_mah = profile["battery_mah"] * profile["usable_fraction"]
    return {
        "stages": stages,
        "sleep_breakdown": {
            "total_ma": sleep_ma,
            "regulator_ma": regulator_iq_ma,
            "other_node_ma": sleep_ma - regulator_iq_ma,
            "duration_s_day": 86400 - active_s,
            "regulator_mah_day": (86400 - active_s) * regulator_iq_ma / 3600,
            "other_node_mah_day": (86400 - active_s) * (sleep_ma - regulator_iq_ma) / 3600,
        },
        "daily_mah": daily_mah,
        "average_ma": daily_mah / 24,
        "battery_days": usable_mah / daily_mah if daily_mah else float("inf"),
        "usable_mah": usable_mah,
        "tx_duration_s": tx_s,
        "recording_s": recording_s,
        "processing_s": processing_s,
    }


def compare_recording_durations(event_probability, sleep_ma, airtime_s, regulator_iq_ma, durations=(10, 30, 60, 120)):
    """Compare battery/workload projections without asserting detection performance."""
    rows = []
    for duration in durations:
        result = calculate_battery(
            event_probability,
            sleep_ma,
            airtime_s,
            regulator_iq_ma,
            recording_s=duration,
        )
        recording_stage = next(stage for stage in result["stages"] if stage[0] == "Record vibration")
        recording_mah = recording_stage[1] * recording_stage[2] / 3600
        rows.append({
            "Recording (s)": duration,
            "Listening (min/day)": BATTERY_PROFILE["cycles_per_day"] * duration / 60,
            "Daily battery use (mAh)": result["daily_mah"],
            "Estimated life (days)": result["battery_days"],
            "Recording energy (% daily)": 100 * recording_mah / result["daily_mah"],
            "Detection performance": "Not measured",
        })
    return rows


def calculate_cluster(config=None, planning_fraction=0.5):
    """Reuse the RF/network engine with half the hourly window reserved by default."""
    c = deepcopy(DEFAULTS)
    event_probability = config.get("trigger_probability", 1.0) if config else 1.0
    if config:
        c.update(config)
    fraction = min(1.0, max(0.01, float(planning_fraction)))
    hourly_window_s = 3600.0
    recording_s = float(c.get("recording_s", BATTERY_PROFILE["record_s"]))
    if not isfinite(recording_s) or recording_s <= 0:
        raise ValueError("Recording duration must be a finite, positive value.")
    # One node's wake, recording and feature processing finish before its TDMA report.
    acquisition_s = BATTERY_PROFILE["wake_s"] + recording_s + BATTERY_PROFILE["process_s"] * recording_s / BATTERY_PROFILE["record_s"]
    reporting_window_s = min(hourly_window_s * fraction, hourly_window_s - acquisition_s)
    if reporting_window_s <= 0:
        raise ValueError("Recording and processing leave no time for reporting in the hourly deadline.")
    c.update(
        frequency_mhz=433,
        payload_a_bytes=214,
        payload_b_bytes=214,
        tx_dbm=20,
        sf=int(c.get("sf", 7)),
        bandwidth_khz=125.0,
        coding_rate=5,
        preamble_symbols=8,
        implicit_header=False,
        crc=True,
        scheduling="Fixed TDMA",
        slot_ms=900.0,
        guard_ms=20.0,
        clock_error_ms=5.0,
        channels=1,
        receivers=1,
        retries=0,
        ack=False,
        trigger_probability=float(event_probability),
        window_s=reporting_window_s,
        delivery_target=0.95,
    )
    c.pop("planning_fraction", None)
    c.pop("sleep_ma", None)
    c.pop("regulator_iq_ma", None)
    for key in ("sleep_mode", "sleep_efficiency", "switched_leakage_ua", "other_rail_ua", "other_battery_ua"):
        c.pop(key, None)
    validate_config(c)
    radio = radio_summary(c)
    network = simulate_network(c, radio, detail=False, payload="a")
    theoretical = floor(hourly_window_s / network["slot_s"] + 1e-10)
    return {
        "deployed": network["deployed_nodes"],
        "reachable": network["reachable_nodes"],
        "admitted": int(network["admitted_nodes"]),
        "served": network["served_nodes"],
        "capacity_per_head": network["capacity_nodes"] // int(c["gateways"]),
        "theoretical_per_head": theoretical,
        "expected_event_packets": network["expected_successful_transmissions"] * BATTERY_PROFILE["cycles_per_day"],
        "delivery_probability": network["delivery_probability"],
        "airtime_ms": radio["airtime_a_ms"],
        "slot_s": network["slot_s"],
        "acquisition_s": acquisition_s,
        "reporting_window_s": reporting_window_s,
        "config": c,
        "network": network,
        "radio": radio,
    }


def default_prototype_config():
    return {
        "area_ha": 100.0,
        "trees_per_ha": 100,
        "gateways": 1,
        "sf": 7,
        "trigger_probability": 1.0,
        "planning_fraction": 0.5,
        "recording_s": BATTERY_PROFILE["record_s"],
        "regulator_iq_ma": REGULATOR_PROFILE["enabled_no_load_max_ma"],
        "sleep_efficiency": 0.8,
        "switched_leakage_ua": 1.0,
        "other_rail_ua": 0.0,
        "other_battery_ua": 10.0,
    }
