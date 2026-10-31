"""Regulated-rail stage accounting and battery-side conversion, in SI units."""

from math import inf, log1p

import pandas as pd


def _stage(c, name, ms, ma):
    return {"stage": name, "duration_ms": ms, "current_ma": ma, "energy_mj": c["voltage_v"] * ma * ms / 1000}


def _lifetime(c, consumption_j):
    capacity_j = (
        c["battery_v"] * c["battery_mah"] * 3.6 * c["usable_fraction"] * c["temperature_factor"] * c["aging_factor"]
    )
    net_j = consumption_j - c["solar_wh_day"] * 3600
    if net_j <= 0:
        return inf
    k = -log1p(-c["self_discharge_monthly"]) / 30
    return log1p(k * capacity_j / net_j) / k if k else capacity_j / net_j


def calculate_energy(c, radio, net_a, net_b, cnn):
    shared = [
        _stage(c, "Wake", c["wake_ms"], c["wake_current_ma"]),
        _stage(c, "Analog front end", c["afe_ms"], c["afe_current_ma"]),
        _stage(c, "Record", c["recording_s"] * 1000, c["recording_current_ma"]),
        _stage(c, "Bandpass / SNR", c["preprocessing_ms"], c["preprocessing_current_ma"]),
    ]
    rejected = sum(s["energy_mj"] for s in shared)
    p, rounds = c["trigger_probability"], c["recordings_per_day"]
    result = {"cnn_mj": cnn["inference_energy_mj"]}
    for label, net in [("a", net_a), ("b", net_b)]:
        attempts = net["attempts_per_event"]
        tx_ms = radio[f"airtime_{label}_ms"]
        tx_current = radio["tx_current_ma"] + c["mcu_tx_ma"]
        stages = shared + [_stage(c, "MFCC", c["mfcc_ms"], c["mfcc_current_ma"])]
        if label == "b":
            stages.append(
                {
                    "stage": "CNN",
                    "duration_ms": cnn["effective_latency_ms"],
                    "current_ma": cnn["effective_current_ma"],
                    "energy_mj": cnn["inference_energy_mj"],
                }
            )
        processing_mj = sum(s["energy_mj"] for s in stages)
        processing_ms = sum(s["duration_ms"] for s in stages)
        stages += [
            _stage(c, "Radio startup", c["radio_startup_ms"] * attempts, c["mcu_tx_ma"] + c["radio_standby_ma"]),
            _stage(c, "LoRa TX", tx_ms * attempts, tx_current),
        ]
        if c["ack"]:
            stages.append(
                _stage(c, "ACK receive window", c["ack_wait_ms"] * attempts, c["mcu_tx_ma"] + c["radio_rx_ma"])
            )
        accepted = sum(s["energy_mj"] for s in stages)
        accepted_ms = sum(s["duration_ms"] for s in stages)
        rejected_ms = sum(s["duration_ms"] for s in shared)
        active_s = rounds * (p * accepted_ms + (1 - p) * rejected_ms) / 1000
        sleep_ms = max(0, 86400 - active_s) * 1000
        sleep_ma = c["sleep_current_ma"] + c["radio_sleep_ma"]
        sleep_mj = c["voltage_v"] * sleep_ma * sleep_ms / 1000
        daily_j = (rounds * (p * accepted + (1 - p) * rejected) + sleep_mj) / 1000
        battery_j = daily_j / c["regulator_efficiency"] + c["battery_v"] * c["quiescent_ma"] * 86.4
        cycle_mj = p * accepted + (1 - p) * rejected
        tx_energy = c["voltage_v"] * tx_current * tx_ms / 1000
        result[f"tx_{label}_mj"] = tx_energy
        result[label] = {
            "accepted_mj": accepted,
            "rejected_mj": rejected,
            "cycle_mj": cycle_mj,
            "daily_j": daily_j,
            "monthly_j": daily_j * 30,
            "annual_j": daily_j * 365,
            "battery_daily_j": battery_j,
            "battery_days": _lifetime(c, battery_j),
            "average_mw": daily_j / 86.4,
            "battery_current_ma": battery_j / (c["battery_v"] * 86.4),
            "active_s_day": active_s,
            "sleep_s_day": max(0, 86400 - active_s),
            "stages": pd.DataFrame(stages),
            "sleep_energy_j": sleep_mj / 1000,
            "performance_source": c["performance_source"],
        }
        served = net["served_nodes"]
        scheduled = net.get("admitted_nodes", net.get("scheduled_nodes", served))
        deployed = net["deployed_nodes"]
        # Every deployed node still wakes and processes; only admitted nodes expend scheduled TX energy.
        base_active_s = rounds * (p * processing_ms + (1 - p) * rejected_ms) / 1000
        base_j = (
            rounds * (p * processing_mj + (1 - p) * rejected)
            + c["voltage_v"] * sleep_ma * max(0, 86400 - base_active_s)
        ) / 1000
        all_rail_j = deployed * base_j + scheduled * (daily_j - base_j)
        all_battery_j = all_rail_j / c["regulator_efficiency"] + deployed * c["battery_v"] * c["quiescent_ma"] * 86.4
        requests = net.get("expected_successful_transmissions", served * p) * rounds
        gateway_j = (
            c["gateways"] * (c["gateway_idle_w"] + c["voltage_v"] * c["radio_rx_ma"] / 1000 * c["receivers"]) * 86400
        )
        gateway_j += requests * (c["backhaul_j_packet"] + c["gateway_process_w"] * tx_ms / 1000)
        server_j = c["server_idle_w"] * 86400
        if label == "a":
            server_j += requests * c["server_inference_w"] * c["server_inference_ms"] / 1000
        total_j = all_battery_j + gateway_j + server_j
        result[f"deployment_{label}"] = {
            "nodes_kwh_day": scheduled * battery_j / 3.6e6,
            "all_deployed_kwh_day": all_battery_j / 3.6e6,
            "gateway_kwh_day": gateway_j / 3.6e6,
            "server_kwh_day": server_j / 3.6e6,
            "total_kwh_day": total_j / 3.6e6,
            "requests_day": requests,
            "cost": deployed * c["node_cost"] + c["gateways"] * c["gateway_cost"],
            "electricity_cost_year": total_j / 3.6e6 * 365 * c["electricity_cost_kwh"],
        }
    saving = result["tx_a_mj"] - result["tx_b_mj"]
    result.update(
        tx_saving_mj=saving,
        net_saving_mj=saving - result["cnn_mj"],
        break_even_ms=max(0, saving) * 1000 / (c["voltage_v"] * cnn["effective_current_ma"]),
        break_even_ma=max(0, saving) * 1000 / (c["voltage_v"] * cnn["effective_latency_ms"]),
    )
    result["net_cycle_saving_mj"] = result["a"]["cycle_mj"] - result["b"]["cycle_mj"]
    result["break_even_note"] = (
        "Core single-attempt TX saving includes MCU TX current; full cycle also includes startup, ACKs and retries."
    )
    result["server_scale"] = pd.DataFrame(
        [
            {
                "nodes": n,
                "requests_day": n * p * rounds,
                "inference_j_day": n * p * rounds * c["server_inference_w"] * c["server_inference_ms"] / 1000,
            }
            for n in (100, 1000, 8000)
        ]
    )
    return result
