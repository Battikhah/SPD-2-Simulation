import json
from copy import deepcopy

import pandas as pd
import pytest

from simulator.config import DEFAULTS, PRESETS
from simulator.engine import simulate, validate_config
from simulator.io import dumps_scenario, export_json, import_measurements, loads_scenario, report_markdown
from simulator.research import pareto_front, sweep


def cfg(**updates):
    return dict(deepcopy(DEFAULTS), **updates)


def test_cnn_current_cannot_change_rf_or_parameters():
    a, b = simulate(cfg()), simulate(cfg(inference_current_ma=120.0))
    assert a["radio"] == b["radio"]
    assert a["cnn"]["parameters"] == b["cnn"]["parameters"]
    assert a["energy"]["b"]["daily_j"] < b["energy"]["b"]["daily_j"]


def test_density_changes_population_not_propagation():
    a, b = simulate(cfg()), simulate(cfg(trees_per_ha=200.0))
    assert a["radio"] == b["radio"]
    assert b["network_a"]["total_trees"] == 2 * a["network_a"]["total_trees"]


@pytest.mark.parametrize(
    "updates",
    [
        {"trigger_probability": float("nan")},
        {"battery_mah": -1},
        {"sf": 6, "implicit_header": False},
        {"payload_a_bytes": 256},
        {"bandwidth_khz": 62.5, "frequency_mhz": 915},
        {"dram_kb": 520},
        {"custom_trees": [[1, float("inf")]]},
        {"gateways": 0},
        {"regulator_efficiency": 0},
        {"retries": 1.5},
    ],
)
def test_invalid_configuration_is_rejected(updates):
    with pytest.raises(ValueError):
        validate_config(cfg(**updates))


def test_scenario_roundtrip_and_export_are_strict_json():
    c = cfg()
    assert loads_scenario(dumps_scenario(c)) == c
    data = json.loads(export_json(simulate(c)))
    assert data["config"]["sf"] == 7
    assert data["provenance"]["hardware_verified"] is False
    assert "Cluster-head reporting capacity" in report_markdown(simulate(c, detail=False))


def test_purchased_hardware_defaults_schedule_and_inventory_are_explicit():
    from simulator.hardware import HARDWARE_BASELINE

    c = cfg()
    assert c["frequency_mhz"] == 433
    assert c["battery_v"] == 3.7
    assert c["battery_mah"] == 2600
    assert c["recording_hours"] == [0, 1, 2, 3, 4, 5, 6, 10, 14, 18, 22, 23]
    assert next(iter(PRESETS)) == "Purchased hardware — SX1278 433 MHz"
    assert PRESETS[next(iter(PRESETS))] == {}
    assert PRESETS["Original Senior Design — RFM95W/SX1276, 915 MHz, SF7"] == {
        "frequency_mhz": 915,
        "battery_mah": 3000.0,
    }
    assert HARDWARE_BASELINE["reported_on"] == "2026-10-28"
    no_self_discharge = c | {"self_discharge_monthly": 0.0}
    assert simulate(no_self_discharge)["energy"]["a"]["battery_days"] == pytest.approx(
        simulate(no_self_discharge | {"battery_mah": 7800})["energy"]["a"]["battery_days"] / 3
    )
    assert HARDWARE_BASELINE["battery"]["model"] == "505573"
    assert HARDWARE_BASELINE["battery"]["quantity"] == 3
    assert "unknowns" in HARDWARE_BASELINE
    output = export_json(simulate(c))
    assert "XL6009E1" in output and "MCP602E" in output and "unknowns" in output
    report = __import__("simulator.io", fromlist=["report_markdown"]).report_markdown(simulate(c))
    assert "SX1278" in report and "505573" in report


@pytest.mark.parametrize(
    "updates",
    [
        {"recording_hours": [0] * 12},
        {"recording_hours": [0, 1, 2]},
        {"recording_hours": [24] + list(range(1, 12))},
        {"recording_hours": [0.5] + list(range(1, 12))},
        {"filter_low_hz": 0, "filter_high_hz": 2500},
        {"filter_low_hz": 2500, "filter_high_hz": 100},
        {"filter_high_hz": 4001},
    ],
)
def test_invalid_recording_schedule_or_filter_is_rejected(updates):
    with pytest.raises(ValueError):
        validate_config(cfg(**updates))


def test_legacy_scenario_custom_round_count_has_unknown_timestamps():
    legacy = {key: value for key, value in DEFAULTS.items() if key != "recording_hours"}
    legacy["recordings_per_day"] = 5
    loaded = loads_scenario(json.dumps(legacy))
    assert loaded["recordings_per_day"] == 5
    assert loaded["recording_hours"] == []


def test_import_cannot_claim_measurement_without_evidence():
    with pytest.raises(ValueError):
        loads_scenario('{"sf": 13}')
    c = import_measurements(cfg(), "Inference", pd.DataFrame({"latency_ms": [100, 200], "current_ma": [50, 70]}))
    assert c["inference_latency_ms"] == 150
    assert c["performance_source"] == "Measured"
    assert "import" in c["measurement_note"].lower()


def test_sweep_reproducible_and_extracts_nondominated_rows():
    space = {"sf": [7, 9], "inference_latency_ms": [200.0, 800.0]}
    a = sweep(cfg(area_ha=1.0), space, "Full factorial", 10)
    assert len(a) == 4
    pd.testing.assert_frame_equal(sweep(cfg(area_ha=1), space, "Random", 4), sweep(cfg(area_ha=1), space, "Random", 4))
    front = pareto_front(a)
    assert len(front) > 0
    assert set(front.index).issubset(a.index)


@pytest.mark.parametrize("count", [8, 24])
def test_daily_count_sweep_clears_incompatible_exact_hours(count):
    result = sweep(cfg(area_ha=1), {"recordings_per_day": [count]})
    assert result.iloc[0].error == ""
    assert result.iloc[0].recordings_per_day == count


def test_sweep_keeps_and_validates_explicit_schedule_overrides():
    result = sweep(cfg(area_ha=1), {"recordings_per_day": [8], "recording_hours": [DEFAULTS["recording_hours"]]})
    assert "match recordings/day" in result.iloc[0].error


def test_current_profile_cannot_hide_sweep_overrides():
    c = cfg(
        area_ha=1,
        current_profile=[{"time_ms": 0, "current_ma": 100}, {"time_ms": 100, "current_ma": 100}],
        performance_source="Measured",
        measurement_note="test fixture, synthetic",
    )
    data = sweep(c, {"inference_latency_ms": [200.0, 800.0], "inference_current_ma": [30.0, 100.0]})
    assert sorted(data.cnn_mj.tolist()) == pytest.approx(sorted([19.8, 66.0, 79.2, 264.0]))


def test_zero_delivery_is_not_feasible():
    r = simulate(cfg(area_ha=0.01, propagation="Free-space", packet_success=0, battery_target_days=0))
    assert not r["feasibility"]["architecture_a"]
    assert not r["feasibility"]["architecture_b"]
    assert any("delivery" in issue for issue in r["feasibility"]["issues"])


def test_server_storage_failure_affects_a_not_edge_model():
    r = simulate(cfg(area_ha=0.01, propagation="Free-space", server_model_capacity_mb=0.001, battery_target_days=0))
    assert not r["feasibility"]["architecture_a"]
    assert r["feasibility"]["architecture_b"]


def test_one_tree_expected_daily_traffic_is_not_zero_by_seed():
    r = simulate(cfg(area_ha=1, trees_per_ha=1, propagation="Free-space"))
    assert r["energy"]["deployment_a"]["requests_day"] > 0


def test_aggregate_energy_counts_admitted_batteries_before_delivery_loss():
    r = simulate(cfg(area_ha=1, trees_per_ha=1, propagation="Free-space", packet_success=0.5))
    assert r["network_a"]["admitted_nodes"] == 1
    assert r["network_a"]["served_nodes"] == 0.5
    assert r["energy"]["deployment_a"]["nodes_kwh_day"] == pytest.approx(r["energy"]["a"]["battery_daily_j"] / 3.6e6)


def test_full_trigger_worst_case_is_reported_without_changing_current_scenario():
    c = cfg(area_ha=1, deployed_limit=5000, trigger_probability=0.25)
    r = simulate(c, detail=True)
    assert r["config"]["trigger_probability"] == 0.25
    assert r["worst_case"]["network_a"]["reachable_nodes"] == r["network_a"]["reachable_nodes"]
    assert r["worst_case"]["network_a"]["scheduled_nodes"] >= r["network_a"]["scheduled_nodes"]
    assert r["worst_case"]["network_a"]["capacity_rejected"] >= r["network_a"]["capacity_rejected"]
    assert "worst_case" not in simulate(c, detail=False)


def test_dynamic_gateway_capacity_separates_expected_and_all_nodes_reporting_load():
    c = cfg(
        area_ha=10,
        nodes_per_tree=20,
        deployed_limit=5000,
        window_s=900,
        propagation="Free-space",
        scheduling="Dynamic TDMA",
        trigger_probability=0.25,
        layout="Custom",
        custom_trees=[[158, 158]] * 250,
    )
    r = simulate(c, detail=True)
    current, worst = r["network_a"], r["worst_case"]["network_a"]
    assert current["gateway_load"].expected_offered_requests.sum() == pytest.approx(1250)
    assert worst["gateway_load"].expected_offered_requests.sum() == 5000
    assert current["gateway_load"].admitted_population.sum() == pytest.approx(current["admitted_nodes"])
    assert current["gateway_load"].scheduled_requests_sample.sum() == current["scheduled_nodes"]
    assert worst["admitted_nodes"] == worst["capacity_nodes"]
    assert 0 < worst["capacity_rejected"] < 5000
    assert worst["admitted_nodes"] + worst["capacity_rejected"] == 5000
