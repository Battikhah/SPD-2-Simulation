"""Checks for the simple user-facing battery and capacity estimates."""

import math

import pytest
from streamlit.testing.v1 import AppTest

from simulator.prototype import (
    REGULATOR_PROFILE,
    calculate_battery,
    calculate_cluster,
    calculate_sleep_components,
    compare_recording_durations,
    default_prototype_config,
)


def test_battery_profile_reproduces_user_supplied_full_event_case():
    result = calculate_battery(event_probability=1.0, airtime_s=0.338)
    assert math.isclose(result["sleep_breakdown"]["total_ma"], calculate_sleep_components()["total_battery_ma"])
    assert math.isclose(result["daily_mah"], 12.965864, rel_tol=1e-6)
    assert math.isclose(result["average_ma"], 0.540244, rel_tol=1e-6)
    component_result = result
    assert math.isclose(component_result["battery_days"], 160.4212419, rel_tol=1e-7)


def test_recording_duration_scales_recording_and_processing_but_not_detection_claims():
    durations = compare_recording_durations(1.0, calculate_sleep_components()["total_battery_ma"], 0.338, 0.1)
    assert [row["Recording (s)"] for row in durations] == [10, 30, 60, 120]
    assert [row["Listening (min/day)"] for row in durations] == [2, 6, 12, 24]
    assert [row["Detection performance"] for row in durations] == ["Not measured"] * 4
    assert math.isclose(durations[0]["Estimated life (days)"], 160.4212419, rel_tol=1e-7)
    assert math.isclose(durations[-1]["Estimated life (days)"], 19.8919022, rel_tol=1e-7)
    with pytest.raises(ValueError, match="positive"):
        calculate_battery(recording_s=0)


def test_previous_20ma_sleep_estimate_is_available_as_a_sensitivity_case():
    result = calculate_battery(event_probability=1.0, sleep_ma=20.0, airtime_s=0.338)
    assert math.isclose(result["battery_days"], 4.2547, rel_tol=1e-4)


def test_default_sleep_current_is_derived_from_component_estimate():
    result = calculate_battery()
    expected = calculate_sleep_components()["total_battery_ma"]
    assert math.isclose(result["sleep_breakdown"]["total_ma"], expected)
    assert math.isclose(result["sleep_breakdown"]["regulator_ma"], 0.1)


def test_explicit_sleep_current_override_remains_available_to_calculations():
    result = calculate_battery(sleep_ma=1.0)
    split = result["sleep_breakdown"]
    assert REGULATOR_PROFILE["name"] == "Pololu S7V8F3"
    assert math.isclose(split["regulator_ma"], 0.1)
    assert math.isclose(split["other_node_ma"], 0.9)
    assert math.isclose(split["regulator_ma"] + split["other_node_ma"], 1.0)
    assert math.isclose(result["daily_mah"], 33.975)

    override = calculate_battery(sleep_ma=1.0, regulator_iq_ma=0.2)
    assert math.isclose(override["daily_mah"], result["daily_mah"])
    assert math.isclose(override["sleep_breakdown"]["other_node_ma"], 0.8)


def test_regulator_idle_allowance_must_fit_within_total_sleep_budget():
    with pytest.raises(ValueError, match="cannot exceed"):
        calculate_battery(sleep_ma=0.05, regulator_iq_ma=0.1)


def test_no_event_packets_still_counts_recording_and_processing():
    quiet = calculate_battery(event_probability=0.0, sleep_ma=1.0)
    all_events = calculate_battery(event_probability=1.0, sleep_ma=1.0)
    assert quiet["daily_mah"] < all_events["daily_mah"]
    assert quiet["daily_mah"] > 0


def test_battery_inputs_reject_impossible_values():
    with pytest.raises(ValueError, match="probability"):
        calculate_battery(event_probability=1.1)
    with pytest.raises(ValueError, match="Sleep current"):
        calculate_battery(sleep_ma=-1)


def test_component_sleep_estimate_converts_rail_current_and_adds_explicit_allowances():
    result = calculate_sleep_components()
    assert math.isclose(result["rail_total_ua"], 11.2)
    assert math.isclose(result["rest_battery_ua"], 22.4864864865)
    assert math.isclose(result["total_battery_ua"], 122.4864864865)
    assert math.isclose(result["total_battery_ma"], 0.1224864864865)
    assert "not a low-load guarantee" in result["assumptions"]["efficiency"]


def test_component_sleep_rejects_invalid_efficiency_or_allowances():
    with pytest.raises(ValueError, match="efficiency"):
        calculate_sleep_components(efficiency=0)
    with pytest.raises(ValueError, match="nonnegative"):
        calculate_sleep_components(switched_leakage_ua=-1)


def test_cluster_limit_distinguishes_half_hour_planning_from_hourly_ceiling():
    result = calculate_cluster({"area_ha": 1.0, "trees_per_ha": 1, "gateways": 1}, 0.5)
    assert result["theoretical_per_head"] == 4000
    assert result["capacity_per_head"] == 2000
    assert result["deployed"] == 1
    assert result["reachable"] <= 1


@pytest.mark.parametrize("recording_s,acquisition_s,capacity", [(10, 17, 3981), (120, 182, 3797)])
def test_full_hour_reporting_window_reserves_acquisition_time(recording_s, acquisition_s, capacity):
    result = calculate_cluster({"area_ha": 1.0, "trees_per_ha": 1, "gateways": 1, "recording_s": recording_s}, 1.0)
    assert result["acquisition_s"] == acquisition_s
    assert result["reporting_window_s"] == 3600 - acquisition_s
    assert result["capacity_per_head"] == capacity
    assert result["theoretical_per_head"] == 4000


def test_slow_spreading_factor_increases_slot_and_reduces_capacity():
    base = calculate_cluster({"area_ha": 1.0, "trees_per_ha": 1, "gateways": 1, "sf": 7}, 0.5)
    slow = calculate_cluster({"area_ha": 1.0, "trees_per_ha": 1, "gateways": 1, "sf": 12}, 0.5)
    assert slow["slot_s"] > base["slot_s"]
    assert slow["theoretical_per_head"] < base["theoretical_per_head"]


def test_meeting_app_keeps_only_four_scenario_controls_and_baseline_values():
    app = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not app.exception
    assert any("Palm monitoring — project overview" in item.value for item in app.title)
    assert [item.value for item in app.subheader] == ["How it works", "Battery estimate", "Deployment limit"]
    assert {item.key for item in app.number_input} == {"meeting_area", "meeting_density", "meeting_heads"}
    assert [item.key for item in app.selectbox] == ["meeting_recording"]
    assert not app.radio and not app.slider and not app.checkbox and not app.dataframe
    assert not app.get("download_button")
    assert next(item.value for item in app.metric if item.label == "Estimated life") == "160 days"
    assert next(item.value for item in app.metric if item.label == "Sleep current") == "122.5 µA"
    assert next(item.value for item in app.metric if item.label == "Daily use") == "13.0 mAh"
    assert next(item.value for item in app.metric if item.label == "Nodes in scenario") == "10,000"
    assert next(item.value for item in app.metric if item.label == "Cluster heads") == "1"
    assert next(item.value for item in app.metric if item.label == "Planned slots per head/hour") == "2,000"
    assert "at least **5 cluster heads**" in " ".join(item.value for item in app.markdown)
    assert any("not measured" in item.value for item in app.caption)


def test_meeting_inputs_recalculate_deployment_and_battery():
    app = AppTest.from_file("../app.py", default_timeout=60).run()
    app.number_input(key="meeting_area").set_value(200.0).run()
    assert not app.exception
    assert next(item.value for item in app.metric if item.label == "Nodes in scenario") == "20,000"
    assert "at least **10 cluster heads**" in " ".join(item.value for item in app.markdown)

    app.selectbox(key="meeting_recording").select(120).run()
    assert not app.exception
    assert next(item.value for item in app.metric if item.label == "Estimated life") != "160 days"


def test_stale_engineering_session_state_cannot_change_meeting_assumptions():
    app = AppTest.from_file("../app.py", default_timeout=60)
    app.session_state["prototype_cfg"] = default_prototype_config() | {
        "sleep_ma": 2.5, "sleep_mode": "Manual battery estimate", "sf": 12,
        "trigger_probability": 0.0, "planning_fraction": 1.0,
    }
    app.session_state["show_advanced_tools"] = True
    app.run()
    assert not app.exception
    assert next(item.value for item in app.metric if item.label == "Sleep current") == "122.5 µA"
    assert next(item.value for item in app.metric if item.label == "Planned slots per head/hour") == "2,000"
    assert not app.radio and not app.checkbox
