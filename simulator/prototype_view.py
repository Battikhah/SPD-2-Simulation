"""Minimal meeting view for the proposed palm-monitoring design."""

from math import ceil

import streamlit as st

from .prototype import calculate_battery, calculate_cluster, calculate_sleep_components


def render():
    if "meeting_scenario" not in st.session_state:
        st.session_state.meeting_scenario = {
            "area_ha": 100.0,
            "trees_per_ha": 100,
            "gateways": 1,
            "recording_s": 10,
        }
    scenario = st.session_state.meeting_scenario

    st.set_page_config(page_title="Palm monitoring project overview", page_icon="🌴", layout="centered")
    st.title("Palm monitoring — project overview")
    st.caption("Proposed design; calculations are estimates, not measured results.")

    with st.expander("Scenario", expanded=False):
        a, b = st.columns(2)
        scenario["area_ha"] = a.number_input(
            "Plantation area (hectares)", min_value=0.1, max_value=2500.0,
            value=float(scenario["area_ha"]), step=1.0, key="meeting_area",
        )
        scenario["trees_per_ha"] = b.number_input(
            "Trees per hectare", min_value=1, max_value=100,
            value=int(scenario["trees_per_ha"]), step=1, key="meeting_density",
        )
        c, d = st.columns(2)
        scenario["gateways"] = c.number_input(
            "Cluster heads", min_value=1, max_value=20,
            value=int(scenario["gateways"]), step=1, key="meeting_heads",
        )
        scenario["recording_s"] = d.selectbox(
            "Recording length", [10, 30, 60, 120],
            index=[10, 30, 60, 120].index(int(scenario["recording_s"])),
            format_func=lambda seconds: f"{seconds} seconds", key="meeting_recording",
        )

    # Fixed assumptions keep hidden research or engineering settings out of this view.
    deployed = int(round(scenario["area_ha"] * scenario["trees_per_ha"]))
    config = {
        "area_ha": scenario["area_ha"],
        "trees_per_ha": scenario["trees_per_ha"],
        "gateways": scenario["gateways"],
        "recording_s": scenario["recording_s"],
        "sf": 7,
        "trigger_probability": 1.0,
    }
    sleep = calculate_sleep_components()
    cluster = calculate_cluster(config, planning_fraction=0.5)
    battery = calculate_battery(
        event_probability=1.0,
        sleep_ma=sleep["total_battery_ma"],
        airtime_s=cluster["airtime_ms"] / 1000,
        regulator_iq_ma=0.1,
        recording_s=scenario["recording_s"],
    )
    nodes_per_head = cluster["capacity_per_head"]
    minimum_heads = ceil(deployed / nodes_per_head) if nodes_per_head else 0

    st.subheader("How it works")
    st.write("Piezo sensor → ESP32 extracts vibration features → SX1278 LoRa (433 MHz) sends event packets → cluster head → cloud classification.")
    st.caption(
        f"Each node records {scenario['recording_s']} seconds 12 times daily, sleeps between recordings, "
        "and transmits only when triggered. The design assumes cloud classification."
    )
    st.caption("Assumption: one sensing node per palm, powered by a 3.7 V, 2,600 mAh LiPo. A Pololu regulator is planned, not purchased, for the 3.3 V rail.")

    st.subheader("Battery estimate")
    life, sleep_current, daily_use = st.columns(3)
    life.metric("Estimated life", f"{battery['battery_days']:.0f} days")
    sleep_current.metric("Sleep current", f"{sleep['total_battery_ua']:.1f} µA")
    daily_use.metric("Daily use", f"{battery['daily_mah']:.1f} mAh")
    st.caption("Assumes a final custom circuit with the sensor/amplifier switched off between recordings, 12 event transmissions/day, and 80% usable capacity. Current draw still needs measurement.")
    if scenario["recording_s"] != 10:
        st.caption("Processing time is estimated; battery life needs measurement.")

    st.subheader("Deployment limit")
    nodes_col, heads_col, slots_col = st.columns(3)
    nodes_col.metric("Nodes in scenario", f"{deployed:,}")
    heads_col.metric("Cluster heads", f"{scenario['gateways']:,}")
    slots_col.metric("Planned slots per head/hour", f"{nodes_per_head:,}")
    st.write(f"With one node per palm, the timing model needs at least **{minimum_heads} cluster heads** for {deployed:,} nodes. Selected: {scenario['gateways']}.")
    st.caption("This is a timing limit only. Actual radio coverage and pest detection still need testing.")
    st.caption("Next validation: measure node current and compare exciter playback with the piezo recording.")

    return True
