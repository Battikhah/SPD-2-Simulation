"""Exercise the live dashboard with the real simulation engine."""

from streamlit.testing.v1 import AppTest

from simulator.config import DEFAULTS
from simulator.io import dumps_scenario


def advanced_app():
    """Open the retained detailed dashboard for its compatibility checks."""
    return AppTest.from_file("../engineering_app.py", default_timeout=30).run()


def test_dashboard_smoke_and_spreading_factor_recalculates_results():
    app = advanced_app()
    assert not app.exception
    assert app.session_state["cfg"]["frequency_mhz"] == 433
    assert app.session_state["cfg"]["battery_mah"] == 2600
    assert next(metric for metric in app.metric if metric.label == "Selected module").value == "SX1278"
    before = next(metric for metric in app.metric if metric.label == "Reachable nodes").value

    app.selectbox(key="sf").set_value(12).run()
    assert not app.exception
    assert next(widget for widget in app.selectbox if widget.key == "sf").value == 12
    after = next(metric for metric in app.metric if metric.label == "Reachable nodes").value
    assert after != before


def test_overview_explains_transmission_cadence_and_current_node_battery():
    app = advanced_app()
    assert not app.exception
    overview = " ".join(item.value for item in app.markdown) + " ".join(item.value for item in app.caption) + " ".join(item.value for item in app.subheader)
    assert "intermittently, not continuously" in overview
    assert "12 opportunities/day" in overview
    assert "3.0 expected initial reports/day per admitted node" in overview
    assert "3.7 V × 2600 mAh = 9.62 Wh nominal" in overview
    assert "505573" in overview
    assert "22:00–06:00 hourly" in overview
    assert "06:00–22:00 every 4 h" in overview
    assert "MCP602E" in overview
    assert "exploratory placeholders" in overview
    assert "Cluster-head reporting capacity" in overview
    assert "one report per node per day" in overview
    assert any("Worst-case rejected nodes/round" in frame.value.columns for frame in app.dataframe)

    app.slider(key="trigger_probability").set_value(0.0).run()
    assert not app.exception
    overview = " ".join(item.value for item in app.markdown) + " ".join(item.value for item in app.caption)
    assert "0.0 expected initial reports/day per admitted node" in overview

    app.slider(key="trigger_probability").set_value(1.0).run()
    assert not app.exception
    overview = " ".join(item.value for item in app.markdown) + " ".join(item.value for item in app.caption)
    assert "12.0 expected initial reports/day per admitted node" in overview

    scenario = DEFAULTS | {"battery_v": 3.0, "battery_mah": 2000.0}
    next(widget for widget in app.radio if widget.label == "Mode").set_value("Engineering").run()
    app.file_uploader(key="scenario_upload").set_value(
        ("scenario.json", dumps_scenario(scenario).encode(), "application/json")
    ).run()
    next(widget for widget in app.button if widget.label == "Load uploaded scenario").click().run()
    assert not app.exception
    assert app.session_state["cfg"]["battery_v"] == 3.0
    assert app.session_state["cfg"]["battery_mah"] == 2000.0
    overview = " ".join(item.value for item in app.markdown) + " ".join(item.value for item in app.caption)
    assert "3 V × 2000 mAh = 6.00 Wh nominal" in overview


def test_network_tab_switches_between_current_and_worst_case_load():
    app = advanced_app()
    assert not app.exception
    app.radio(key="network_request_load").set_value("Worst case · all reachable report").run()
    assert not app.exception
    assert app.radio(key="network_architecture").value.startswith("A")
    app.radio(key="network_architecture").set_value("B · local CNN").run()
    assert not app.exception


def test_changing_daily_round_count_clears_exact_schedule():
    app = advanced_app()
    app.number_input(key="recordings_per_day").set_value(8).run()
    assert not app.exception
    assert app.session_state["cfg"]["recordings_per_day"] == 8
    assert app.session_state["cfg"]["recording_hours"] == []
    overview = " ".join(item.value for item in app.markdown) + " ".join(item.value for item in app.caption)
    assert "exact recording timestamps are unknown" in overview


def test_old_session_with_custom_round_count_gets_unknown_timestamps():
    app = advanced_app()
    legacy_cfg = app.session_state["cfg"]
    legacy_cfg["recordings_per_day"] = 8
    legacy_cfg.pop("recording_hours")
    app.run()
    assert not app.exception
    assert app.session_state["cfg"]["recording_hours"] == []


def test_nondefault_schedule_caption_uses_actual_hours():
    app = advanced_app()
    app.number_input(key="recordings_per_day").set_value(4).run()
    app.session_state["cfg"]["recording_hours"] = [0, 6, 12, 18]
    app.run()
    assert not app.exception
    overview = " ".join(item.value for item in app.markdown) + " ".join(item.value for item in app.caption)
    assert "4 recording start hours/day at 00:00, 06:00, 12:00, 18:00" in overview
    assert "nighttime 22:00–06:00 hourly" not in overview


def test_priority_controls_recalculate_their_dependent_results():
    app = advanced_app()
    assert not app.exception

    def metric(label):
        return next(item.value for item in app.metric if item.label == label)

    def change(collection, key, value):
        getattr(app, collection)(key=key).set_value(value).run()
        assert not app.exception

    range_before = metric("Modeled range")
    airtime_before = metric("Airtime · payload A")
    change("selectbox", "sf", 12)
    assert metric("Modeled range") != range_before
    assert metric("Airtime · payload A") != airtime_before

    change("selectbox", "frequency_mhz", 433)
    assert metric("Selected module") == "SX1278"
    change("selectbox", "frequency_mhz", 915)
    assert metric("Selected module") == "RFM95W/SX1276 comparison"
    change("selectbox", "frequency_mhz", 433)
    range_before = metric("Modeled range")
    change("slider", "tx_dbm", 20)
    assert metric("Modeled range") != range_before

    change("slider", "inference_latency_ms", 200)
    assert metric("CNN inference energy") == "46.20 mJ"
    change("slider", "inference_current_ma", 100)
    assert metric("CNN inference energy") == "66.00 mJ"

    trees_before = metric("Total physical trees")
    change("slider", "area_ha", 50)
    assert metric("Total physical trees") != trees_before
    trees_before = metric("Total physical trees")
    change("slider", "trees_per_ha", 200)
    assert metric("Total physical trees") != trees_before
    capacity_before = metric("Gateway capacity")
    change("slider", "gateways", 2)
    assert metric("Gateway capacity") != capacity_before
    successes_before = metric("Expected successful transmissions")
    change("slider", "trigger_probability", 1.0)
    assert metric("Expected successful transmissions") != successes_before


def test_scenario_upload_updates_existing_widget_state():
    app = advanced_app()
    next(widget for widget in app.radio if widget.label == "Mode").set_value("Engineering").run()
    scenario = DEFAULTS | {"frequency_mhz": 433, "sf": 9, "area_ha": 10.0}
    app.file_uploader(key="scenario_upload").set_value(
        ("scenario.json", dumps_scenario(scenario).encode(), "application/json")
    ).run()
    next(widget for widget in app.button if widget.label == "Load uploaded scenario").click().run()
    assert not app.exception
    assert app.selectbox(key="frequency_mhz").value == 433
    assert app.selectbox(key="sf").value == 9
    assert app.slider(key="area_ha").value == 10.0


def test_measured_current_profile_is_cleared_when_manual_override_changes():
    app = advanced_app()
    app.file_uploader(key="measurement_upload").set_value(
        ("profile.csv", b"time_ms,current_ma\n0,20\n100,40\n200,20\n", "text/csv")
    ).run()
    next(widget for widget in app.selectbox if widget.key == "measurement_kind").set_value("Current profile").run()
    next(widget for widget in app.button if widget.label == "Import measurements").click().run()
    assert not app.exception
    assert app.session_state["cfg"]["current_profile"]
    assert app.selectbox(key="performance_source").value == "Measured"

    app.slider(key="inference_latency_ms").set_value(400).run()
    assert not app.exception
    assert app.session_state["cfg"]["current_profile"] == []
    assert app.selectbox(key="performance_source").value == "Exploratory assumption"


def test_workload_preset_and_analytical_coverage_selector():
    app = advanced_app()
    assert not app.exception

    app.selectbox(key="workload_preset").set_value(350_000).run()
    app.button(key="apply_workload_preset").click().run()
    assert not app.exception
    assert app.session_state["cfg"]["parameters"] == 350_000
    assert app.session_state["cfg"]["model_mode"] == "Workload"

    app.radio(key="coverage_method").set_value("Analytical area-density").run()
    assert not app.exception
    assert any(metric.label == "Analytical area-density area" for metric in app.metric)


def test_integer_workload_sweep_and_constant_tornado():
    app = advanced_app()
    app.multiselect[0].set_value(["parameters"]).run()
    app.text_input(key="sweep_parameters").set_value("50000,100000").run()
    next(w for w in app.button if w.label == "Run sweep").click().run()
    assert not app.exception
    data = app.session_state["sweep_result"]
    assert data.error.eq("").all()
    app.text_input(key="sweep_parameters").set_value("50000").run()
    next(w for w in app.button if w.label == "Run sweep").click().run()
    app.selectbox(key="sweep_chart").set_value("Tornado").run()
    assert not app.exception


def test_custom_inference_values_outside_slider_range():
    app = advanced_app()
    app.number_input(key="custom_latency_ms").set_value(12000.0).run()
    app.number_input(key="custom_current_ma").set_value(600.0).run()
    assert not app.exception
    assert app.session_state["cfg"]["inference_latency_ms"] == 12000
    assert app.session_state["cfg"]["inference_current_ma"] == 600
