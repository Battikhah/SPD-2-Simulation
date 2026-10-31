from simulator.config import DEFAULTS
from simulator.explanations import FORMULA_GUIDE, input_provenance


def test_every_scenario_input_has_default_and_evidence_label():
    rows = input_provenance(DEFAULTS)
    assert {row["Input"] for row in rows} == set(DEFAULTS)
    assert all(row["Default evidence / source"] and row["Current status"] for row in rows)
    assert next(row for row in rows if row["Input"] == "recording_current_ma")["Default evidence / source"].startswith("Exploratory")
    assert next(row for row in rows if row["Input"] == "battery_mah")["Default evidence / source"].startswith("User-provided")


def test_formula_guide_covers_major_model_stages_and_explains_assumptions():
    guide = {item["Topic"]: item for item in FORMULA_GUIDE}
    assert {
        "Radio range", "LoRa airtime", "Plantation and coverage", "Reporting capacity",
        "Delivery probability", "CNN workload and memory", "Inference timing and energy",
        "Recording and SNR gate", "Daily energy and battery life", "Deployment totals and optimization",
        "Shadowing and reliability", "Schedules, retries, and monitoring",
    } <= set(guide)
    assert all(item["Formula"] and item["Why it is used"] and item["Constants and source"] for item in guide.values())
    assert "not measured" in guide["Reporting capacity"]["Constants and source"]
    assert "uncharacterized SX1278" in guide["Radio range"]["Constants and source"]
