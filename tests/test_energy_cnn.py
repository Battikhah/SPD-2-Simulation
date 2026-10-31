from copy import deepcopy

import pytest

from simulator.cnn import analyze_cnn
from simulator.config import DEFAULTS
from simulator.energy import calculate_energy


def cfg(**updates):
    return dict(deepcopy(DEFAULTS), **updates)


@pytest.mark.parametrize("latency,current,expected", [(800, 70, 184.8), (200, 70, 46.2), (800, 100, 264.0)])
def test_inference_energy(latency, current, expected):
    assert analyze_cnn(cfg(inference_latency_ms=latency, inference_current_ma=current))[
        "inference_energy_mj"
    ] == pytest.approx(expected)


def test_memory_is_contiguous_and_not_520kb_pool():
    r = analyze_cnn(cfg(tensor_arena_kb=200.0, contiguous_kb=100.0))
    assert r["status"] == "Infeasible"
    assert any("contiguous" in s for s in r["issues"])
    assert analyze_cnn(cfg())["status"] != "Verified feasible"


def test_layer_designer_counts_params_and_macs_separately():
    c = cfg(
        model_mode="Layer designer",
        input_shape=[15, 13, 1],
        layers=[
            {"type": "Conv2D", "filters": 8, "kernel": 3, "stride": 1, "padding": "same"},
            {"type": "GlobalAveragePool"},
            {"type": "Dense", "units": 2},
        ],
    )
    r = analyze_cnn(c)
    assert r["parameters"] == 98
    assert r["macs"] == 15 * 13 * 8 * 9 + 16
    assert r["peak_activation_kb"] > 0


def test_zero_trigger_skips_mfcc_inference_and_tx():
    c = cfg(trigger_probability=0.0)
    radio = {"airtime_a_ms": 338.176, "airtime_b_ms": 46.336, "tx_current_ma": 90.0}
    network = {"attempts_per_event": 1.0, "served_nodes": 100.0, "deployed_nodes": 100, "successful_transmissions": 0.0}
    e = calculate_energy(c, radio, network, network, analyze_cnn(c))
    assert e["a"]["daily_j"] == pytest.approx(e["b"]["daily_j"])
    assert e["a"]["rejected_mj"] < e["a"]["accepted_mj"]
    assert e["deployment_a"]["requests_day"] == 0
    assert e["a"]["battery_daily_j"] > e["a"]["daily_j"]


def test_current_profile_integrates_total_inference_energy():
    c = cfg(current_profile=[{"time_ms": 0, "current_ma": 100}, {"time_ms": 100, "current_ma": 100}])
    r = analyze_cnn(c)
    assert r["inference_energy_mj"] == pytest.approx(33.0)


def test_full_recording_buffer_is_not_assumed_without_streaming():
    assert analyze_cnn(cfg())["raw_audio_kb"] == pytest.approx(156.25)
