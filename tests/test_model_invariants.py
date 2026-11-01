"""Directional sanity checks on the simulator; not calibration against hardware."""

import math
import random

import pytest

from simulator.config import OPTIONS
from simulator.engine import simulate


def summary(**overrides):
    return simulate(overrides, detail=False)["summary"]


@pytest.mark.parametrize("propagation", ["Free-space", "Log-distance"])
def test_range_does_not_shrink_with_tx_power(propagation):
    ranges = [summary(propagation=propagation, tx_dbm=t)["range_m"] for t in (2, 8, 14, 20)]
    assert ranges == sorted(ranges)


def test_sensitivity_improves_with_spreading_factor():
    sens = [summary(sf=sf)["sensitivity_dbm"] for sf in (7, 8, 9, 10, 11, 12)]
    assert sens == sorted(sens, reverse=True)


def test_battery_life_falls_with_trigger_rate_and_rises_with_capacity():
    life = [summary(trigger_probability=p)["battery_b_days"] for p in (0, 0.25, 0.5, 1)]
    assert life == sorted(life, reverse=True)
    cap = [summary(battery_mah=m)["battery_a_days"] for m in (1000, 2600, 5000)]
    assert cap == sorted(cap)


def test_more_gateways_never_reduce_reachable_nodes():
    reach = [summary(gateways=g)["reachable_nodes"] for g in (1, 2, 4)]
    assert reach == sorted(reach)


def test_random_valid_configs_are_finite_and_consistent():
    rng = random.Random(1)
    for _ in range(40):
        cfg = {k: rng.choice(list(v)) for k, v in OPTIONS.items()}
        cfg |= {
            "performance_source": "Exploratory assumption",
            "layout": rng.choice(["Grid", "Random"]),
            "implicit_header": True,
            "ldro": "auto",
            "frequency_mhz": 433,
            "bandwidth_khz": 125.0,
            "tx_dbm": rng.choice([2, 10, 17, 20]),
            "trigger_probability": rng.choice([0, 0.5, 1.0]),
            "retries": rng.randint(0, 5),
            "area_ha": rng.choice([1, 10]),
            "gateways": rng.randint(1, 4),
            "recording_hours": [],
        }
        s = summary(**cfg)
        assert not any(isinstance(v, float) and math.isnan(v) for v in s.values())
        assert s["served_a"] <= s["reachable_nodes"] + 1e-6
        assert s["served_b"] <= s["reachable_nodes"] + 1e-6
