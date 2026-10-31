import numpy as np
import pytest

from simulator.network import layout, simulate_network
from simulator.radio import airtime_ms, fspl_db, radio_summary


def scenario(**updates):
    c = dict(
        frequency_mhz=915,
        sf=7,
        bandwidth_khz=125,
        coding_rate=5,
        tx_dbm=17,
        area_ha=100,
        trees_per_ha=100,
        trigger_probability=1,
        shadowing_db=4,
        path_loss_exponent=3.5,
        vegetation_loss_db=10,
    )
    return c | updates


def test_airtime_regressions_and_invalid_configurations():
    assert airtime_ms(214) == pytest.approx(338.176)
    assert airtime_ms(13) == pytest.approx(46.336)
    with pytest.raises(ValueError):
        airtime_ms(256)
    with pytest.raises(ValueError):
        airtime_ms(13, sf=6)
    assert airtime_ms(13, sf=6, implicit_header=True) > 0
    with pytest.raises(ValueError):
        radio_summary(scenario(bandwidth_khz=62.5))


def test_frequency_path_loss_and_manufacturer_profiles():
    a = radio_summary(scenario())
    b = radio_summary(scenario(frequency_mhz=433))
    assert a["module"] == "RFM95W/SX1276 comparison"
    assert b["module"] == "SX1278"
    assert "SX1278" not in a["sensitivity_basis"]
    assert "low-band" not in a["sensitivity_basis"]
    assert "proxy" in b["sensitivity_basis"]
    assert fspl_db(1000, 915) - fspl_db(1000, 433) == pytest.approx(6.499, abs=0.01)
    assert a["sensitivity_dbm"] == -123
    assert b["sensitivity_dbm"] == -125
    assert b["range_m"] > a["range_m"]


def test_geography_and_sf_tradeoff():
    a = scenario()
    b = a | {"sf": 12}
    na = simulate_network(a, radio_summary(a))
    nb = simulate_network(b, radio_summary(b))
    assert na["total_trees"] == nb["total_trees"] == 10000
    assert nb["reachable_nodes"] > na["reachable_nodes"]
    assert nb["capacity_nodes"] < na["capacity_nodes"]
    assert nb["scheduled_nodes"] < nb["reachable_nodes"]
    assert nb["served_nodes"] <= nb["scheduled_nodes"]
    assert np.array_equal(layout(a)[0], layout(b)[0])


def test_fixed_tdma_8000_round_and_receiver_resources():
    c = scenario(
        area_ha=80, propagation="Free-space", slot_ms=900, guard_ms=0, clock_error_ms=0, channels=8, receivers=1
    )
    n = simulate_network(c, radio_summary(c))
    assert n["round_demand_s"] == pytest.approx(7200)
    assert n["reserved_utilization"] == pytest.approx(2)
    assert n["scheduled_nodes"] == 4000
    assert n["capacity_rejected"] == 4000
    assert n["required_gateways"] == 2
    schedule = n["schedule"]
    assert schedule.end_s.max() <= 3600
    assert (schedule.start_s.to_numpy()[1:] >= schedule.end_s.to_numpy()[:-1] - 1e-9).all()


def test_event_gate_fixed_reservation_and_seed_reproducibility():
    c = scenario(area_ha=1, trigger_probability=0)
    n = simulate_network(c, radio_summary(c))
    assert n["scheduled_nodes"] == 100
    assert n["requests_per_round"] == 0
    assert n["served_nodes"] > 0  # An inactive sensing node still belongs to the serviceable deployment.
    assert n["successful_transmissions"] == 0
    assert n["airtime_utilization"] == 0
    assert n["reserved_utilization"] > 0
    c["scheduling"] = "Dynamic TDMA"
    assert simulate_network(c, radio_summary(c))["scheduled_nodes"] == 0
    c["trigger_probability"] = 0.5
    a = simulate_network(c, radio_summary(c))
    b = simulate_network(c, radio_summary(c), detail=False)
    for key in ("reachable_nodes", "scheduled_nodes", "served_nodes", "requests_per_round"):
        assert a[key] == b[key]


def test_multiple_gateways_do_not_double_count_trees_or_overbook():
    c = scenario(
        area_ha=5,
        gateways=2,
        gateway_positions=[[20, 20], [200, 200]],
        channels=4,
        receivers=2,
        retries=2,
        ack=True,
        window_s=10,
    )
    n = simulate_network(c, radio_summary(c))
    assert n["covered_trees"] <= n["total_trees"]
    assert n["scheduled_nodes"] <= n["reachable_nodes"]
    for _, group in n["schedule"].groupby(["gateway", "channel"]):
        assert (group.start_s.to_numpy()[1:] >= group.end_s.to_numpy()[:-1] - 1e-9).all()


def test_custom_layout_and_equipped_fraction():
    c = scenario(layout="Custom", custom_trees=[[0, 0], [10, 10], [20, 20]], equipped_fraction=1 / 3, nodes_per_tree=2)
    n = simulate_network(c, radio_summary(c))
    assert n["total_trees"] == 3
    assert n["deployed_nodes"] == 2
    assert (n["nodes"].status == "Unequipped").sum() == 2


def test_monte_carlo_sample_is_seeded_and_aloha_latency_within_window():
    c = scenario(propagation="Monte Carlo", scheduling="ALOHA", window_s=10)
    a = simulate_network(c, radio_summary(c))
    b = simulate_network(c, radio_summary(c), detail=False)
    assert a["latency_worst_s"] <= 10
    assert a["latency_worst_s"] == b["latency_worst_s"]
    assert a["served_nodes"] == b["served_nodes"]
    sample = simulate_network(c, radio_summary(c))["sampled_link_margins_db"]
    assert np.array_equal(a["sampled_link_margins_db"], sample)


def test_hypothetical_event_attempts_and_admitted_population_at_zero_trigger():
    c = scenario(area_ha=1, trigger_probability=0, retries=2)
    n = simulate_network(c, radio_summary(c))
    assert n["attempts_per_event"] > 1
    assert n["admitted_nodes"] == n["scheduled_nodes"] == 100
    assert 0 < n["served_nodes"] < n["admitted_nodes"]


def test_monte_carlo_realization_affects_primary_results_and_preserves_long_run_counts():
    c = scenario(propagation="Monte Carlo", shadowing_db=12)
    a = simulate_network(c, radio_summary(c))
    b = simulate_network(c | {"seed": 43}, radio_summary(c | {"seed": 43}))
    assert a["long_run_reachable_nodes"] >= a["reachable_nodes"]
    assert a["reachable_nodes"] != b["reachable_nodes"]


def test_height_affects_slant_distance_and_bounded_coverage():
    c = scenario(propagation="Free-space", area_ha=1)
    a = simulate_network(c, radio_summary(c))
    high = c | {"rx_height_m": 1000000}
    b = simulate_network(high, radio_summary(high))
    assert b["covered_trees"] < a["covered_trees"]
    assert a["bounded_coverage_ha"] <= 1
    assert radio_summary(high)["range_m"] < radio_summary(c)["range_m"]


def test_aloha_cannot_finish_a_packet_longer_than_window():
    from simulator.config import DEFAULTS

    c = DEFAULTS | {
        "area_ha": 1.0,
        "trees_per_ha": 1.0,
        "sf": 12,
        "scheduling": "ALOHA",
        "window_s": 0.01,
        "trigger_probability": 1.0,
        "propagation": "Free-space",
    }
    n = simulate_network(c, radio_summary(c))
    assert n["scheduled_nodes"] == 0
    assert n["served_nodes"] == 0
    assert n["capacity_rejected"] == 1
    assert n["gateway_load"].capacity_is_hard_limit.eq(False).all()
    assert n["gateway_load"].capacity_rejected.sum() == 1


def test_daily_expectation_is_not_one_seeded_event_round():
    from simulator.config import DEFAULTS

    c = DEFAULTS | {
        "area_ha": 1.0,
        "trees_per_ha": 1.0,
        "trigger_probability": 0.25,
        "propagation": "Free-space",
        "retries": 2,
        "ack": False,
    }
    n = simulate_network(c, radio_summary(c))
    assert n["expected_successful_transmissions"] > 0
    assert n["attempts_per_event"] == 3


def test_analytical_area_density_union_respects_boundaries():
    from simulator.config import DEFAULTS

    c = DEFAULTS | {
        "area_ha": 1.0,
        "trees_per_ha": 100.0,
        "propagation": "Free-space",
        "gateways": 2,
        "gateway_positions": [[50.0, 50.0], [50.0, 50.0]],
    }
    n = simulate_network(c, radio_summary(c))
    assert n["analytical_coverage_ha"] == pytest.approx(1.0)
    assert n["analytical_covered_trees"] == pytest.approx(100.0)


def test_multiple_sensors_do_not_count_as_multiple_monitored_trees():
    from simulator.config import DEFAULTS

    c = DEFAULTS | {
        "area_ha": 1.0,
        "trees_per_ha": 1.0,
        "nodes_per_tree": 2,
        "packet_success": 0.5,
        "propagation": "Free-space",
    }
    n = simulate_network(c, radio_summary(c), detail=False)
    assert n["served_nodes"] == pytest.approx(1.0)
    assert n["served_trees"] == pytest.approx(0.75)


def test_gateway_load_reports_individual_capacity_and_offered_load():
    c = scenario(
        area_ha=1,
        gateways=2,
        gateway_positions=[[50, 50], [50, 50]],
        trigger_probability=0.25,
        deployed_limit=500,
        propagation="Free-space",
    )
    n = simulate_network(c, radio_summary(c), detail=False)
    heads = n["gateway_load"]
    assert len(heads) == 2
    assert heads.loc[0, "reachable_nodes"] == 100
    assert heads.loc[0, "capacity_nodes"] == 4000
    assert heads.loc[0, "expected_offered_requests"] == pytest.approx(25)
    assert heads.loc[0, "scheduled_requests_sample"] == 100
    assert heads.loc[1, "reachable_nodes"] == 0


def test_per_gateway_load_catches_one_overloaded_head_when_aggregate_fits():
    c = scenario(
        area_ha=80,
        gateways=2,
        gateway_positions=[[200, 200], [200, 200]],
        propagation="Free-space",
        deployed_limit=5000,
        slot_ms=900,
        guard_ms=0,
        clock_error_ms=0,
    )
    n = simulate_network(c, radio_summary(c), detail=False)
    heads = n["gateway_load"]
    assert n["capacity_nodes"] == 8000
    assert n["reachable_nodes"] == 5000
    assert heads.loc[0, "reachable_nodes"] == 5000
    assert heads.loc[0, "capacity_rejected"] == 1000
    assert heads.loc[1, "capacity_rejected"] == 0
