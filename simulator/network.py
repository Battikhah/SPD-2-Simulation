"""Explicit orchard links and per-gateway scheduling with seeded event realizations."""

import json
import math
from functools import lru_cache

import numpy as np
import pandas as pd

from .radio import airtime_ms, link_budget_db, link_probability, path_loss_db


def dimensions(c):
    if c.get("dimensions_override", False):
        return c.get("width_m", 1000), c.get("length_m", 1000)
    side = math.sqrt(c.get("area_ha", 100) * 10000)
    return side, side


@lru_cache(maxsize=32)
def _layout(serialized):
    c = json.loads(serialized)
    width, length = dimensions(c)
    count = round(width * length / 10000 * c.get("trees_per_ha", 100))
    rng = np.random.default_rng(c.get("seed", 42))
    mode = str(c.get("layout", "Grid")).lower()
    if mode == "custom":
        trees = np.asarray(c.get("custom_trees", []), dtype=float).reshape(-1, 2)
    elif mode == "random":
        trees = rng.uniform([0, 0], [width, length], (count, 2))
    else:
        # Density fixes physical population; requested spacing selects row count.
        cols = max(1, round(width / c.get("tree_spacing_m", 10)))
        rows = max(1, math.ceil(count / cols))
        index = np.arange(count)
        trees = np.column_stack(((index % cols + 0.5) * width / cols, (index // cols + 0.5) * length / rows))
    gateway_count = c.get("gateways", 1)
    specified = np.asarray(c.get("gateway_positions", []), dtype=float).reshape(-1, 2)
    if len(specified) >= gateway_count:
        gateways = specified[:gateway_count]
    else:
        cols = max(1, math.ceil(math.sqrt(gateway_count * width / length)))
        rows = math.ceil(gateway_count / cols)
        i = np.arange(gateway_count)
        gateways = np.column_stack(((i % cols + 0.5) * width / cols, (i // cols + 0.5) * length / rows))
        gateways[: len(specified)] = specified
    trees.setflags(write=False)
    gateways.setflags(write=False)
    return trees, gateways


def layout(c):
    keys = (
        "area_ha",
        "dimensions_override",
        "width_m",
        "length_m",
        "trees_per_ha",
        "tree_spacing_m",
        "layout",
        "custom_trees",
        "seed",
        "gateways",
        "gateway_positions",
    )
    return _layout(json.dumps({key: c[key] for key in keys if key in c}, sort_keys=True))


def analytical_coverage(c, gateways, resolution=100):
    """Area-density approximation from boundary-clipped union, using midpoint quadrature."""
    width, length = dimensions(c)
    x = (np.arange(resolution) + 0.5) * width / resolution
    y = (np.arange(resolution) + 0.5) * length / resolution
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    d = np.linalg.norm(points[:, None, :] - gateways[None, :, :], axis=2)
    d = np.hypot(d, c.get("rx_height_m", 4) - c.get("tx_height_m", 1.5))
    probability = link_probability(c, link_budget_db(c) - path_loss_db(c, d))
    area = width * length / 10000 * float((probability.max(axis=1) >= c.get("min_reliability", 0.9)).mean())
    return area


def simulate_network(c, radio, detail=True, payload="a"):
    """Schedule actual nodes; served_nodes is expected serviceable monitored population.

    Fixed TDMA reserves every scheduled node even if its SNR event gate is closed.
    Dynamic modes schedule a seeded realized active set; monitored population uses
    expected event load. Retry attempts assume independent channel outcomes.
    """
    trees, gateways = layout(c)
    total = len(trees)
    gateway_count = len(gateways)
    resources = min(c.get("channels", 1), c.get("receivers", 1))
    rng = np.random.default_rng(c.get("seed", 42))
    equipped = round(total * c.get("equipped_fraction", 1))
    selected_trees = np.sort(rng.permutation(total)[:equipped])
    tree_ids = np.repeat(selected_trees, c.get("nodes_per_tree", 1))
    if c.get("deployed_limit", 0) > 0:
        tree_ids = tree_ids[: c["deployed_limit"]]
    deployed = len(tree_ids)
    distance = np.linalg.norm(trees[:, None, :] - gateways[None, :, :], axis=2)
    distance = np.hypot(distance, c.get("rx_height_m", 4) - c.get("tx_height_m", 1.5))
    margins = link_budget_db(c) - path_loss_db(c, distance)
    probabilities = link_probability(c, margins)
    long_run_covered = np.max(probabilities, axis=1) >= c.get("min_reliability", 0.9)
    monte = "monte" in str(c.get("propagation", "")).lower()
    if monte:
        sampled = margins - rng.normal(0, c.get("shadowing_db", 4), margins.shape)
        sampled -= (rng.random(margins.shape) < c.get("obstruction_probability", 0.1)) * c.get("obstruction_loss_db", 5)
        # A static channel realization may fail despite passing long-run reliability.
        probabilities = ((probabilities >= c.get("min_reliability", 0.9)) & (sampled >= 0)).astype(float)
        margins = sampled
    best_gateway = np.argmax(probabilities + np.clip(margins, -1000, 1000) * 1e-12, axis=1)
    best_prob = probabilities[np.arange(total), best_gateway]
    tree_reachable = best_prob >= c.get("min_reliability", 0.9)
    node_prob = best_prob[tree_ids] * c.get("packet_success", 0.98)
    assigned = best_gateway[tree_ids]
    reachable = tree_reachable[tree_ids]
    active = rng.random(deployed) < c.get("trigger_probability", 0.25)
    mode = str(c.get("scheduling", "Fixed TDMA")).lower()
    fixed = "fixed" in mode
    aloha = "aloha" in mode
    retries = c.get("retries", 0)
    attempts = retries + 1
    toa = radio[f"airtime_{payload}_ms"] / 1000
    ack_toa = 0.0
    if c.get("ack", False):
        ack_toa = (
            airtime_ms(
                c.get("ack_bytes", 5),
                sf=c.get("sf", 7),
                bandwidth_khz=c.get("bandwidth_khz", 125),
                coding_rate=c.get("coding_rate", 5),
                preamble_symbols=c.get("preamble_symbols", 8),
                implicit_header=c.get("implicit_header", False),
                crc=c.get("crc", True),
                ldro=c.get("ldro", "auto"),
            )
            / 1000
        )
        node_prob *= best_prob[tree_ids] * c.get("packet_success", 0.98)
    wait = c.get("ack_wait_ms", 50) / 1000 if c.get("ack", False) else 0
    guard = (c.get("guard_ms", 20) + 2 * c.get("clock_error_ms", 5)) / 1000
    minimum = attempts * (toa + ack_toa + wait) + guard
    slot = max(c.get("slot_ms", 900) / 1000, minimum) if fixed else minimum
    window = c.get("window_s", 3600)
    capacity_per_gateway = int(math.floor(window / slot + 1e-10)) * resources
    candidates = reachable if fixed else reachable & active
    scheduled = np.zeros(deployed, dtype=bool)
    schedule_rows = []
    latency = []
    trigger = c.get("trigger_probability", 0.25)
    expected_serviceable = 0.0
    monitored_probability = np.zeros(deployed)
    admitted = 0.0
    required_gateways = 0
    gateway_load = []
    for gw in range(gateway_count):
        eligible = np.flatnonzero(reachable & (assigned == gw))
        ids = np.flatnonzero(candidates & (assigned == gw))
        # Strongest-link association; independent receiver lanes cannot exceed RF channels.
        chosen = (ids if minimum <= window else ids[:0]) if aloha else ids[:capacity_per_gateway]
        scheduled[chosen] = True
        demand = len(eligible) if fixed else len(eligible) * trigger
        service_fraction = (
            float(minimum <= window)
            if aloha
            else min(1.0, capacity_per_gateway / demand) if demand else 1.0
        )
        gateway_load.append(
            {
                "gateway": gw,
                "reachable_nodes": len(eligible),
                "expected_offered_requests": len(eligible) * trigger,
                "sampled_offered_requests": int((reachable & active & (assigned == gw)).sum()),
                "scheduled_requests_sample": len(chosen),
                "admitted_population": len(eligible) * service_fraction,
                "capacity_nodes": capacity_per_gateway,
                "daily_capacity_reports": capacity_per_gateway * c.get("recordings_per_day", 1),
                "capacity_is_hard_limit": not aloha,
                "capacity_rejected": int(ids.size - len(chosen)),
            }
        )
        required_gateways += math.ceil(demand / max(capacity_per_gateway, 1))
        if aloha:
            load = len(eligible) * trigger * attempts * toa / (window * resources)
            node_prob[eligible] *= math.exp(-2 * load)
        delivery = 1 - (1 - node_prob[eligible]) ** attempts
        if fixed:
            expected_serviceable += float((1 - (1 - node_prob[chosen]) ** attempts).sum())
            monitored_probability[chosen] = 1 - (1 - node_prob[chosen]) ** attempts
            admitted += len(chosen)
        else:
            expected_serviceable += float(delivery.sum()) * service_fraction
            monitored_probability[eligible] = delivery * service_fraction
            admitted += len(eligible) * service_fraction
        starts = ((np.arange(len(chosen)) // resources) * slot).astype(float)
        if aloha:
            starts = rng.uniform(0, max(0, window - toa), len(chosen))
        completion = starts + (toa if aloha else slot)
        latency.extend(completion[active[chosen]].tolist())
        if detail:
            for k, node in enumerate(chosen):
                lane = k % resources
                start = float(starts[k])
                end = start + (toa if aloha else slot)
                schedule_rows.append((gw, lane, int(node), start, end))
    delivered = 1 - (1 - node_prob) ** attempts
    expected_attempts = (
        sum((1 - node_prob) ** n for n in range(attempts))
        if c.get("ack", False)
        else np.full(deployed, float(attempts))
    )
    reporting = scheduled & active
    reporting_count = int(reporting.sum())
    actual_attempts = float(expected_attempts[reporting].sum())
    success_reports = float(delivered[reporting].sum())
    hypothetical = reachable & scheduled if fixed else reachable
    fallback_probability = c.get("packet_success", 0.98) ** (2 if c.get("ack", False) else 1)
    hypothetical_attempts = (
        float(expected_attempts[hypothetical].mean())
        if hypothetical.any()
        else (sum((1 - fallback_probability) ** n for n in range(attempts)) if c.get("ack", False) else float(attempts))
    )
    reserved_demand = int(reachable.sum()) * (1 if fixed else trigger) * slot
    width, length = dimensions(c)
    tree_failure = np.ones(total)
    np.multiply.at(tree_failure, tree_ids, 1 - monitored_probability)
    served_trees = float((1 - tree_failure).sum())
    result = dict(
        total_trees=total,
        deployed_nodes=deployed,
        covered_trees=int(tree_reachable.sum()),
        reachable_nodes=int(reachable.sum()),
        long_run_covered_trees=int(long_run_covered.sum()),
        long_run_reachable_nodes=int(long_run_covered[tree_ids].sum()),
        bounded_coverage_ha=width * length / 10000 * float(tree_reachable.mean()) if total else 0.0,
        bounded_coverage_basis="Plantation area × covered tree fraction; uniform-density area proxy, overlapping gateways counted once",
        scheduled_nodes=int(scheduled.sum()),
        served_nodes=float(expected_serviceable),
        served_trees=served_trees,
        served_trees_basis="Expected unique monitored trees; any admitted sensor succeeds, independent sensor delivery assumed",
        admitted_nodes=float(admitted),
        monitored_nodes=float(expected_serviceable),
        monitored_nodes_basis="Expected successful monitored population; admitted_nodes is population before link-delivery loss",
        outside_nodes=int((~reachable).sum()),
        capacity_rejected=int((candidates & ~scheduled).sum()),
        capacity_nodes=capacity_per_gateway * gateway_count,
        required_gateways=max(1, required_gateways),
        reserved_utilization=reserved_demand / (window * resources * gateway_count),
        airtime_utilization=actual_attempts * (toa + ack_toa) / (window * resources * gateway_count),
        latency_avg_s=float(np.mean(latency)) if latency else 0.0,
        latency_worst_s=max(latency, default=0.0),
        delivery_probability=float(expected_serviceable / admitted) if admitted else 0.0,
        attempts_per_event=hypothetical_attempts,
        slot_s=slot,
        round_demand_s=reserved_demand / (resources * gateway_count),
        requests_per_round=reporting_count,
        offered_requests_per_round=int((reachable & active).sum()),
        expected_requests_per_round=float(reachable.sum()) * trigger,
        successful_transmissions=success_reports,
        expected_successful_transmissions=expected_serviceable * trigger,
        scheduled_active_nodes=reporting_count,
        realized_successful_transmissions=int((rng.random(deployed) < delivered)[reporting].sum()),
        schedule_basis="Seeded active requests; strongest-link association; dynamic monitored population uses expected-load capacity approximation",
        reliability_basis=(
            "Static sampled shadowing/obstruction after long-run eligibility; packet retries independent"
            if monte
            else "Lognormal margin probability × exploratory packet success; independent retries"
        ),
        contention_basis="Pure ALOHA exp(-2G); first-attempt timeline only; no capture or inter-SF interference"
        if aloha
        else "Dedicated independent receiver lanes",
        gateway_load=pd.DataFrame(gateway_load),
    )
    if detail:
        analytical_area = analytical_coverage(c, gateways)
        result.update(
            analytical_coverage_ha=analytical_area,
            analytical_covered_trees=min(total, analytical_area * c.get("trees_per_ha", 100)),
            analytical_coverage_basis="100×100 midpoint area quadrature of long-run gateway union, clipped to orchard; uniform density, separate from explicit coordinate scheduling.",
        )
        node_margin = margins[tree_ids, assigned]
        status = np.where(
            ~reachable,
            "Outside coverage",
            np.where(scheduled, "Served", np.where(~active & ~fixed, "Inactive candidate", "Capacity limited")),
        )
        nodes = pd.DataFrame(
            dict(
                x=trees[tree_ids, 0],
                y=trees[tree_ids, 1],
                tree_id=tree_ids,
                gateway=assigned,
                margin_db=node_margin,
                probability=node_prob,
                status=status,
                active=active,
                scheduled=scheduled,
            )
        )
        unequipped = np.flatnonzero(~np.isin(np.arange(total), np.unique(tree_ids)))
        if len(unequipped):
            nodes = pd.concat(
                [
                    nodes,
                    pd.DataFrame(
                        dict(
                            x=trees[unequipped, 0],
                            y=trees[unequipped, 1],
                            tree_id=unequipped,
                            gateway=-1,
                            margin_db=np.nan,
                            probability=0.0,
                            status="Unequipped",
                            active=False,
                            scheduled=False,
                        )
                    ),
                ],
                ignore_index=True,
            )
        if monte:
            result["sampled_link_margins_db"] = sampled
        result.update(
            nodes=nodes,
            schedule=pd.DataFrame(schedule_rows, columns=["gateway", "channel", "node", "start_s", "end_s"]),
            gateways=gateways,
        )
    return result
