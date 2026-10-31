"""Reproducible discrete design-space search; objectives are modeled, not measured."""

from copy import deepcopy
from itertools import product
from math import prod

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .config import DEFAULTS
from .engine import simulate


def sweep(c, space, method="Full factorial", samples=1000, seed=42):
    if not space or any(k not in DEFAULTS or not isinstance(v, list) or not v for k, v in space.items()):
        raise ValueError("Choose known configuration fields, each with a nonempty list of values.")
    if not 1 <= samples <= 20000:
        raise ValueError("Use 1–20,000 samples per run.")
    keys, choices = list(space), list(space.values())
    size = prod(map(len, choices))
    rng = np.random.default_rng(seed)
    if method == "Full factorial":
        if size > 20000:
            raise ValueError(
                f"Full factorial contains {size:,} designs; limit is 20,000. Select random or Latin hypercube search."
            )
        values = product(*choices)
    elif method in ("Random", "Monte Carlo"):
        values = ([v[int(rng.integers(len(v)))] for v in choices] for _ in range(samples))
    elif method in ("Latin hypercube", "Latin Hypercube"):
        points = qmc.LatinHypercube(len(keys), seed=seed).random(samples)
        values = ([v[min(len(v) - 1, int(x * len(v)))] for x, v in zip(point, choices)] for point in points)
    else:
        raise ValueError("Unknown search method.")
    rows = []
    for index, value in enumerate(values):
        config = deepcopy(c)
        parameters = dict(zip(keys, value))
        config.update(parameters)
        if "recordings_per_day" in parameters and "recording_hours" not in parameters and len(config["recording_hours"]) != config["recordings_per_day"]:
            config["recording_hours"] = []
        if {"inference_latency_ms", "inference_current_ma"} & parameters.keys():
            config.update(
                current_profile=[],
                performance_source="Exploratory assumption",
                measurement_note="Search override of latency/current; not a physical benchmark.",
            )
        if method == "Monte Carlo":
            config.update(seed=seed + index, propagation="Monte Carlo")
        try:
            result = simulate(config, detail=False)
            rows.append(dict(result["summary"], **{"search_index": index, "error": ""}))
            rows[-1].update(parameters)
        except (ValueError, OverflowError) as exc:
            rows.append(
                dict(
                    parameters,
                    search_index=index,
                    error=str(exc),
                    feasible=False,
                    feasible_a=False,
                    feasible_b=False,
                    eligible_served_a=0,
                    eligible_served_b=0,
                )
            )
    return pd.DataFrame(rows)


def pareto_front(df, architecture="b"):
    """Non-dominated designs: more eligible nodes, less total energy and fewer gateways."""
    served = (
        f"eligible_trees_{architecture}"
        if f"eligible_trees_{architecture}" in df
        else f"eligible_served_{architecture}"
    )
    energy = f"system_{architecture}_kwh_day"
    if df.empty or not {served, energy, "gateways"}.issubset(df):
        return df.iloc[:0]
    valid = df.loc[(df.get("error", "") == "") & (df[served] > 0)].dropna(subset=[served, energy, "gateways"])
    # Sort energy ascending; only earlier/equal-energy rows can dominate a candidate.
    valid = valid.sort_values([energy, "gateways", served], ascending=[True, True, False])
    frontier = []
    for index, row in valid.iterrows():
        point = (-row[served], row[energy], row["gateways"])
        if any(
            all(a <= b for a, b in zip(other, point)) and any(a < b for a, b in zip(other, point))
            for _, other in frontier
        ):
            continue
        frontier = [
            (i, p)
            for i, p in frontier
            if not (all(a <= b for a, b in zip(point, p)) and any(a < b for a, b in zip(point, p)))
        ]
        frontier.append((index, point))
    return df.loc[[index for index, _ in frontier]]


def sensitivity(c, field, values):
    return sweep(c, {field: list(values)}, samples=min(20000, len(values)))


def tornado(c, fields, fraction=0.2, metric="system_b_kwh_day"):
    baseline = simulate(c, detail=False)["summary"][metric]
    rows = []
    for key in fields:
        values = [c[key] * (1 - fraction), c[key] * (1 + fraction)]
        data = sweep(c, {key: values}, samples=2)
        for label, (_, row) in zip(("Low", "High"), data.iterrows()):
            rows.append({"parameter": key, "level": label, "change": row.get(metric, np.nan) - baseline})
    return pd.DataFrame(rows)
