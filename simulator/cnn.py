"""CNN bookkeeping, not an ESP32 execution benchmark or TFLM allocator."""

from math import ceil, prod

import numpy as np
import pandas as pd


def layer_workload(input_shape, layers, precision):
    """Count sequential Conv2D/pooling/dense operations, keeping live input/output."""
    shape = list(input_shape)
    total_params = total_macs = peak = 0
    rows = []
    for layer in layers:
        old = prod(shape)
        kind = layer["type"]
        params = macs = 0
        if kind in ("Conv2D", "MaxPool", "AveragePool"):
            if len(shape) != 3:
                raise ValueError(f"{kind} requires an H×W×C input.")
            h, w, channels = shape
            kernel = int(layer.get("kernel", 3 if kind == "Conv2D" else 2))
            stride = int(layer.get("stride", 1 if kind == "Conv2D" else 2))
            if kernel < 1 or stride < 1:
                raise ValueError("Kernel and stride must be positive integers.")
            if layer.get("padding", "valid") == "same":
                out_h, out_w = ceil(h / stride), ceil(w / stride)
            else:
                out_h, out_w = (h - kernel) // stride + 1, (w - kernel) // stride + 1
            filters = int(layer.get("filters", 8)) if kind == "Conv2D" else channels
            if min(out_h, out_w, filters) < 1:
                raise ValueError("A layer produces an empty output.")
            shape = [out_h, out_w, filters]
            if kind == "Conv2D":
                params = (kernel * kernel * channels + 1) * filters
                macs = prod(shape) * kernel * kernel * channels
        elif kind == "GlobalAveragePool":
            shape = [shape[-1]]
        elif kind == "Flatten":
            shape = [old]
        elif kind == "Dense":
            units = int(layer.get("units", 2))
            if units < 1:
                raise ValueError("Dense units must be positive.")
            params, macs = (old + 1) * units, old * units
            shape = [units]
        elif kind != "Activation":
            raise ValueError(f"Unsupported layer: {kind}")
        peak = max(peak, old + prod(shape))
        total_params += params
        total_macs += macs
        rows.append({"layer": kind, "output": str(shape), "parameters": params, "macs": macs})
    return total_params, total_macs, peak * (1 if precision == "INT8" else 4) / 1024, pd.DataFrame(rows)


def analyze_cnn(c):
    parameters, macs, activation = c["parameters"], c["macs"], c["activation_kb"]
    layers = pd.DataFrame(columns=["layer", "output", "parameters", "macs"])
    if c["model_mode"] == "Layer designer":
        parameters, macs, activation, layers = layer_workload(c["input_shape"], c["layers"], c["precision"])
    weight_kb = parameters * (1 if c["precision"] == "INT8" else 4) / 1024
    # shortcut: FlatBuffer metadata is estimated at 5% plus 4 KiB; import an actual model for exact storage.
    serialized = weight_kb * 1.05 + 4
    heap = c["dram_kb"] - c["firmware_ram_kb"] - c["stack_kb"] - c["audio_buffer_kb"] - c["dma_buffer_kb"]
    arena_needed = activation + c["scratch_kb"] + 8
    ram_used = c["dram_kb"] - heap + c["tensor_arena_kb"]
    latency, current = c["inference_latency_ms"], c["inference_current_ma"]
    inference_energy = c["voltage_v"] * current * latency / 1000
    if c["current_profile"]:
        profile = c["current_profile"]
        t = np.array([p["time_ms"] for p in profile])
        i = np.array([p["current_ma"] for p in profile])
        latency = float(t[-1] - t[0])
        inference_energy = float(c["voltage_v"] * np.trapezoid(i, t) / 1000)
        current = inference_energy * 1000 / c["voltage_v"] / latency
    issues = []
    if c["tensor_arena_kb"] > min(heap, c["contiguous_kb"]):
        issues.append("Tensor arena exceeds available heap or contiguous internal RAM.")
    if c["tensor_arena_kb"] < arena_needed:
        issues.append("Tensor arena is below the estimated live-activation, scratch and metadata requirement.")
    if serialized + c["firmware_flash_kb"] > c["flash_kb"]:
        issues.append("Serialized model and firmware exceed configured flash capacity.")
    if not c["operators_supported"]:
        issues.append("One or more model operators are unsupported by the selected backend.")
    if latency + c["preprocessing_ms"] + c["mfcc_ms"] > c["deadline_ms"]:
        issues.append("Inference plus preprocessing exceeds the configured processing deadline.")
    estimated_latency = macs * c["cycles_per_mac"] / (c["cpu_mhz"] * 1000 * c["cpu_utilization"])
    return {
        "parameters": parameters,
        "macs": macs,
        "weight_kb": weight_kb,
        "serialized_kb": serialized,
        "peak_activation_kb": activation,
        "arena_required_kb": arena_needed,
        "heap_available_kb": heap,
        "contiguous_kb": min(c["contiguous_kb"], max(heap, 0)),
        "ram_used_kb": ram_used,
        "flash_used_kb": serialized + c["firmware_flash_kb"],
        "estimated_latency_ms": estimated_latency,
        "effective_latency_ms": latency,
        "effective_current_ma": current,
        "inference_energy_mj": inference_energy,
        "status": "Infeasible" if issues else "Plausibly feasible but unverified",
        "issues": issues,
        "layers": layers,
        "raw_audio_kb": c["recording_s"] * c["sample_rate_hz"] * 2 / 1024,
        "memory_note": "Sequential activation estimate; TFLM arena, fragmentation and operator scratch require on-device measurement. Audio assumes streaming buffers, not a full recording in RAM.",
    }
