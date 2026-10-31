# Palm sensing simulation research report

Generated UTC: 2026-10-28T08:39:57.349067+00:00

**All results are calculated estimates; no hardware verification is implied.**

Configuration: 433 MHz, SF7, 125.0 kHz, 17.0 dBm.

## User-reported hardware baseline

```json
{
  "reported_on": "2026-10-28",
  "battery": {
    "model": "505573",
    "voltage_v": 3.7,
    "capacity_mah": 2600,
    "quantity": 3
  },
  "charger": {
    "model": "HiLetgo TP4056 USB-C",
    "quantity": null,
    "protection": "Advertised; unverified"
  },
  "converter": {
    "model": "XL6009E1 step-up",
    "quantity": null
  },
  "sensor": {
    "model": "WANGCL analog ceramic piezo module",
    "voltage": "3.3 V/5 V",
    "kit_quantity": 5
  },
  "amplifier": {
    "model": "MCP602E dual CMOS op amp, SOP-8",
    "quantity": 4,
    "report_model": "MCP6002"
  },
  "mcu": {
    "model": "ESP-WROOM-32 / ESP-32S",
    "quantity": 3,
    "cpu_mhz_max": 240,
    "pcb_revision": null
  },
  "radio": {
    "model": "SX1278",
    "frequency_mhz": 433,
    "quantity": 4
  },
  "gateway": {
    "model": "Raspberry Pi 4 Model B, 4 GB, SPI-connected LoRa receiver",
    "purchase_confirmed": false
  },
  "unknowns": [
    "Final power wiring, regulator efficiency, and actual sleep current",
    "Exact PCB revision and whether ESP32 boards include development-board circuitry",
    "Piezo wiring, amplifier gain, resistor values, and final schematic",
    "Transmit power, antenna model/gain, and measured RF performance",
    "Whether the planned Raspberry Pi gateway was purchased"
  ]
}
```

Hardware quantities describe reported purchases; they do not scale per-node battery capacity or limit modeled deployment counts.
Channel model: Log-distance. Seed: 42.
Inference profile: Exploratory assumption. 

| Metric | Architecture A | Architecture B |
|---|---:|---:|
| Physical trees | 10000 | 10000 |
| Reachable nodes | 5452 | 5452 |
| Expected monitored nodes | 3851.20 | 3851.20 |
| Expected unique monitored trees | 3851.20 | 3851.20 |
| Airtime (ms) | 338.176 | 46.336 |
| Node battery energy (J/day) | 28.349 | 28.547 |
| Battery life (days) | 690.8 | 687.0 |
| Entire deployment (kWh/day) | 0.392735 | 0.392404 |

## Cluster-head reporting capacity

Each modeled gateway site has a per-round scheduled capacity of 4000 nodes for A and 4000 for B, across 1 RF channel(s) and 1 receiver(s). TDMA capacity applies at each assigned head; ALOHA collisions are modeled probabilistically. Unused capacity at another head cannot serve overloaded assignments.
At a 100% trigger rate, A admits 4000.0 of 5452 reachable nodes per recording opportunity and rejects 1,452; B admits 4000.0 and rejects 1,452.
This worst case means all reachable nodes request a report on every configured recording opportunity.
The modeled head is a gateway receiving site; no additional relay hop or buffer is assumed.

CNN inference energy: 184.800 mJ. Single-attempt transmission saving: 125.199 mJ.
Core break-even inference time: 541.989 ms.

## Feasibility

Infeasible for full-deployment targets

- Architecture A: 4,548 deployed nodes fail the modeled link-reliability requirement.
- Architecture A: 1,452 reachable nodes exceed reporting-window capacity.
- Architecture B: 4,548 deployed nodes fail the modeled link-reliability requirement.
- Architecture B: 1,452 reachable nodes exceed reporting-window capacity.

## Limitations

- No hardware, RF field, classifier accuracy or battery measurement is bundled. No design is marked verified feasible.
- Log-distance exponent, shadowing, foliage and independent retry outcomes are exploratory; no plantation-specific propagation calibration is bundled.
- Coverage circle is theoretical and unbounded; explicit tree counts respect orchard boundaries and gateway overlap.
- Strongest-link gateway association is heuristic. Dynamic monitored capacity uses expected event load; seeded actual events and schedules are shown separately.
- ALOHA uses exp(-2G), with no capture, adjacent-channel rejection or inter-SF interference simulation.
- Sequential CNN activation and serialized-size estimates do not replace TFLM arena instrumentation, an operator audit or a compiled ESP32 test.
- MAC-based latency is an independent exploratory estimate. Manual latency/current controls are not predictions from model parameter count.
- Battery model assumes constant usable capacity factors and load, constant solar yield and exponential self discharge; voltage sag and discharge curves are unmodeled.
- All deployed nodes wake and process. Scheduled nodes add communication energy; no battery savings are claimed for silently dropping unserved nodes.
- This models acoustic gating and workload, not pest classification accuracy. Trigger probability must be calibrated against actual acoustic windows.

## Sources

- [Semtech SX1278 product documentation](https://www.semtech.com/products/wireless-rf/lora-connect/sx1278) — Manufacturer product reference: Purchased radio family and low-frequency band reference; module-board configuration and RF performance remain unverified.
- [Microchip MCP602 product documentation](https://www.microchip.com/en-us/product/mcp602) — Manufacturer product reference: Purchased amplifier family; not a substitute for the report's MCP6002 circuit or an amplifier-current measurement.
- [Semtech SX1276 datasheet and product documents](https://www.semtech.com/products/wireless-rf/lora-connect/sx1276) — Manufacturer specified: Airtime equation, SF6 implicit header, low-data-rate optimization and raw-LoRa payload limit.
- [HopeRF RFM95W/96W/98W v2.0](https://www.hoperf.com/uploads/RFM95W-V2.0_1695351437.pdf) — Manufacturer specified: Legacy low-band sensitivity table and TX current anchors used as family-reference proxies for the uncharacterized purchased SX1278 board; current interpolation is estimated, not board measured.
- [Espressif ESP32 memory types](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-guides/memory-types.html) — Manufacturer specified: 320 KiB DRAM / 200 KiB IRAM distinction and static allocation limits.
- [ESP-NN implementation and benchmarks](https://github.com/espressif/esp-nn) — Published benchmark reference: Backend reference only; no benchmark transferred to this CNN or original ESP32.
- [TFLite Micro memory management](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/memory_management.md) — Implementation documentation: Arena head/tail, planning and operator scratch require actual allocation measurements.
- Senior Design 1 Report.pdf — Not attached. 15×13 input, 10 s / 8 kHz, 12 rounds/day and 900 ms slot are supplied-prompt assumptions, not verified report facts.

## Reproducible configuration

```json
{
  "schema_version": 1,
  "config": {
    "frequency_mhz": 433,
    "sf": 7,
    "bandwidth_khz": 125.0,
    "coding_rate": 5,
    "tx_dbm": 17.0,
    "tx_gain_dbi": 2.0,
    "rx_gain_dbi": 2.0,
    "tx_height_m": 1.5,
    "rx_height_m": 4.0,
    "cable_loss_db": 1.0,
    "misc_loss_db": 3.0,
    "required_margin_db": 10.0,
    "propagation": "Log-distance",
    "path_loss_exponent": 3.5,
    "reference_distance_m": 1.0,
    "vegetation_loss_db": 10.0,
    "shadowing_db": 4.0,
    "obstruction_loss_db": 5.0,
    "obstruction_probability": 0.1,
    "min_reliability": 0.9,
    "interference_loss_db": 0.0,
    "seed": 42,
    "payload_a_bytes": 214,
    "payload_b_bytes": 13,
    "metadata_bytes": 0,
    "protocol_bytes": 0,
    "preamble_symbols": 8,
    "implicit_header": false,
    "crc": true,
    "ldro": "auto",
    "tx_current_override_ma": 0.0,
    "radio_standby_ma": 1.6,
    "radio_sleep_ma": 0.0002,
    "radio_rx_ma": 11.5,
    "mcu_tx_ma": 40.0,
    "area_ha": 100.0,
    "width_m": 1000.0,
    "length_m": 1000.0,
    "dimensions_override": false,
    "trees_per_ha": 100.0,
    "tree_spacing_m": 10.0,
    "layout": "Grid",
    "custom_trees": [],
    "nodes_per_tree": 1,
    "equipped_fraction": 1.0,
    "deployed_limit": 0,
    "gateways": 1,
    "gateway_positions": [],
    "channels": 1,
    "receivers": 1,
    "scheduling": "Fixed TDMA",
    "slot_ms": 900.0,
    "guard_ms": 20.0,
    "clock_error_ms": 5.0,
    "window_s": 3600.0,
    "retries": 0,
    "ack": false,
    "ack_bytes": 5,
    "ack_wait_ms": 50.0,
    "packet_success": 0.98,
    "delivery_target": 0.9,
    "trigger_probability": 0.25,
    "parameters": 260833,
    "macs": 11283328,
    "precision": "INT8",
    "input_shape": [
      15,
      13,
      1
    ],
    "layers": [],
    "model_mode": "Workload",
    "backend": "TFLite Micro + ESP-NN",
    "cpu_mhz": 240,
    "cpu_utilization": 1.0,
    "cycles_per_mac": 8.0,
    "inference_latency_ms": 800.0,
    "inference_current_ma": 70.0,
    "performance_source": "Exploratory assumption",
    "measurement_note": "",
    "current_profile": [],
    "voltage_v": 3.3,
    "active_current_ma": 70.0,
    "flash_kb": 4096.0,
    "firmware_flash_kb": 1024.0,
    "dram_kb": 320.0,
    "firmware_ram_kb": 80.0,
    "stack_kb": 16.0,
    "audio_buffer_kb": 16.0,
    "dma_buffer_kb": 8.0,
    "tensor_arena_kb": 120.0,
    "scratch_kb": 16.0,
    "contiguous_kb": 160.0,
    "activation_kb": 48.0,
    "operators_supported": true,
    "deadline_ms": 10000.0,
    "recording_s": 10.0,
    "sample_rate_hz": 8000,
    "filter_low_hz": 100.0,
    "filter_high_hz": 2500.0,
    "recording_current_ma": 40.0,
    "wake_ms": 50.0,
    "wake_current_ma": 80.0,
    "afe_ms": 10.0,
    "afe_current_ma": 10.0,
    "preprocessing_ms": 100.0,
    "preprocessing_current_ma": 70.0,
    "mfcc_ms": 200.0,
    "mfcc_current_ma": 70.0,
    "radio_startup_ms": 5.0,
    "sleep_current_ma": 0.01,
    "snr_gate_db": 4.0,
    "recordings_per_day": 12,
    "recording_hours": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      10,
      14,
      18,
      22,
      23
    ],
    "battery_v": 3.7,
    "battery_mah": 2600.0,
    "regulator_efficiency": 0.9,
    "quiescent_ma": 0.02,
    "usable_fraction": 0.8,
    "temperature_factor": 1.0,
    "aging_factor": 0.9,
    "self_discharge_monthly": 0.02,
    "solar_wh_day": 0.0,
    "battery_target_days": 365.0,
    "gateway_idle_w": 3.0,
    "gateway_process_w": 1.0,
    "backhaul_j_packet": 0.05,
    "server_idle_w": 10.0,
    "server_inference_w": 20.0,
    "server_inference_ms": 10.0,
    "server_model_capacity_mb": 512.0,
    "node_cost": 25.0,
    "gateway_cost": 100.0,
    "electricity_cost_kwh": 0.12
  }
}
```