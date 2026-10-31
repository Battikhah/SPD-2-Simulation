# Purchased hardware baseline — user clarification, 28 October 2026

This clarification supersedes hardware defaults in the original pasted request.
The inventory below was supplied by the user; receipts, board markings, wiring
and physical performance were not independently inspected in this update.

## Planned 3.3 V regulator for the final circuit

The user selected the Pololu S7V8F3 for the modeled final-node power path:
3.7 V LiPo → fixed 3.3 V regulator → ESP32 and SX1278. It accepts 2.7–11.8 V.
Pololu rates its module for about 500 mA–1 A continuous output across much of the
input/output range; available current depends on voltage and thermal conditions.
The specifications list up to 100 µA quiescent current when enabled under the
specified no-load test; actual current depends on input voltage and load.
The simple simulator estimates sleep from components. It adds the 100 µA enabled/no-load
regulator allowance to (not inside) the converted 3.3 V loads. Defaults use 10 µA
for the ESP32 chip deep-sleep RTC-timer reference, 0.2 µA for the SX1278 typical
sleep reference, a 1 µA exploratory leakage allowance for the power-gated
sensor/amplifier branch, zero other 3.3 V loads for a custom circuit, and a 10 µA
unsourced battery-side allowance for protection/charger/divider leakage. Rail-load
conversion assumes 80% efficiency at sleep load, excluding the separately budgeted
regulator quiescent current. The result is about 122.5 µA battery-side, not a
measurement or verified board bound. The simple simulator uses this component
estimate directly and no longer includes a manual sleep-current selector.

The ESP32 number is chip-level, not a guarantee for the ESP-WROOM-32 module or
purchased development boards, which may include flash, regulators, USB interfaces,
LEDs and other loads. Exact sleep current needs measurement on the actual assembly.
Power gating must avoid back-power through signal pins.

Sources: [Espressif ESP32 datasheet, table 4-2](https://documentation.espressif.com/esp32_datasheet_en.html);
[SX1278 datasheet, table 2.5.1](https://uelectronics.com/wp-content/uploads/2026/01/SX1278-Datasheet.pdf);
[Pololu S7V8F3 specifications](https://www.pololu.com/product/2122/specs).
The 80% efficiency, 1 µA switched leakage, zero other rail current, and 10 µA
battery-side leakage are modeling assumptions, not sourced guarantees.

The S7V8F3 is planned; it is not listed among the purchased components below. The
purchased XL6009E1 remains inventory, but is not selected for the simple
final-circuit estimate. Sensor/amplifier power switching during sleep is a planned
design assumption, not verified wiring. Active-state currents remain the supplied
battery-side estimates; the sleep-load efficiency is not applied to them.

Source: [Pololu S7V8F3 specifications](https://www.pololu.com/product/2122/specs)
and [product information](https://www.pololu.com/product/2122/).

The active-state
battery-side estimates are 180 mA for 2 s wake, 160 mA for 10 s recording, 180 mA
for 5 s processing, and 300 mA for at least 0.5 s transmission. With the 122.5 µA
component sleep estimate, twelve events/day, 2,600 mAh, and 80% usable capacity
give about 160.42 days. Conversion losses are
already reflected in these estimated currents; the simple view does not apply
separate regulator, aging, or self-discharge deductions from the advanced
comparison. Classification runs in the cloud. All currents remain unmeasured
complete-node estimates.

Recording-duration scenarios (10, 30, 60 and 120 seconds) hold battery-side
recording current and packet size fixed. Processing scales linearly from 5 seconds
at a 10-second recording, a workload assumption that must be benchmarked on the
firmware. Trigger probability is held fixed and is not a detection model. The
cluster report window also leaves time for wake, recording and processing before
the hourly TDMA deadline. Detection effectiveness is not modeled; see
[`ACOUSTIC_TEST_PROTOCOL.md`](ACOUSTIC_TEST_PROTOCOL.md) for the TreeVibes
exciter-to-piezo bench-replay procedure.

| Component | User-supplied starting point | Remaining uncertainty |
|---|---|---|
| Battery | 3.7 V, 2,600 mAh rechargeable LiPo, model 505573; three purchased | Actual capacity, discharge curve, cutoff and cell condition |
| Charger | HiLetgo TP4056 USB-C module, advertised battery protection | Exact board circuit, protection implementation, charge current and load sharing |
| Purchased converter | XL6009E1 step-up module | Final wiring, input/output conditions, efficiency and idle current; not selected in the simple model |
| Sensor | WANGCL analog ceramic piezoelectric vibration module, 3.3/5 V; five-piece kit | Exact module circuitry, bias, current and sensor connections |
| Amplifier | MCP602E dual CMOS op amp, SOP-8; four purchased | Gain, resistor values, channels used, loading and power gating |
| MCU board | ESP-WROOM-32 / ESP-32S dual-core board; three-pack; up to 240 MHz | PCB revision, onboard regulator, LEDs, USB circuitry and board sleep current |
| Radio | Four SX1278 modules for 433 MHz | Module RF layout, TX setting/current, antenna type/gain and field link performance |
| Gateway | Planned Raspberry Pi 4 Model B, 4 GB, with SPI LoRa receiver | Purchase and final receiver implementation unconfirmed |

Inventory quantities do not imply three parallel batteries, 7,800 mAh per node,
four radios per node, or a four-node plantation. A simulated sensing node uses one
2,600 mAh battery unless its capacity is explicitly edited. Nominal energy is
3.7 × 2.6 = **9.62 Wh per battery**, before usable-capacity factors and losses.
Plantation-scale scenarios describe hypothetical deployment beyond purchased stock.

## Recording and processing schedule

Times are local to Asia/Dubai. Intervals are half-open: 22:00–06:00 is nighttime;
06:00 belongs to the daytime schedule.

- Night: 22:00, 23:00, 00:00, 01:00, 02:00, 03:00, 04:00, 05:00 — eight recordings.
- Day: 06:00, 10:00, 14:00, 18:00 — four recordings.
- Each recording: 10 seconds at 8 kHz.
- Processing specification: 100–2,500 Hz bandpass, 4 dB SNR gate, then MFCC only
  for accepted windows; Architecture B additionally runs the local CNN.
- Deep sleep between recording cycles is the intended operating policy.

The model evaluates twelve per-day measurement/reporting rounds, with the one-hour
reporting-window constraint relevant to hourly nighttime rounds. It does not claim
the daytime gaps make an overloaded hourly nighttime schedule feasible. Recording
hours describe sensing triggers, not exact RF packet transmission timestamps;
TDMA scheduling and propagation still determine admission/delivery.

Changing the daily count away from this schedule creates a custom-count scenario.
Its clock times are unspecified until a corresponding schedule is supplied.
Filter edges describe the intended acoustic pipeline; this workload simulator does
not execute filtering or establish pest-classification accuracy on recorded signals.

## Hardware distinctions

The purchased **SX1278 at 433 MHz** is the baseline. The **RFM95W/SX1276 at 915 MHz**
remains a different-hardware comparison based on the original proposal. Semtech
specifies SX1278 operation over 137–525 MHz, so a 915 MHz comparison must never be
presented as a tuning option for the purchased SX1278. Chip-family typical
sensitivity/current data are planning references, not measurements of these modules.

The purchased **MCP602E** must not be silently substituted with the report's MCP6002.
The MCP602 datasheet specifies 2.7–6.0 V operation and typical quiescent current of
230 microamps per amplifier. If both channels are powered, their no-load typical
sum is about 460 microamps, before external circuitry and load. This is not the
current of the entire sensing board and is not automatically added to a recording
current input that already includes the front end.

## Purchased XL6009 power path remains unresolved

The purchased XL6009 step-up module is distinct from the planned Pololu S7V8F3
baseline above. It is not part of the simple 3.3 V power path. Its battery-to-board
wiring and usable operating range remain unresolved; no efficiency or protection
performance is inferred from the product listing.

In the separate advanced regulated-rail comparison, numerical inputs for regulator efficiency, regulator quiescent current,
recording current, sleep current, TX power and antenna gain remain **editable
exploratory placeholders**. In particular, 10 microamps is not a measured sleep
current for the purchased development board or the complete assembled node.
Enter the complete powered sensing/MCU circuit's recording and sleep currents;
regulator quiescent and radio sleep currents are accounted separately by the model.
Continuously powered amplifier/module/LED loads must not disappear during sleep.

Battery lifetime remains conditional on these unknowns. The next most useful
evidence is the final power schematic and measured battery-side sleep/recording
current, followed by radio current and antenna/link observations.

## Primary references

- [Semtech SX1278 product and family datasheet](https://www.semtech.com/products/wireless-rf/lora-connect/sx1278)
- [Microchip MCP602 product](https://www.microchip.com/en-us/product/mcp602)
- [Microchip MCP601/1R/2/3/4 datasheet DS20001314H](https://ww1.microchip.com/downloads/aemDocuments/documents/APID/ProductDocuments/DataSheets/MCP601-1R-2-3-4-2.7V-to-6.0V-Single-Supply-CMOS-Op-Amps-DS20001314.pdf)

No authoritative WANGCL module schematic or battery-model datasheet was supplied.
