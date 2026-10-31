# CSV calibration and scenario inputs

Import UTF-8 CSV through ESP32 Settings. The selected kind determines required
columns. Additional columns are ignored. Missing/nonfinite/nonnumeric values are
rejected rather than repaired. Retain original laboratory files with instrument,
voltage, CPU clock, backend, sample rate, antenna and firmware metadata.

| Kind | Required columns | Effect |
|---|---|---|
| Inference | `latency_ms,current_ma` | Mean latency and duration-weighted mean current; preserves mean event energy |
| Current profile | `time_ms,current_ma` | Strictly increasing times beginning at 0, trapezoidal rail energy; at least two samples |
| RSSI | `distance_m,rssi_dbm` | Fits exponent, excess vegetation loss and shadow standard deviation for the current link-budget assumptions |
| Packet success | `sent,received` | Aggregate received/sent baseline probability; counts must be valid integers |
| Trees | `x,y` | Orchard-local metre coordinates; switches to Custom physical layout |
| Gateways | `x,y` | Sets explicit gateway positions and count |

RSSI requires at least three observations over two or more distances, all at or
above the reference distance. A nonpositive slope or negative excess loss rejects
the calibration: review antenna gains/reference assumptions rather than silently
clipping the result. This simple regression does not model terrain or spatial
correlation. Use same-position replicate measurements for a better shadow estimate.

The packet baseline is multiplied by modeled channel success. Import packet data
measured under strong-signal conditions to avoid double-counting channel losses, or
set shadowing/required-margin assumptions to reflect what the baseline includes.

The current profile covers the CNN stage only and is rail-side current. Include
all MCU inference current in this waveform; do not add it again as a separate stage.
Adjust voltage to the actual measurement supply. Manual latency/current editing or
a performance sweep clears the waveform and removes the measurement/benchmark label.

Published benchmark profiles require the user to provide actual values and a source
note for matching hardware, workload, precision, clock and backend. There is no
bundled “actual measured” or “published” dataset masquerading as a benchmark.

Scenario JSON saves the full schema under `{"schema_version":1,"config":{...}}`.
Partial configuration JSON is accepted and fills omitted fields with defaults;
unknown fields, unsupported settings and malformed arrays reject the import.
Full result JSON stores source/limitation provenance and numeric tables. Undefined
or unbounded modeled outputs become JSON `null`, with an explanatory provenance field.
