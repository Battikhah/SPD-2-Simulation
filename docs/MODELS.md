# Formulas, units and model boundaries

The app's **Assumptions and Sources** tab has a searchable formula guide with
the equation, its purpose, constants and evidence, plus examples using the current
scenario. Related tabs include expandable calculation notes. The full input ledger
shows current values alongside defaults; a changed or manually labeled value is
not automatically a verified measurement.

The guide is maintained in `simulator/explanations.py`. Unknown circuit currents,
losses and efficiency remain explicit assumptions. In particular, the 40 mA
recording-chain input and 3.3 V rail are not confirmed measurements. The 4 dB SNR
threshold does not calculate the separate trigger-probability input.

The engine retains the conventional FSPL constant 32.44 for km/MHz, obtained using
approximately 3×10⁸ m/s for light speed. ITU-R P.525 equation (6) prints 32.4;
using the exact SI speed gives about 32.4478. This small rounding distinction is
documented without changing existing numerical results. The Gaussian-CDF
polynomial coefficients are numerical approximation constants; a primary citation
for that specific approximation is not recorded in this project.

All displayed predictions are calculated from manufacturer typical values or
explicit exploratory inputs. “Measured” describes user-supplied performance data,
not a certification of the application or hardware.

The [28 October hardware clarification](HARDWARE_BASELINE.md) supersedes the old
915 MHz / 3,000 mAh starting values in the original prompt. The purchased baseline
is SX1278 at 433 MHz with a 2,600 mAh battery per sensing node. The old 915 MHz
scenario remains a different-hardware comparison. All unmeasured electrical inputs
remain assumptions; inventory quantities never scale node battery capacity.

## RF

`FSPL_dB = 32.44 + 20 log10(distance_m / 1000) + 20 log10(frequency_MHz)`.
At equal distance/gains, the 915-to-433 MHz FSPL difference is about 6.50 dB.
Module typical sensitivity additionally differs between documented HF/LF receiver
conditions. Changing frequency does not independently scale TX current.

`P_RX = TX_dBm + G_TX + G_RX − cable_loss − miscellaneous_loss − interference_loss − path_loss`.
`margin = P_RX − sensitivity − required_margin`.

Free-space is an idealized baseline. Log-distance uses
`PL(d)=FSPL(d0)+10 n log10(max(d,d0)/d0)+vegetation_loss`.
The channel probability is `P(margin − shadowing − obstruction >= 0)`, with
Gaussian shadowing in dB. Monte Carlo adds a seeded static channel realization
and obstruction mixture after long-run reliability eligibility. Long-run and
sampled counts are exported separately. Retries assume independent packet outcomes;
correlated fading would reduce their benefit.

Range is solved from the same long-run probability criterion by binary search.
Slant distance includes antenna-height difference; displayed radius is horizontal.
No calibrated height-dependent foliage improvement or terrain model is assumed.

Unbounded theoretical area is `πR² / 10,000` hectares. Actual coverage counts use
individual tree coordinates and the union of qualifying gateway links. Bounded
area is `orchard_area × covered_tree_fraction`, a uniform-density proxy (not an
exact geometric intersection for uneven/custom layouts). Duplicate gateway
coverage never duplicates a physical tree.

The grid preserves area/density-derived population. Requested tree spacing controls
row/column arrangement; actual spacing can differ when density and spacing conflict.
Custom coordinates define the physical population directly. Coordinates are local
metres, not latitude/longitude. Dimensions override area only when enabled.

## Airtime

`T_symbol = 2^SF / BW_Hz` seconds.
`N_payload = 8 + max(ceil((8PL − 4SF + 28 + 16CRC − 20IH) / (4(SF − 2DE))) × CR_denominator, 0)`.
`T_packet = (preamble_symbols + 4.25 + N_payload) × T_symbol`.

CR_denominator is 5,6,7,8 for 4/5…4/8. IH is 1 for implicit header. DE follows
symbol times above 16 ms in automatic mode. SF6 requires implicit headers and
matching receiver setup. Payload includes all editable metadata/protocol overhead;
the raw payload limit is 255 bytes. Fragmentation is not silently synthesized.

Sensitivity comes from the band/BW/SF-specific HopeRF typical tables, not a fitted
fixed-radius mapping. Current interpolates documented 7/13/17/20 dBm anchors and
is explicitly estimated between anchors. A numeric override can replace it.

## Scheduling

`resources_per_gateway = min(channels, receivers)`.
`minimum_slot = (retries+1) × (packet_airtime + ACK_airtime + ACK_wait) + guard + 2 clock_error`.
Fixed TDMA uses `max(configured_slot, minimum_slot)`; dynamic/event modes use the
minimum. Fixed reservations include quiet nodes. Dynamic timelines use seeded
active node requests. Gateway association selects the strongest link; it does not
solve an optimal global reassignment problem.

`capacity_per_gateway = floor(window / slot) × resources`.
Each schedule is allocated to actual reachable node identifiers and receiver lanes.
Fixed TDMA can schedule 4,000 reservations of 0.9 s in one hour on one receiver.
8,000 such reservations need 7,200 s; the conflict is flagged. SF12 slots expand
when actual packet time exceeds the configured reservation.

### A larger plantation does not increase one cluster head's capacity

Here a cluster head means one modeled gateway receiving site, with its configured
receiver resources. An additional relay hierarchy, head packet buffers and
head-to-gateway forwarding queues are not modeled. The displayed ceiling is a
radio-schedule limit, not a measured hardware node limit.

Physical/deployed nodes can grow with hectares. Reachable nodes additionally pass
the link requirement. Admitted nodes must also fit the receiving head's reporting
window. Successful monitoring is lower still when delivery fails. Spare capacity
at another head does not automatically serve a node assigned to an overloaded head.

The all-reporting stress case assumes every reachable node triggers on every
recording round. For the supplied twelve-round daily schedule, one 4,000-slot
receiver has 48,000 report reservations across those rounds, supporting at most
4,000 nodes that each need all twelve reports. It does not support 48,000 such
nodes. Each hourly nighttime round must meet its own deadline; unused daytime
hours cannot repair a missed nighttime reporting deadline.

Current traffic and the all-reporting stress case are separate. The stress case
does not silently change the user's event probability or battery-energy scenario.
Dynamic/event-driven capacity based on mean activity is not a guarantee that all
nodes can report simultaneously. ALOHA has probabilistic collision loss rather
than a guaranteed TDMA admission limit.

Expected monitored population is admitted population times delivery probability.
Daily requests use expected successful reports per round × rounds/day, not one
frozen seeded event sample. Realized requests/successes are separate outputs.
Dynamic population capacity normalizes expected event demand; this is an approximation,
not a guarantee under simultaneous bursts. ALOHA uses `success_collision=exp(-2G)`
with no capture/inter-SF interference; its timeline depicts first attempts only.
Packets that cannot finish in the reporting window are rejected.

Without ACK, requested retry copies are all transmitted. With ACK, expected
attempts use the truncated geometric sum `Σ(1−p)^k`. Delivery after n copies is
`1−(1−p)^n`. ACK and forward path success are assumed independent. These estimates
are not an empirical PER curve. Margin and packet-delivery targets are separate.

## CNN and memory

Convolution parameters: `(kernel² × input_channels + 1) × filters` including bias.
MACs: `output_height × output_width × filters × kernel² × input_channels`.
Dense parameters: `(input_elements+1) × units`; MACs: `input_elements × units`.
Pooling and activations have no counted MACs in this estimator; their runtime is
not zero and remains part of independently entered inference timing.

Weights are parameters × bytes/weight. Serialized storage uses a labeled 5% plus
4 KiB metadata estimate. Sequential live activations use the largest adjacent
input/output pair; arena lower bound adds scratch plus 8 KiB planning metadata.
Tensor-arena input must satisfy both the estimated requirement and available heap/
largest contiguous block. The original ESP32 data budget is at most 320 KiB DRAM,
not the total 520 KiB SRAM. Code/IRAM, flash, stack and buffers are separate.

Weights are assumed mapped from flash. Audio assumes streamed processing buffers;
10 s × 8,000 samples/s × 2 bytes is 156.25 KiB if a full recording is retained.
Users must allocate that explicitly if their firmware is not streaming.

`estimated_latency_ms = MACs × cycles_per_MAC / (CPU_MHz × 1000 × utilization)`.
This estimate is independent of manual latency/current and is not an ESP-NN benchmark.
The reference 260,833 parameters / 11,283,328 MACs is a workload from the prompt;
an actual architecture and execution measurement were not supplied. Operator support
is a user declaration, never proof of a compiled operator resolver.

## Energy and battery

`E_stage_mJ = rail_voltage_V × current_mA × duration_ms / 1000`.
Current-profile energy uses trapezoidal integration. An imported profile supersedes
the scalar performance input until a manual performance override clears it.

Rejected windows run wake/front-end/record/SNR only. Accepted windows add MFCC,
optional local CNN and packet/ACK stages. Architecture A transmits MFCC features;
B transmits a result. Sleep covers only the remainder of the day, never a second
full day in addition to active time. Negative remaining sleep raises a feasibility
issue through the daily processing deadline.

`E_daily_rail = rounds × [(1−p) E_rejected + p E_accepted] + E_remaining_sleep`.
`E_daily_battery = E_daily_rail / regulator_efficiency + V_battery × I_quiescent_mA × 86.4` joules.
Battery capacity is `V × mAh × 3.6 × usable × temperature × aging` joules.
Self discharge uses `k=−ln(1−monthly_fraction)/30` per day. With net daily load L
after solar, lifetime is `ln(1+k E0/L)/k` (E0/L when k=0). If L<=0, lifetime is
unbounded in this idealized model, not a physical battery guarantee.

Per-node results describe a scheduled node. All deployed nodes still wake/process;
only admitted nodes add scheduled communication energy. Aggregate battery energy
counts physical admitted nodes before delivery loss, so lost packets do not make
batteries disappear. Solar offsets battery load but not energy actually consumed.

Gateways include Pi idle power plus continuous receiver current, incremental packet
processing and per-packet backhaul. Central server idle energy is separate; incremental
server inference occurs only for successfully delivered Architecture A requests.
Processing-power controls represent incremental power above idle. Infrastructure
power values are exploratory inputs, not manufacturer measurements for a complete Pi setup.

`E_CNN < E_TX_A − E_TX_B` is the single-attempt inference break-even. TX includes
MCU current during transmission. Full expected cycle savings additionally include
startup, retries, ACK windows and remaining-sleep differences.

## Search and feasibility

All search methods call the same calculation pipeline. Monte Carlo changes channel
seed; selected discrete parameter ranges are sampled rather than interpolated.
Invalid configurations remain rows with an explanation. Sweeping latency/current
clears an imported waveform and labels the new values exploratory.

Target-eligible monitored nodes satisfy battery-life, processing deadline, configured
delivery target and (for B) modeled memory/operator requirements. Full-deployment
feasibility additionally requires no out-of-range or capacity-rejected nodes.
The Pareto front maximizes eligible monitored trees while minimizing entire-deployment
energy and gateways. It is conditional on the chosen assumptions and search grid.

With multiple sensors per tree, a separate expected unique monitored-tree count
uses `1 − product(1 − admitted_sensor_delivery_probability)` per physical tree.
This assumes independent sensor packet outcomes, which is optimistic under shared
fading. The default optimizer and Pareto front use this unique-tree count so extra
sensors do not create extra physical trees. Node and tree objectives remain separate.

No accuracy objective is inferred from parameter count. Larger models have independently
editable MACs, activation requirements and latency/current; a complexity objective
alone cannot establish better pest detection.

## Primary references

- [Semtech SX1276 documents](https://www.semtech.com/products/wireless-rf/lora-connect/sx1276)
- [HopeRF RFM95W/96W/98W v2.0](https://www.hoperf.com/uploads/RFM95W-V2.0_1695351437.pdf)
- [Espressif ESP32 memory](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-guides/memory-types.html)
- [ESP-NN](https://github.com/espressif/esp-nn)
- [TFLite Micro arena management](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/memory_management.md)
