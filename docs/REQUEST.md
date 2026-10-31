# MASTER ZERO-SHOT PROMPT — VERSION 2
## ESP32 + SX1276 LoRa + TinyML + Palm Plantation Digital Twin

### 1. Your Role and Mission

Act as a senior embedded systems engineer, LoRa RF specialist, TinyML researcher, energy systems engineer, wireless-network architect, scientific computing developer, and full-stack software engineer.

**Build a complete, functional, interactive engineering simulation platform for our Senior Design project: Early Red Palm Weevil Detection using Acoustic Sensing.**

The application must simulate the relationships between radio configuration, communication range, plantation coverage, connected trees, CNN inference performance, ESP32 energy consumption, battery life, gateway capacity, and server energy.

This is not a request for a static dashboard or simulation proposal. Build and test the functioning application.

Use the attached **Senior Design 1 Report.pdf** as the project reference.

Do not ask preliminary questions. Implement reasonable initial defaults, expose uncertain assumptions, and clearly distinguish calculated results, estimates, manufacturer specifications, and actual measurements.

Ignore radio regulations and legal transmission restrictions for this simulation. Focus on engineering and physical feasibility.

---

# 2. The Three Main Simulation Engines

Build three fully interconnected simulators.

**Simulator A — SX1276 LoRa and Plantation Coverage**

Users configure:
- Carrier frequency: **433 MHz or 915 MHz**.
- Spreading factor: **SF7, SF8, SF9, SF10, SF11, SF12**, with SF6 supported as an advanced configuration where technically applicable.
- Transmit power: configurable dBm.
- Bandwidth.
- Coding rate.
- Antennas.
- Distance.
- Packet sizes.
- Gateway configurations.
- Tree density and plantation geometry.

The simulator must calculate how the selected radio configuration changes propagation, sensitivity, range, covered trees, supported nodes, airtime, electrical energy and battery life.

**Simulator B — ESP32 CNN and Electrical Performance**

Users configure:
- CNN parameter count.
- CNN architecture.
- MAC operations.
- INT8/FP32 model precision.
- ESP32 CPU clock.
- Tensor arena.
- Available RAM.
- **Inference latency in milliseconds, directly editable.**
- **Inference current in milliamps, directly editable.**
- Model execution backend.
- Preprocessing latency.
- Recording current.
- Deep-sleep current.

The simulator must calculate model feasibility, execution energy, memory requirements, and local-inference performance.

**Simulator C — Combined System Digital Twin**

Automatically integrate both simulators.

Calculate:
- Architecture A versus Architecture B.
- Coverage radius.
- Number of reachable trees.
- Number of successfully supportable nodes.
- Gateway requirements.
- SNR-triggered transmissions.
- Individual node energy.
- Plantation-wide energy.
- Server energy.
- Battery life.
- Inference break-even.
- Network capacity.
- Feasible and infeasible configurations.

**All three engines must share the same live simulation state. Changing any input must automatically recalculate every dependent result.**

---

# 3. Critical Requirement: Dynamic Connection Between SF, Range, Trees and Nodes

THIS IS A NON-NEGOTIABLE CORE FEATURE.

The number of simulated connected nodes must not remain fixed when the user changes spreading factor, frequency, transmit power or other link-related settings.

For example, changing SF7 to SF9 or SF12 must cause the simulator to recalculate:

1. Receiver sensitivity.
2. Link budget.
3. Maximum modeled communication distance.
4. Estimated coverage area.
5. Number of palm trees inside that coverage.
6. Number of possible sensor nodes within reach.
7. Number of nodes the gateway can actually support.
8. Expected successful transmissions.
9. Network utilization.
10. Total sensing-node energy.
11. Gateway/server workload.
12. Aggregate energy consumption.
13. Estimated lifetime and costs.

These calculations must be based on RF propagation and network capacity, not arbitrary mappings such as "SF7 = 500 nodes" or "SF12 = 8,000 nodes."

### 3.1 Fundamental distinction: reachable versus supportable nodes

Maintain separate variables:

- Total physical trees on the plantation.
- Total deployed sensing nodes.
- Trees within radio coverage.
- Nodes satisfying minimum link margin.
- Nodes that the gateway can accommodate within the transmission schedule.
- Successfully connected and served nodes.
- Nodes outside coverage.
- Nodes rejected because of network capacity.

Increasing SF generally improves receiver sensitivity, extending potential coverage.

However, increased SF also increases packet airtime, potentially reducing the number of transmissions that can fit within a fixed communication window.

Therefore:

**Reachable nodes can increase while network capacity decreases.**

The simulation must account for both effects.

Define the actual number of served nodes as the result of geographical reachability, available node population, gateway scheduling capacity, and link reliability.

Use a simplified `min(reachable nodes, supportable nodes)` approximation only where the network model justifies it. The advanced simulation should schedule actual reachable nodes and evaluate packet success.

### 3.2 Geographical coverage calculation

Provide a plantation area in hectares and optional geographic dimensions.

For an initial circular free-space approximation:

A_coverage = π × R²

Convert the result to hectares.

For a uniform tree density:

N_trees_covered = A_coverage_hectares × trees_per_hectare

Then account for:
- Plantation boundaries.
- Gateway position.
- Uneven tree distribution.
- Trees already covered by another gateway.
- Number of sensors per tree.
- Link-margin probability.
- Gateway capacity.

The actual count must be limited to physically existing trees.

Support both:
- Analytical area-density approximation.
- Explicit two-dimensional orchard layouts with simulated individual tree coordinates.

The explicit layout mode is preferred for meaningful network deployment analysis.

### 3.3 Interactive plantation visualization

Provide an interactive top-down plantation map showing:

- Palm tree positions.
- Gateway positions.
- Modeled radio coverage.
- Reachable trees.
- Successfully served nodes.
- Nodes beyond range.
- Nodes within range but not served because of gateway congestion.
- Signal strength or link margin.
- Optional interference zones.

Use different colors or symbols for reachability and network-service status.

When spreading factor changes, update the coverage visualization immediately.

If SF7 reaches fewer trees than SF12 under the selected model, this must be visible.

If SF12 reaches more trees but cannot schedule all of them, display that limitation separately.

### 3.4 Compare radio configurations geographically

Provide a side-by-side comparison:

| Metric | SF7 | SF9 | SF12 |
|---|---|---|---|
| Receiver sensitivity | Calculated/reference | Calculated/reference | Calculated/reference |
| Coverage radius | Modeled | Modeled | Modeled |
| Covered area | Derived | Derived | Derived |
| Reachable trees | Derived | Derived | Derived |
| Network transmission capacity | Derived | Derived | Derived |
| Actually served nodes | Derived | Derived | Derived |
| Packet airtime | Exact | Exact | Exact |
| Energy per TX | Calculated | Calculated | Calculated |
| Gateway requirements | Calculated | Calculated | Calculated |

Never hardcode the results.

---

# 4. Radio Frequency Selection: 433 or 915 MHz

Provide a prominent radio-frequency selector:

**433 MHz | 915 MHz**

Default: 915 MHz.

Changing frequency must update:
- Appropriate HopeRF radio module variant.
- Relevant manufacturer RF characteristics.
- Free-space path loss.
- Antenna parameters, where frequency-dependent.
- Receiver sensitivity, if applicable.
- Propagation estimates.
- Modeled maximum range.
- Plantation coverage.
- Reachable trees.
- Connected nodes.
- Network and total energy results.

Use RFM95W characteristics for 915 MHz and an appropriate 433 MHz SX1276-based variant such as RFM96W/RFM98W for 433 MHz.

Use manufacturer specifications for the selected variant.

Do not model frequency changes as automatically changing transmit current or receiver sensitivity without a documented basis.

The standard free-space relationship must be implemented with consistent units:

FSPL(dB) = 32.44 + 20log10(d_km) + 20log10(f_MHz)

At equal distance and equal antenna gains, 433 MHz has approximately 6.5 dB less free-space path loss than 915 MHz.

This is a free-space comparison, not a guarantee of a 433 MHz range advantage in every plantation.

Account for antenna gains, practical antenna dimensions, foliage, palm trunks, installation height and environmental losses where appropriate.

**Both frequency selections must be independently simulated using their corresponding hardware assumptions.**

---

# 5. SX1276 Spreading Factor and Radio Configuration

### Required controls

- SF7–SF12, plus optional SF6 advanced mode.
- BW: supported LoRa bandwidth options.
- CR: 4/5, 4/6, 4/7, 4/8.
- TX output power in dBm.
- Current draw in mA.
- Carrier frequency: 433/915 MHz.
- Antenna gain.
- Antenna height.
- Cable loss.
- Preamble symbols.
- Header mode.
- CRC.
- Low-data-rate optimization.
- Radio standby and sleep currents.
- MCU current during radio transmission.
- Optional acknowledgment and retransmission settings.

Each control must have a numeric input or selector and valid hardware-specific limits.

### Receiver sensitivity

Use SX1276 datasheet sensitivity data where applicable.

Sensitivity must depend on the selected:
- SF.
- Bandwidth.
- RF band and module.
- Receiver configuration.

Do not use an arbitrary fixed radius per spreading factor.

### LoRa airtime

Implement the Semtech SX1276 airtime equations exactly.

Regression tests:

At SF7, BW125, CR4/5, eight preamble symbols, explicit header and CRC:

- 214-byte packet: 338.176 ms.
- 13-byte packet: 46.336 ms.

Use these as validation cases, not fixed runtime values.

### Payload controls

Provide presets:
- 214-byte MFCC packet.
- 13-byte inference-result packet.
- 8-byte minimal result packet.
- User-defined packet size.

Allow editing metadata, node IDs, timestamps, battery telemetry, classification information, and protocol overhead.

Calculate actual radio airtime and energy for every payload.

---

# 6. Propagation and Range Simulation

Create at least three propagation modes:

**Mode 1 — Free-Space Theoretical**

Use Friis/free-space path loss.

Clearly label this an idealized model.

**Mode 2 — Log-Distance Plantation Model**

Use a configurable environmental path-loss exponent, reference distance, shadowing standard deviation, and additional vegetation losses.

**Mode 3 — Monte Carlo Plantation Model**

Generate individual tree positions and randomized link conditions, including log-normal shadowing and configurable obstruction losses.

Optionally allow measured RSSI/path-loss observations to calibrate the model.

### Link-budget calculation

P_RX = P_TX + G_TX + G_RX − L_path − L_misc

LinkMargin = P_RX − ReceiverSensitivity − RequiredMargin

Allow users to set the required link margin.

Define reachable nodes according to the chosen link-reliability criterion.

For probabilistic reception, use the selected statistical channel model. Do not present a threshold approximation as an empirically validated packet-error-rate curve.

### Important coupling requirement

When SF changes:

Receiver sensitivity changes → communication feasibility changes → geographical coverage changes → reachable trees change → candidate sensing-node population changes.

The simulator must then independently evaluate whether radio airtime and scheduling permit service to all those nodes.

When TX power changes, the same dependency chain applies.

When the user switches from 915 MHz to 433 MHz, rerun the propagation, coverage and node-capacity calculations using the selected hardware assumptions.

---

# 7. Plantation and Network Layout

Provide configurable plantation parameters:

- Total orchard area in hectares.
- Orchard width and length.
- Number of palm trees per hectare.
- Tree spacing.
- Arrangement: grid, random or custom.
- Number of sensing nodes per tree.
- Number of gateways.
- Gateway placement.
- Gateway channels and available receivers.
- Percentage of trees equipped with nodes.
- Minimum acceptable link reliability.

Initial exploratory defaults:
- 100-hectare plantation.
- 100 trees per hectare.
- One sensor node per tree.
- One gateway.
- 915 MHz.
- SF7.
- +17 dBm.
- 125 kHz bandwidth.

The density is a user-adjustable simulation assumption, not a verified value for the target plantation.

Show separate counts for:
- 10,000 total physical trees under these defaults.
- Deployed sensing nodes.
- Nodes in range.
- Scheduled nodes.
- Successfully served nodes.

Users may override the plantation dimensions or import actual tree coordinates.

A change in SF or frequency must change the derived covered/served nodes while keeping the physical plantation population unchanged.

Do not silently increase the number of trees physically planted merely because range increases.

---

# 8. Network Capacity and TDMA Simulation

Implement:
- Dedicated fixed TDMA.
- Dynamic TDMA.
- Event-driven transmissions.
- Multi-channel LoRa.
- Gateway receiver-resource allocation.
- Optional ALOHA-style contention.
- Retransmissions.
- Guard intervals.
- Clock synchronization error.
- Gateway receiver occupancy.
- Packet-delivery probability.

The original report specifies 12 recording opportunities per day, including hourly nighttime activity, and a 900 ms TDMA slot.

The simulator must flag that:

8,000 nodes × 900 ms = 7,200 seconds.

This cannot fit into a one-hour nighttime round on one dedicated channel.

Calculate:
- Reserved slot utilization.
- Actual radio airtime utilization.
- Peak reporting-window demand.
- Maximum feasible serviced nodes.
- Minimum required channels or gateways.
- Average and worst-case packet latency.
- Congestion and retry effects.

**A larger SF must simultaneously update both coverage potential and the capacity penalty caused by longer airtime.**

Display the optimal SF not solely by range, but by how many nodes can be reliably served within the requirements.

---

# 9. ESP32-WROOM-32 Configuration

Model the original ESP32-WROOM-32 with no PSRAM.

Provide controls for:
- CPU frequency: 80, 160, 240 MHz.
- CPU utilization.
- Active current.
- Sleep current.
- Wake current and time.
- Operating voltage.
- Flash storage.
- Internal data RAM.
- Firmware overhead.
- Stack usage.
- Audio buffers.
- ADC/DMA buffers.
- Tensor arena.
- Operator scratch memory.
- Allocatable contiguous memory.

Model limitations using the actual ESP32 memory organization, not a single freely allocatable 520 KB pool.

---

# 10. CNN Model Designer

Allow users to configure CNN architecture, including:

- Input dimensions.
- Convolution layers.
- Filter counts.
- Kernel sizes.
- Strides.
- Padding.
- Pooling.
- Dense layers.
- Activations.
- Quantization.
- INT8 versus FP32.
- Model parameter count.
- MAC operations.

Include experimental workload presets:
- 50K parameters.
- 100K parameters.
- 250K parameters.
- 350K parameters.
- 500K parameters.
- 1M parameters.

Use the report's 15×13 MFCC features as the default model input.

Include a reference INT8 CNN with approximately 260,833 parameters and 11,283,328 MACs.

Do not assume parameter count determines inference duration, energy, or memory requirements.

Calculate these independently.

### Memory feasibility

Show:
- Weight storage.
- Serialized model size.
- Peak live activations.
- Tensor-arena requirements.
- Scratch buffers.
- Available heap.
- Contiguous allocation requirements.
- Stack allocation.
- Audio-buffering overhead.

Classify models as:
- Verified feasible.
- Plausibly feasible but unverified.
- Infeasible.

Never claim a model is verified merely because its weights fit in flash.

---

# 11. Critical Requirement: Editable CNN Inference Latency and Current

The user MUST be able to directly configure:

**CNN Inference Latency (ms)**

**CNN Inference Current (mA)**

These must be independent, prominent interactive controls.

Recommended slider ranges:

Inference latency:
- Minimum: 1 ms.
- Maximum: 10,000 ms.
- User-entered custom values supported.

Inference current:
- Minimum: 1 mA.
- Maximum: 500 mA for exploratory scenarios.
- User-entered custom values supported.

Do not assume all exploratory values correspond to realistic ESP32 operation.

Provide presets:
- Fast inference.
- Medium inference.
- Slow inference.
- Low-current inference.
- High-current inference.
- Published benchmark profile.
- Actual measured profile.
- Custom user profile.

Default hypothetical scenario:
- Latency: 800 ms.
- Current: 70 mA.
- Voltage: 3.3 V.

Compute:

E_CNN(mJ) = V × I_mA × t_ms / 1000

For the default:

E_CNN = 3.3 × 70 × 800 / 1000 = 184.8 mJ

Changing latency or current must immediately update:
- CNN inference energy.
- Architecture B cycle energy.
- Daily and yearly energy.
- Battery lifetime.
- Break-even conditions.
- Optimal radio configuration.
- Feasibility under deadlines.
- Overall system energy.
- Configuration rankings.

For example:

Changing inference latency from 800 ms to 200 ms at 70 mA must change inference energy from 184.8 mJ to 46.2 mJ.

Changing inference current from 70 mA to 100 mA at 800 ms must change inference energy from 184.8 mJ to 264 mJ.

Changing these manual overrides must not silently alter the architecture's parameter count or claim that new performance was physically benchmarked.

Support measured current-vs-time profiles as an advanced input.

---

# 12. Acoustic Sensing Pipeline

Model the common pipeline:

1. Wake up.
2. Enable analog front end.
3. Record 10 seconds at 8 kHz.
4. Bandpass filter.
5. Calculate SNR and burst statistics.
6. Apply 4 dB SNR gate.
7. Extract MFCCs if accepted.
8. Run CNN locally only for Architecture B.
9. Transmit the appropriate packet.
10. Return to sleep.

Allow independent duration/current controls for each stage.

The default rejected-window behavior must skip MFCC extraction, CNN inference and event transmission in both architectures.

Include:
- Processing timelines.
- Current-versus-time graphs.
- Energy-per-stage breakdown.
- Peak memory usage.
- Active/sleep durations.

---

# 13. Architecture A Versus Architecture B

Architecture A sends approximately 214 bytes of MFCC features for server inference.

Architecture B runs the quantized CNN locally and transmits a small result packet.

Both use the same acoustic input, preprocessing, quantization, model and probability thresholds as far as technically possible.

Compute:

E_A = E_shared + E_MFCC + E_TX214

E_B = E_shared + E_MFCC + E_CNN + E_TXresult

Include differences in radio state, MCU state, startup overhead and sleep energy in the complete calculation.

Core break-even:

E_CNN < E_TX214 − E_TXresult

Display:
- Exact transmission-energy savings.
- CNN inference-energy cost.
- Net gain or loss.
- Maximum allowable inference time.
- Maximum allowable inference current.
- Battery-life differences.
- Difference per triggered event.
- Difference per full measurement cycle.

Every comparison must update when SF, frequency, TX power, CNN latency or CNN current changes.

---

# 14. Dynamic Multi-Factor Dependency Engine

Implement a centralized dependency graph or equivalent deterministic calculation pipeline.

The user must not need to manually recalculate dependent values.

Example 1:

**Change SF7 → SF12**

Automatically recalculate:

SF → receiver sensitivity → range → covered area → reachable trees → potentially connected nodes.

Simultaneously:

SF → packet airtime → radio TX energy → packet scheduling capacity → supportable nodes.

Then:

Supported node count + event rate → total transmissions → network utilization → gateway/server load.

Finally:

Radio energy + CNN energy + shared sensing energy → battery lifetime and system energy.

Example 2:

**Change 915 MHz → 433 MHz**

Automatically recalculate:
- Selected radio hardware profile.
- RF propagation.
- Link margin.
- Coverage.
- Reachable trees.
- Supported nodes.
- Relevant hardware power assumptions.
- Overall energy and network results.

Example 3:

**Change CNN latency 800 ms → 300 ms**

Automatically recalculate:
- Inference energy.
- Cycle energy.
- Break-even outcome.
- Battery lifetime.
- Architecture ranking.

Do not change physical tree positions or radio coverage from CNN latency alone unless a separately modeled timing, scheduling or node-feasibility constraint actually causes a secondary effect.

Example 4:

**Change CNN inference current 70 mA → 120 mA**

Automatically recalculate:
- Inference power.
- Inference energy.
- Battery consumption.
- Architecture preference.
- Energy optimization results.

The interface should include a dependency-inspection panel so researchers can see which variables changed and why.

---

# 15. Battery and Power Simulation

User-configurable inputs:
- Battery voltage.
- Battery capacity.
- Regulator efficiency.
- Quiescent current.
- Battery usable capacity.
- Temperature-dependent capacity.
- Battery aging.
- Self-discharge.
- Optional solar energy.
- Recording count per day.
- SNR-trigger probability.
- Transmit retry rate.
- Sleep current.
- Inference current and duration.

Calculate:
- Energy per accepted recording.
- Energy per rejected recording.
- Energy per scheduled cycle.
- Daily energy.
- Monthly energy.
- Annual energy.
- Battery lifetime.
- Average power.
- Battery-side current.
- Energy consumed by all served nodes.

Default battery:
3.7 V, 3,000 mAh.

Provide SNR trigger presets:
0%, 10%, 25%, 50%, 75%, 90%, 100%.

Do not mix battery-side energy with regulated-rail energy without accounting for power conversion correctly.

---

# 16. Server and Gateway Energy

Model:
- Raspberry Pi 4 idle and processing power.
- Continuous LoRa RX.
- Wi-Fi/4G backhaul energy.
- Gateway counts.
- Server idle power.
- Server inference power.
- Server inference latency.
- Expected inference requests.
- Model storage and compute resources.

The server performs inference only in Architecture A.

Calculate server energy per inference and total energy for:
- 100 served nodes.
- 1,000 served nodes.
- 8,000 served nodes.
- Automatically derived served-node counts.

Distinguish:
- Energy consumed by one ESP32 battery.
- Energy consumed by all ESP32 batteries.
- Gateway energy.
- Central server energy.
- Entire deployment energy.

---

# 17. Interactive Graphs

Implement the following connected visualizations.

### Radio and coverage graphs

- SF versus range.
- SF versus coverage area.
- SF versus reachable trees.
- SF versus actually supported nodes.
- SF versus airtime.
- SF versus radio energy.
- SF versus maximum gateway capacity.
- Frequency versus modeled range.
- Frequency versus path loss.
- TX dBm versus modeled range.
- TX dBm versus energy.
- Payload size versus airtime.
- Distance versus link margin.
- Reachable-node versus served-node comparison.

### CNN graphs

- Model parameters versus weight storage.
- MAC count versus estimated inference latency.
- Inference latency versus energy.
- Inference current versus energy.
- Model size versus available RAM.
- Tensor-arena usage.
- CPU frequency versus estimated inference performance.

### Combined comparison graphs

- Architecture A versus B energy.
- CNN energy versus LoRa energy saved.
- Battery life versus SF.
- Battery life versus CNN latency.
- Battery life versus inference current.
- Battery life versus transmit power.
- Battery life versus event-trigger rate.
- Total energy versus number of connected nodes.
- Break-even inference time versus SF.
- Break-even inference current versus SF.
- Gateway capacity versus SF.
- Number of trees reached versus number of trees actually serviceable.

### Advanced research graphs

- Multidimensional heatmaps.
- Three-dimensional response surfaces.
- Pareto-front plots.
- Tornado sensitivity charts.
- Monte Carlo probability distributions.
- Energy waterfall charts.
- RF coverage maps.
- Packet scheduling timelines.
- Battery-current waveforms.

All charts must be interactive and recalculate from current inputs.

---

# 18. Massive Parameter Sweep and Optimization

Allow users to select several parameters and automatically explore thousands of configurations.

Examples:

SF = 7 through 12.

Frequency = 433 or 915 MHz.

TX power = selectable supported hardware range.

CNN latency = 100 through 3,000 ms.

Inference current = 30 through 150 mA.

SNR trigger rate = 0–100%.

Model parameters = 50K–1M.

Number of gateways = 1–20.

Plantation size = 1–1,000 hectares.

Tree density = configurable.

Support:
- Full factorial search.
- Random search.
- Latin hypercube sampling.
- Monte Carlo analysis.
- Sensitivity analysis.
- Multi-objective optimization.
- Pareto-front extraction.

Optimize for:
1. Maximum battery lifetime.
2. Maximum reliably served trees.
3. Maximum coverage.
4. Minimum energy per sensing node.
5. Minimum energy per served tree.
6. Minimum gateway count.
7. Minimum network airtime.
8. Maximum practical CNN complexity.
9. Minimum entire-system energy.
10. Best balance among range, node count, battery life and model performance.

One especially important optimization objective is:

**Maximize the number of successfully monitored trees while maintaining the specified battery-life target and packet-delivery reliability.**

Do not maximize SF alone.

A lower SF may cover fewer trees but support more scheduled transmissions, whereas a higher SF may reach more trees but overload the communication schedule.

The optimizer must determine the trade-off quantitatively.

---

# 19. Feasibility and Impossible Configurations

Classify each configuration as:

- Verified feasible.
- Conditionally feasible.
- Infeasible.

Check:
- Supported radio configuration.
- Appropriate module for frequency.
- Payload-size limitations.
- Receiver sensitivity.
- Required link margin.
- CNN operator compatibility.
- Internal memory constraints.
- Tensor-arena requirements.
- Real-time execution deadlines.
- Battery-life requirements.
- Gateway capacity.
- TDMA scheduling.
- Packet reliability.
- Physical plantation coverage.

Distinguish hard hardware failures from uncertain modeled failures and user-target failures.

Every rejected configuration must display an explanation.

Examples:

"This model cannot allocate its required tensor arena in the available contiguous internal memory."

"SF12 reaches 85% of the modeled plantation but exceeds the available transmission scheduling capacity."

"SF7 meets network-capacity requirements but fails the required link margin for 20% of deployed nodes."

"Local inference consumes more energy than the radio transmission saving."

"Architecture B meets the energy target only when measured inference latency is below the calculated break-even duration."

---

# 20. User Interface

Use a professional web dashboard with Python scientific calculation engines, preferably Streamlit and Plotly.

Include:

**Top-level tabs**
1. Overview
2. Radio Settings
3. Plantation Map
4. Gateway and Network Capacity
5. ESP32 Settings
6. CNN Model Designer
7. Architecture Comparison
8. Battery Simulator
9. Sensitivity Analysis
10. Optimization Laboratory
11. Assumptions and Sources
12. Export Results

Prominent controls on the main dashboard must include:
- **433 / 915 MHz selector**
- **SF7–SF12 selector**
- **Transmit-power dBm slider**
- **CNN latency slider (ms)**
- **CNN inference-current slider (mA)**
- **Plantation size**
- **Trees per hectare**
- **Gateway count**
- **SNR-trigger probability**

These are the highest-priority interactive inputs.

Changing any of them must visibly and automatically update all dependent results.

Use:
- Beginner Mode.
- Engineering Mode.
- Research Mode.

Provide tooltips, formulas and assumptions for technical values.

Use a consistent unit system.

---

# 21. Scenario Presets

Include:

1. Original Senior Design — 915 MHz, SF7.
2. 433 MHz, SF7.
3. 433 MHz, SF9.
4. 433 MHz, SF12.
5. 915 MHz, SF9.
6. 915 MHz, SF12.
7. Fast CNN, 200 ms.
8. Moderate CNN, 800 ms.
9. Slow CNN, 2,000 ms.
10. Low-current CNN.
11. High-current CNN.
12. Small plantation.
13. Large plantation.
14. Sparse-tree plantation.
15. Dense-tree plantation.
16. 100-node deployment.
17. 1,000-node deployment.
18. 8,000-node candidate deployment.
19. One-gateway deployment.
20. Multi-gateway deployment.

These presets must share the same calculations and models as manual configurations.

---

# 22. Sources and Scientific Integrity

Use:
- Semtech SX1276 datasheet.
- HopeRF RFM95W/RFM96W/RFM98W documentation.
- Espressif ESP32 documentation.
- ESP-NN benchmarks.
- TFLite Micro documentation.
- Peer-reviewed wireless propagation studies where available.
- The attached Senior Design report.

Never fabricate:
- CNN inference benchmarks.
- Electrical current measurements.
- RF propagation measurements.
- Tensor-arena results.
- Receiver sensitivity values.
- Battery performance measurements.

Label results as:
- Manufacturer specified.
- Measured.
- Published benchmark.
- Calculated.
- Estimated.
- Exploratory assumption.

Support importing measured ESP32 current, inference latency, RSSI and packet-success data.

Allow the measured data to calibrate and replace exploratory assumptions.

---

# 23. Validation and Acceptance Tests

The application must pass at least these tests.

**Test 1: Spreading factor dependency**

Change SF7 to SF12.

Verify that receiver sensitivity, radio propagation, modeled range, geographical coverage, reachable trees, airtime, energy, gateway capacity and served-node counts are recalculated.

**Test 2: Frequency dependency**

Change 915 MHz to 433 MHz.

Verify correct module selection and RF propagation recalculation.

Ensure model assumptions are not falsely treated as measured results.

**Test 3: CNN latency**

Change 800 ms to 200 ms at 70 mA and 3.3 V.

Expected inference energy changes from 184.8 mJ to 46.2 mJ.

**Test 4: Inference current**

Change 70 mA to 100 mA at 800 ms and 3.3 V.

Expected inference energy changes from 184.8 mJ to 264.0 mJ.

**Test 5: LoRa airtime**

At SF7, BW125, CR4/5, CRC enabled and explicit header:

214 bytes → 338.176 ms.

13 bytes → 46.336 ms.

**Test 6: Network-capacity conflict**

8,000 nodes × 900 ms = 7,200 seconds.

Flag the inability to fit this dedicated TDMA round within one hour on one channel.

**Test 7: Geographical node population**

When radio range decreases, fewer simulated tree positions should satisfy the link requirements for a fixed gateway and fixed orchard.

The available physical tree population must remain unchanged.

**Test 8: SF capacity trade-off**

Test a case where higher SF increases reachable trees but reduces maximum serviceable nodes because of longer airtime.

Verify the system reports both results separately.

**Test 9: CNN memory**

A model whose required tensor arena exceeds available contiguous internal RAM must not be marked verified feasible.

**Test 10: Full-system consistency**

Changing CNN inference current must affect battery energy but not RF propagation.

Changing SF must affect radio airtime, link budget, range and network capacity without silently changing the CNN's parameter count.

Changing tree density must affect the number of covered trees without changing radio signal propagation itself.

---

# 24. Final Deliverables

Deliver:
1. A working interactive application.
2. Source code for every simulation engine.
3. Installation and launch instructions.
4. Documented formulas.
5. Verified radio airtime calculations.
6. Configurable plantation simulation.
7. Dynamic SF-to-range-to-trees-to-nodes coupling.
8. Frequency comparison between 433 and 915 MHz.
9. Editable CNN inference latency and current.
10. CNN memory and deployment feasibility.
11. Complete energy and battery models.
12. Interactive maps and graphs.
13. Automated optimization.
14. Monte Carlo and sensitivity analysis.
15. Scenario saving and loading.
16. CSV, JSON and research-report exports.
17. Automated tests.
18. Clear measurement gaps and technical limitations.

Run the application and its tests. Resolve implementation failures before considering the work complete.

---

# 25. The Final Research Objective

Build a simulator capable of answering:

**"For a plantation of a given size and tree density, which combination of frequency, spreading factor, transmit power, CNN architecture, inference current, inference time and gateway count can successfully monitor the largest number of trees while using the least battery energy?"**

Also determine:

- When a smaller SF is better.
- When a larger SF is better.
- How 433 MHz compares with 915 MHz.
- How coverage affects node count.
- How airtime limits network capacity.
- How CNN latency affects battery life.
- How inference current affects architecture selection.
- When centralized inference wins.
- When ESP32 inference wins.
- Which configurations cannot work.
- Which uncertain measurements matter most.

**The application must connect the entire causal chain:**

Radio configuration → receiver sensitivity and airtime → range and coverage → reachable trees → gateway scheduling and supported nodes → transmissions and processing demand → node energy → battery life → deployment-wide energy.

CNN architecture → memory and compute requirements → inference performance and current → inference energy → architecture break-even → battery life.

The application must not assume that larger SF, fewer transmitted bytes, larger models or edge inference are automatically better.

Use physics, documented specifications, reproducible calculations, explicit assumptions and measured results when available.

**Build the complete functional simulation platform now.**

If you are about to run out of credits (in other words, you have 7% or less), stop working, create a concise summary for the next step, and schedule it for after the reset.
