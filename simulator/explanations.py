"""Plain-language formula guide and provenance labels for the dashboard."""

from .config import DEFAULTS

FORMULA_GUIDE = [
    {
        "Topic": "Antenna length",
        "Formula": "Quarter-wave length (cm)=299792458/(frequency_MHz×1000000)×25.",
        "Why it is used": "It gives the ideal free-space quarter wavelength as an antenna-size reference. It does not predict the gain or tuning of the purchased antenna.",
        "Constants and source": "299792458 m/s is the exact SI speed of light; 25 converts a quarter wavelength in metres to centimetres. Real antenna construction and surroundings change tuning. [NIST SI constants](https://www.nist.gov/si-redefinition/meet-constants).",
    },
    {
        "Topic": "Radio range",
        "Formula": "FSPL = 32.44 + 20 log10(distance_km) + 20 log10(frequency_MHz); received margin = TX power + antenna gains − losses − path loss − receiver sensitivity − required margin.",
        "Why it is used": "It gives a consistent first estimate of whether a link has enough signal margin. Log-distance and shadowing options model additional non-free-space loss.",
        "Constants and source": "32.44 is the conventional rounded km/MHz approximation using c≈3×10⁸ m/s. ITU-R P.525 prints 32.4; using exact c gives about 32.45, a ~0.008 dB difference from the model constant. HopeRF v2.0 p.17 supplies lower-band RX current 11.5 mA and TX-current anchors 20/28/90 mA at 7/13/17 dBm; the +2 dBm anchor is clamped to 20 mA by this model. The 20 dBm/120 mA anchor uses the high-power manufacturer specification. HopeRF sensitivities are family proxies for the uncharacterized SX1278 board. Other path-loss and antenna inputs are assumptions unless measured.",
    },
    {
        "Topic": "LoRa airtime",
        "Formula": "Symbol time in ms=2^SF/BW_kHz (BW_Hz=1000×BW_kHz). Payload symbols=8+max(ceil((8PL−4SF+28+16CRC−20IH)/(4(SF−2DE)))×CRdenominator,0). Packet time=(preamble+4.25+payload symbols)×symbol time.",
        "Why it is used": "Radio airtime determines when a reporting slot is too short and estimates transmit energy. Payload-symbol count follows the Semtech LoRa modem equation.",
        "Constants and source": "The payload-symbol equation, 2^SF, coding-rate terms, 4.25 symbols, CRC/header terms and 255-byte limit follow Semtech SX1276 docs section 4.1.1.7; DE enables above a 16 ms symbol per 4.1.1.6. PL=payload bytes; CRC=1 when enabled; IH=1 for implicit header; DE=1 for low-data-rate optimization; CR denominator=5…8. The default 8-symbol preamble and payload sizes are editable assumptions, not measured packets. 1000 converts kHz to Hz and ms to seconds as used by code.",
    },
    {
        "Topic": "Plantation and coverage",
        "Formula": "Tree count = round(area_ha × trees_per_ha). Deployed nodes=round(tree count×equipped fraction)×sensors/tree, capped by a nonzero deployment limit; custom coordinates replace the area-derived tree population. With dimension override, area_ha = width_m × length_m / 10,000. Circle area = πR²/10,000 hectares. Bounded coverage = orchard area × fraction of tree locations passing the link target.",
        "Why it is used": "This converts orchard size and density into a population, then counts reachable trees without double-counting overlap between gateways.",
        "Constants and source": "10,000 m² per hectare and π are unit/geometric constants. Tree density, area, spacing and coordinates come from the scenario. Grid columns = round(width/spacing), rows = ceil(population/columns). Seed 42 drives randomized placement. Optional area estimate uses 100×100 midpoint samples (10,000 points), not a surveyed boundary.",
    },
    {
        "Topic": "Reporting capacity",
        "Formula": "Resources/head = min(RF channels, receivers). Minimum slot = (retries+1)×(packet airtime+ACK airtime+ACK wait)+guard+2×clock error. Fixed TDMA slot=max(configured slot, minimum); dynamic/event slot=minimum. Reports/head/round=floor(window/slot)×resources. ALOHA reference load G=reachable nodes×trigger rate×attempts×airtime/(window×resources).",
        "Why it is used": "Each report must finish within its reporting window. Capacity is calculated separately for every assigned head so unused time at another head cannot conceal overload.",
        "Constants and source": "The 900 ms slot, 20 ms guard, ±5 ms clock error, 3600 s window, one channel/receiver, and retry/ACK settings are configurable schedule inputs, not measured radio limits. Two clock-error allowances cover both slot edges. ALOHA capacity is a reference because collisions are probabilistic. All slot times use seconds. Daily slots=rounds/day×slots/round; a missed round cannot borrow another round. No relay queue is modeled.",
    },
    {
        "Topic": "Delivery probability",
        "Formula": "Per-attempt p = link probability × packet_success; ACK mode multiplies another link probability × packet_success. In ALOHA, multiply p by exp(−2G). Independent-retry delivery = 1 − (1 − p)^n, n=retries+1. ALOHA collision success reference = exp(−2G), where G is offered airtime load.",
        "Why it is used": "It estimates the share of admitted reports that arrive, after separating link eligibility and scheduler admission.",
        "Constants and source": "Node packet success p=link probability×packet_success; ACK path success additionally multiplies link_probability×packet_success. Delivery after n attempts=1−(1−p)^n. Packet success (default 0.98) is exploratory unless measured. Pure-ALOHA exp(−2G) omits capture and inter-SF interference. Normal-CDF tail uses t=1/(1+0.2316419|z|) and coefficients 0.319381530, −0.356563782, 1.781477937, −1.821255978, 1.330274429, where z=margin/shadowing_sigma; absolute error <8×10⁻⁸. They are numerical coefficients, not hardware constants; a primary citation for this particular polynomial is not recorded in the project.",
    },
    {
        "Topic": "CNN workload and memory",
        "Formula": "Conv parameters=(kernel²×input channels+1 bias)×filters; MACs=output elements×kernel²×input channels. Dense parameters=(input elements+1 bias)×units. Dense MACs=input elements×units. Weight bytes=parameters×bytes/weight. Serialized KiB=weight bytes/1024×1.05+4. RAM used=DRAM−available heap+selected arena. Flash used=serialized model+firmware flash. Conv output size is floor((input−kernel)/stride)+1 for valid padding, ceil(input/stride) for same padding.",
        "Why it is used": "It counts the selected network workload and screens model storage and working-memory needs before testing on the ESP32.",
        "Constants and source": "INT8 uses 1 byte/weight and FP32 4 bytes/weight. Serialized-size estimate adds 5% plus 4 KiB metadata; peak live activation=max(adjacent input+output), arena lower bound=peak activation+scratch+8 KiB planning allowance. Available heap=DRAM−firmware RAM−stack−audio buffer−DMA buffer; arena must meet lower bound and fit both heap and contiguous memory. Full 16-bit audio buffer=seconds×sample rate×2/1024 KiB; default assumes streaming. The 5%, 4 KiB and 8 KiB are heuristics, not TFLM measurements. The 320 KiB DRAM budget comes from Espressif documentation; other RAM values are scenario inputs.",
    },
    {
        "Topic": "Inference timing and energy",
        "Formula": "Estimated latency = MACs × cycles_per_MAC / (CPU_MHz × 1000 × utilization). Stage energy (mJ) = rail voltage × current (mA) × duration (ms) / 1000.",
        "Why it is used": "It creates a transparent workload estimate and converts entered or imported current profiles into energy. Manual latency/current remain independent inputs, not predictions from MAC count.",
        "Constants and source": "CPU frequency, utilization, cycles/MAC, stage time/current and voltage are editable. Default 240 MHz, 8 cycles/MAC and 70 mA/800 ms are exploratory assumptions; they are not an ESP32 benchmark. Imported current profiles use trapezoidal numerical integration, energy=V×sum((i[n]+i[n+1])/2×Δt)/1000.",
    },
    {
        "Topic": "Radio current",
        "Formula": "Estimated TX radio current is linearly interpolated between output-power/current anchors. TX energy (mJ) = rail V × (radio TX current + MCU TX current) × airtime (ms) / 1000.",
        "Why it is used": "It estimates the radio and MCU energy used while sending a packet. Carrier frequency does not automatically scale TX current.",
        "Constants and source": "HopeRF RFM95 family lower-band table, p.17, gives 11.5 mA RX and TX anchors 20/28/90 mA at 7/13/17 dBm. The 2 dBm/20 mA point is a model clamp; 20 dBm/120 mA follows the high-power specification. These are family references, not measurements of the purchased SX1278 board. MCU TX current is an editable assumption.",
    },
    {
        "Topic": "Recording and SNR gate",
        "Formula": "Expected initial reports per node per day = recording opportunities per day × trigger probability.",
        "Why it is used": "It converts the chosen average trigger rate into expected traffic. The SNR gate runs on each recording window before MFCC, CNN, or radio work.",
        "Constants and source": "The 10 s recording, 8 kHz sample rate, 100–2500 Hz filter, 4 dB gate, and 12 daily times come from the user-provided project setup. The model does not infer trigger_probability from the 4 dB setting; 25% is a separate editable assumption until acoustic measurements establish it.",
    },
    {
        "Topic": "Daily energy and battery life",
        "Formula": "Stage energy mJ=rail V×current mA×duration ms/1000. Daily rail J=rounds×[(1−p) rejected-window J+p accepted-window J]+remaining-sleep J. Battery J/day=rail J/regulator efficiency+battery V×quiescent mA×86.4. Capacity J=battery V×mAh×3.6×usable×temperature×aging. Here p is trigger probability. Active seconds/day=rounds×[(1−p)×rejected duration+p×accepted duration]. Sleep seconds=max(0,86400−active seconds). Sleep J=rail V×(MCU sleep mA+radio sleep mA)×sleep seconds/1000. Average rail mW=daily rail J/86.4. Net load L is battery J/day minus solar Wh/day×3600; k=−ln(1−monthly loss)/30; lifetime days=ln(1+k×capacity/L)/k, or capacity/L when k=0. For L≤0 the model returns unbounded life, not a physical battery guarantee.",
        "Why it is used": "The event gate controls how often MFCC/CNN/radio work runs, while every deployed node still wakes, senses and sleeps. The battery calculation turns that daily load into an idealized lifetime estimate.",
        "Constants and source": "3.7 V and 2600 mAh per node are the user-reported 505573 battery baseline. The 12/day schedule and recording settings were supplied for the project. The 40 mA recording-chain value and 3.3 V rail are exploratory assumptions because the final circuit and regulation are unresolved. Usable fraction 80%, aging factor 90%, temperature factor 100%, regulator efficiency 90%, quiescent current 0.02 mA, monthly self-discharge 2% and solar 0 Wh/day are editable placeholders, not measurements of the purchased wiring. 3.6 converts mAh·V to joules; 86.4 converts mA·V·day to joules; 30 converts a monthly loss fraction to an approximate daily rate. Constant load and exponential self-discharge are simplifying assumptions.",
    },
    {
        "Topic": "Deployment totals and optimization",
        "Formula": "Deployment energy = node + gateway + server energy. Energy per monitored tree = deployment kWh/day × 3,600,000 J/kWh ÷ expected unique monitored trees. A eligibility checks server model storage, delivery, deadline, battery. B instead checks local CNN memory/operator support, delivery, deadline, battery. Full deployment also requires no unreachable or capacity-rejected nodes.",
        "Why it is used": "It includes the infrastructure serving traffic and separates model eligibility from full plantation coverage. Sweeps run the same calculations on alternative inputs; the Pareto view shows designs where improving one goal requires a tradeoff in another.",
        "Constants and source": "Gateway J/day=gateways×(idle W+rail V×RX mA/1000×receivers)×86400+requests×(backhaul J+processing W×packet seconds). Server J/day=idle W×86400 plus, for A, requests×inference W×inference seconds. Deployment cost=deployed nodes×node cost+gateway count×gateway cost. Gateway/server idle and per-request power, model capacity, costs, and target values are scenario assumptions. Unit conversions use 3,600,000 J/kWh and 1000 ms/s. Unique-tree counting uses 1−product(1−sensor delivery probability), assuming independent sensor outcomes; shared fading can make this optimistic. Sensitivity tornado varies an input ±20% by default (configurable in the function) and reports output change from baseline. Battery average current=Battery J/day/(battery V×86.4). Single-attempt break-even ms=max(0,TX_A−TX_B)×1000/(rail V×CNN mA); break-even mA=max(0,TX_A−TX_B)×1000/(rail V×CNN ms). Local inference saves core energy when E_CNN<TX_A−TX_B; full cycle saving=cycle_A−cycle_B. TX energy includes MCU TX current. Annual electricity cost=deployment kWh/day×365×price/kWh. Node deployment J=deployed×base processing/sleep J+admitted×additional communication J, then regulator/quiescent conversion. Requests/day=expected successful reports/round×rounds/day. Pareto dominance means no worse on (−eligible trees, total energy, gateway count) and strictly better on at least one. Full factorial evaluates the Cartesian product; random sampling selects listed values; Latin hypercube maps each coordinate to a listed-value bin. Sampling uses the configured seed; discrete choices are sampled, not interpolated.",
    },
    {
        "Topic": "Shadowing and reliability",
        "Formula": "Log-distance loss = FSPL(reference distance) + 10 × path-loss exponent × log10(distance/reference distance) + vegetation loss. For d below the reference distance, use the reference distance. Link probability=(1−q)Φ(margin/σ)+qΦ((margin−obstruction loss)/σ), where Φ is the standard normal CDF, σ is shadowing in dB and q is obstruction probability (zero except Monte Carlo mode). At σ=0 use a step at zero margin. Free-space eligibility is also a step at zero margin. Radius is solved against the selected probability target, then horizontal radius=sqrt(max(0, slant radius²−height difference²)).",
        "Why it is used": "A link can sometimes work and sometimes fail at the same distance. The probability model represents that variation and supports a stated reliability threshold.",
        "Constants and source": "The lognormal model and Gaussian probability are chosen engineering approximations. Default exponent 3.5, shadowing 4 dB, 10% obstruction chance, 5 dB obstruction loss and 90% reliability threshold are exploratory settings; calibrate them with plantation measurements. The polynomial coefficients approximate the normal CDF numerically (absolute error below 8×10⁻⁸); they are not radio constants.",
    },
    {
        "Topic": "Schedules, retries, and monitoring",
        "Formula": "Fixed TDMA candidates include every reachable node. Dynamic/event capacity fraction=min(1, capacity/(reachable nodes×trigger probability)); seeded active requests determine the realized schedule. Fixed TDMA reserves all admitted slots; dynamic admission scales expected demand by trigger probability. With ACK, expected attempts=sum(k=0…n−1)(1−p)^k; without ACK every retry copy is transmitted. Retry delivery = 1−(1−p)^n. Expected served nodes multiply admitted population by delivery probability. Expected unique monitored trees = sum over trees of 1−product(1−p_i) across admitted sensors.",
        "Why it is used": "These calculations keep schedule admission, packet delivery, and physical trees as separate quantities. They avoid counting multiple sensors on one palm as multiple monitored palms.",
        "Constants and source": "Retry delivery=1−(1−p)^n. ACK expected attempts=sum(k=0…n−1)(1−p)^k; without ACK attempts=n. Transmission outcomes are assumed independent. Trigger probability is an assumed average, not measured from the SNR threshold. Dynamic expected-load admission is an approximation and does not guarantee simultaneous-burst service. Unique-tree calculation assumes independent sensors, which may overstate coverage under shared fading.",
    },
]

# Classify every configurable value so the table never implies that a default was measured.
_USER_INPUT = {
    "frequency_mhz", "battery_v", "battery_mah", "recording_s", "sample_rate_hz",
    "filter_low_hz", "filter_high_hz", "snr_gate_db", "recordings_per_day", "recording_hours",
}
_USER_DESIGN = {
    "sf", "bandwidth_khz", "coding_rate", "preamble_symbols", "slot_ms",
    "guard_ms", "clock_error_ms", "window_s",
}
_MANUFACTURER_PROXY = {
    "radio_standby_ma", "radio_sleep_ma", "radio_rx_ma",
}


def input_provenance(config):
    """Return current/default values and conservative provenance for every model input."""
    rows = []
    for key, default in DEFAULTS.items():
        current = config.get(key, default)
        if key in _USER_INPUT:
            source = "User-provided project baseline; not independently measured here"
        elif key in _USER_DESIGN:
            source = "User/project design setting; configurable assumption"
        elif key in _MANUFACTURER_PROXY:
            source = "Manufacturer-family reference or override; purchased board value unverified"
        elif key == "dram_kb":
            source = "Espressif ESP32 memory-types documentation: 320 KiB DRAM maximum; actual free heap is lower"
        elif key == "active_current_ma":
            source = "Present in scenario settings; currently not used by the calculation pipeline"
        else:
            source = "Exploratory default; replace with a source or measurement when known"
        status = "Same as app default"
        if current != default:
            status = "Scenario override; unverified unless a source or measurement is documented"
        if key in {"inference_latency_ms", "inference_current_ma"} and config.get("performance_source") == "Measured":
            status = "Scenario is labeled measured; verify imported data and source note"
        rows.append(
            {
                "Input": key,
                "Current value": json_value(current),
                "App default": json_value(default),
                "Current status": status,
                "Default evidence / source": source,
                "Source reference": ("User clarification, 28 October 2026; docs/HARDWARE_BASELINE.md" if key in _USER_INPUT else "https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-guides/memory-types.html" if key == "dram_kb" else "https://www.hoperf.com/uploads/RFM95W-V2.0_1695351437.pdf" if key in _MANUFACTURER_PROXY else "Scenario/design assumption; no measurement source recorded"),
            }
        )
    return rows


def json_value(value):
    """Display mixed scenario values in one readable table column."""
    if isinstance(value, (list, dict)):
        import json

        return json.dumps(value, ensure_ascii=False)
    return str(value)
