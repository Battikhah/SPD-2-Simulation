"""LoRa airtime and link budget; RF assumptions are not field measurements."""

import math

import numpy as np

SOURCE = "https://www.semtech.com/products/wireless-rf/lora-connect/sx1278"
REFERENCE_SOURCE = "https://www.hoperf.com/uploads/RFM95W-V2.0_1695351437.pdf"
# HopeRF HF/LF typical tables; LF values are an unverified proxy for the purchased SX1278 board.
SENSITIVITY = {
    915: {
        125: [-118, -123, -126, -129, -132, -133, -136],
        250: [-115, -120, -123, -125, -128, -130, -133],
        500: [-111, -116, -119, -122, -125, -128, -130],
    },
    433: {
        62.5: [-123, -128, -131, -134, -135, -137, -140],
        125: [-121, -125, -128, -131, -134, -136, -137],
        250: [-118, -122, -125, -128, -131, -133, -134],
        500: [-112, -118, -121, -124, -127, -129, -130],
    },
}


def airtime_ms(
    payload_bytes,
    sf=7,
    bandwidth_khz=125,
    coding_rate=5,
    preamble_symbols=8,
    implicit_header=False,
    crc=True,
    ldro="auto",
):
    """Semtech payload-symbol equation; coding_rate is the denominator 5..8."""
    if int(payload_bytes) != payload_bytes or not 0 <= payload_bytes <= 255:
        raise ValueError("LoRa payload must be an integer between 0 and 255 bytes.")
    if sf not in range(6, 13) or coding_rate not in (5, 6, 7, 8):
        raise ValueError("SF must be 6–12 and coding rate denominator 5–8.")
    if bandwidth_khz not in (62.5, 125, 250, 500) or not 6 <= preamble_symbols <= 65535:
        raise ValueError("Unsupported bandwidth or preamble length (6–65535).")
    if sf == 6 and not implicit_header:
        raise ValueError("SF6 requires implicit header mode.")
    symbol_ms = 2**sf / bandwidth_khz
    de = symbol_ms > 16 if str(ldro).lower() == "auto" else ldro in (True, 1, "On", "on")
    numerator = 8 * payload_bytes - 4 * sf + 28 + 16 * bool(crc) - 20 * bool(implicit_header)
    payload_symbols = 8 + max(math.ceil(numerator / (4 * (sf - 2 * de))) * coding_rate, 0)
    return (preamble_symbols + 4.25 + payload_symbols) * symbol_ms


def sensitivity_dbm(c):
    """Manufacturer typical sensitivity, not a measured plantation PER curve."""
    f, bw, sf = c.get("frequency_mhz", 915), c.get("bandwidth_khz", 125), c.get("sf", 7)
    if f not in SENSITIVITY or bw not in SENSITIVITY[f] or sf not in range(6, 13):
        raise ValueError("Unsupported module frequency, bandwidth or spreading factor.")
    if sf == 6 and not c.get("implicit_header", False):
        raise ValueError("SF6 requires implicit header mode.")
    return SENSITIVITY[f][bw][sf - 6] + c.get("sensitivity_offset_db", 0)


def fspl_db(distance_m, frequency_mhz):
    """Free-space loss with distances in metres and frequency in MHz."""
    return 32.44 + 20 * np.log10(np.maximum(distance_m, 1e-9) / 1000) + 20 * np.log10(frequency_mhz)


def path_loss_db(c, distance_m):
    if str(c.get("propagation", "Log-distance")).lower().startswith("free"):
        return fspl_db(np.maximum(distance_m, 1), c.get("frequency_mhz", 915))
    d0 = c.get("reference_distance_m", 1)
    return (
        fspl_db(d0, c.get("frequency_mhz", 915))
        + 10 * c.get("path_loss_exponent", 3.5) * np.log10(np.maximum(distance_m, d0) / d0)
        + c.get("vegetation_loss_db", 10)
    )


def link_budget_db(c):
    return (
        c.get("tx_dbm", 17)
        + c.get("tx_gain_dbi", 2)
        + c.get("rx_gain_dbi", 2)
        - c.get("cable_loss_db", 1)
        - c.get("misc_loss_db", 3)
        - c.get("interference_loss_db", 0)
        - sensitivity_dbm(c)
        - c.get("required_margin_db", 10)
    )


def range_m(c):
    """Radius meeting the statistical margin target under the assumed channel."""
    budget = link_budget_db(c)
    free = str(c.get("propagation", "Log-distance")).lower().startswith("free")
    height_difference = c.get("rx_height_m", 4) - c.get("tx_height_m", 1.5)
    if free:
        slant = 1000 * 10 ** ((budget - 32.44 - 20 * math.log10(c.get("frequency_mhz", 915))) / 20)
        return math.sqrt(max(0.0, slant**2 - height_difference**2))
    # Obstructions are a mixture; binary search the same probability model used for nodes.
    lo, hi = -3.0, 9.0
    for _ in range(50):
        mid = (lo + hi) / 2
        p = float(link_probability(c, budget - path_loss_db(c, 10**mid)))
        if p >= c.get("min_reliability", 0.9):
            lo = mid
        else:
            hi = mid
    return math.sqrt(max(0.0, 10 ** (2 * lo) - height_difference**2))


def link_probability(c, margin_db):
    """P(channel margin >= 0), lognormal shadowing and optional obstruction mixture."""
    m = np.asarray(margin_db)
    if str(c.get("propagation", "")).lower().startswith("free"):
        return (m >= 0).astype(float)
    sigma = c.get("shadowing_db", 4)

    # Gaussian CDF approximation (absolute error < 8e-8); avoids a SciPy dependency.
    def cdf(x):
        if sigma <= 0:
            return (x >= 0).astype(float)
        z = x / sigma
        t = 1 / (1 + 0.2316419 * np.abs(z))
        tail = (
            np.exp(-z * z / 2)
            / math.sqrt(2 * math.pi)
            * t
            * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        )
        return np.where(z >= 0, 1 - tail, tail)

    obstacle_p = c.get("obstruction_probability", 0.1) if "monte" in str(c.get("propagation", "")).lower() else 0
    return (1 - obstacle_p) * cdf(m) + obstacle_p * cdf(m - c.get("obstruction_loss_db", 5))


def tx_current_ma(c):
    """Typical-current anchors, interpolated between them; same PA assumptions in both bands."""
    override = c.get("tx_current_override_ma", 0)
    if override > 0:
        return override
    power = c.get("tx_dbm", 17)
    if not 2 <= power <= 20:
        raise ValueError("Model supports TX output power from 2 to 20 dBm.")
    # 7/13/17 dBm: HopeRF p.17; +20 dBm: high-power supply specification p.13.
    return float(np.interp(power, [2, 7, 13, 17, 20], [20, 20, 28, 90, 120]))


def radio_summary(c):
    kwargs = {
        k: c.get(k, default)
        for k, default in dict(
            sf=7, bandwidth_khz=125, coding_rate=5, preamble_symbols=8, implicit_header=False, crc=True, ldro="auto"
        ).items()
    }
    overhead = c.get("metadata_bytes", 0) + c.get("protocol_bytes", 0)
    radius = range_m(c)
    purchased_profile = c.get("frequency_mhz", 915) == 433
    return dict(
        module="SX1278" if purchased_profile else "RFM95W/SX1276 comparison",
        sensitivity_dbm=sensitivity_dbm(c),
        range_m=radius,
        airtime_a_ms=airtime_ms(c.get("payload_a_bytes", 214) + overhead, **kwargs),
        airtime_b_ms=airtime_ms(c.get("payload_b_bytes", 13) + overhead, **kwargs),
        tx_current_ma=tx_current_ma(c),
        link_budget_db=link_budget_db(c),
        coverage_ha=math.pi * radius**2 / 10000,
        quarter_wave_cm=299792458 / (c.get("frequency_mhz", 915) * 1e6) * 25,
        source=SOURCE if purchased_profile else REFERENCE_SOURCE,
        sensitivity_basis=(
            "Legacy HopeRF low-band family reference proxy; SX1278 board and antenna are uncharacterized"
            if purchased_profile
            else "HopeRF high-band manufacturer typical, highest LNA gain, split RF paths"
        ),
        tx_current_basis=(
            "Legacy HopeRF family-anchor interpolation proxy; purchased SX1278 module current is unverified"
            if purchased_profile
            else "Estimated interpolation of HopeRF manufacturer current anchors"
        ),
        reference_source=REFERENCE_SOURCE,
    )
