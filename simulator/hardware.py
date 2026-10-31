"""User-reported inventory and explicit unknowns; quantities are not deployment limits."""

HARDWARE_BASELINE = {
    "reported_on": "2026-10-28",
    "battery": {"model": "505573", "voltage_v": 3.7, "capacity_mah": 2600, "quantity": 3},
    "charger": {"model": "HiLetgo TP4056 USB-C", "quantity": None, "protection": "Advertised; unverified"},
    "converter": {"model": "XL6009E1 step-up", "quantity": None},
    "sensor": {"model": "WANGCL analog ceramic piezo module", "voltage": "3.3 V/5 V", "kit_quantity": 5},
    "amplifier": {"model": "MCP602E dual CMOS op amp, SOP-8", "quantity": 4, "report_model": "MCP6002"},
    "mcu": {"model": "ESP-WROOM-32 / ESP-32S", "quantity": 3, "cpu_mhz_max": 240, "pcb_revision": None},
    "radio": {"model": "SX1278", "frequency_mhz": 433, "quantity": 4},
    "gateway": {"model": "Raspberry Pi 4 Model B, 4 GB, SPI-connected LoRa receiver", "purchase_confirmed": False},
    "unknowns": [
        "Final power wiring, regulator efficiency, and actual sleep current",
        "Exact PCB revision and whether ESP32 boards include development-board circuitry",
        "Piezo wiring, amplifier gain, resistor values, and final schematic",
        "Transmit power, antenna model/gain, and measured RF performance",
        "Whether the planned Raspberry Pi gateway was purchased",
    ],
}
