"""Streamlit entry point for the palm-weevil LoRa digital twin."""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("engineering_app.py")), run_name="__main__")
