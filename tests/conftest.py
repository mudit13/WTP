"""Pytest bootstrap: make the scripts/ package importable as `lib.*`, matching how the
entry scripts import (`from lib import schema, ...`). These smoke tests cover the pure-Python
logic only (schema, image ops, metrics, io/config) and deliberately do NOT import torch/clip,
so they run on a plain CPU runner with just requirements.txt + pytest installed."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
