"""Timing probe for prepare.py: is build_nights fast enough, or does the
nested-loop + per-iteration pandas-masking pattern need optimizing?

Runs decode + build_nights on whatever is already in data_raw/ and times each
stage separately, so we know the cost before committing to the full run.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import prepare

paths = sorted(prepare.RAW_DIR.glob("isd_*.csv"))
print(f"station-years available: {len(paths)}", flush=True)

t0 = time.monotonic()
obs = pd.concat([prepare.decode_station(p) for p in paths], ignore_index=True)
t_decode = time.monotonic() - t0
print(f"decode: {len(obs)} obs in {t_decode:.1f}s", flush=True)

t1 = time.monotonic()
nights = prepare.build_nights(obs)
t_build = time.monotonic() - t1
print(f"build_nights: {len(nights)} nights in {t_build:.1f}s", flush=True)

print(f"\nTOTAL prepare time: {t_decode + t_build:.1f}s", flush=True)
print(f"per station-year: {(t_decode + t_build) / max(len(paths),1):.2f}s", flush=True)
