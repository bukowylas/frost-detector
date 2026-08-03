"""Timing probe for prepare.py: is build_nights fast enough, or does the
nested-loop + per-iteration pandas-masking pattern need optimizing?

Runs decode + build_nights on whatever is already in data_raw/ and times each
stage separately, so we know the cost before committing to the full run.
"""

import _common  # noqa: F401  -- puts the repository root on sys.path
import pandas as pd

import prepare
from frostlib import paths
from frostlib.progress import Timer

timer = Timer()
listed = paths.raw_station_years()
print(f"station-years available: {len(listed)}", flush=True)

timer.reset()
obs = pd.concat([prepare.decode_station(p) for p in listed], ignore_index=True)
t_decode = timer.elapsed()
print(f"decode: {len(obs)} obs in {t_decode:.1f}s", flush=True)

timer.reset()
nights, _rejected = prepare.build_nights(obs)
t_build = timer.elapsed()
print(f"build_nights: {len(nights)} nights in {t_build:.1f}s", flush=True)

print(f"\nTOTAL prepare time: {t_decode + t_build:.1f}s", flush=True)
print(f"per station-year: {(t_decode + t_build) / max(len(listed),1):.2f}s", flush=True)
