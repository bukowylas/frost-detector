"""Frost base-rate check on the current processed data.

Frost forecasting needs enough sub-freezing events to learn from. This reports,
per station: how many hourly temp readings, how many at/below 0C (frost) and
<=2C (frost-warning threshold), and the base rate. Also checks night/dawn hours
specifically, since radiative frost is a near-dawn phenomenon.
"""

from pathlib import Path

import pandas as pd
from _common import FROST_C, WARN_C

DATA = Path("/mnt/c/Users/marci/Desktop/Cloud Cover/data/processed.csv")
df = pd.read_csv(DATA, parse_dates=["timestamp"])

print(f"rows: {len(df)}   stations: {df['station'].nunique()}\n")

df = df.dropna(subset=["temp_c"]).copy()
df["hour"] = df["timestamp"].dt.hour

print(f"{'station':22s} {'n':>7s} {'<=0C':>7s} {'rate':>6s} "
      f"{'<=2C':>7s} {'rate':>6s} {'min_C':>7s}")
for st, g in df.groupby("station"):
    n = len(g)
    frost = (g["temp_c"] <= FROST_C).sum()
    warn = (g["temp_c"] <= WARN_C).sum()
    print(f"{st:22s} {n:7d} {frost:7d} {100*frost/n:5.1f}% "
          f"{warn:7d} {100*warn/n:5.1f}% {g['temp_c'].min():7.1f}")

# Overall + a dawn-window view (03:00-09:00 local-ish; data is UTC so approximate)
print("\noverall <=0C rate:", f"{100*(df['temp_c'] <= FROST_C).mean():.1f}%")
dawn = df[df["hour"].between(9, 13)]  # UTC ~ pre-dawn/dawn for US longitudes
if len(dawn):
    print(f"dawn-ish window (UTC 09-13) <=0C rate: "
          f"{100*(dawn['temp_c'] <= FROST_C).mean():.1f}%  (n={len(dawn)})")
