"""One-fold timing + smoke test for train.py before the full run.

Runs a single leave-one-year-out fold (hold out the latest year), times it, and
prints the fold metrics -- so we confirm the code works and can extrapolate the
full 13-fold run's cost before committing to it (execution-budget discipline).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import train

df = pd.read_csv(train.DATA)
df["year"] = pd.to_datetime(df["date"]).dt.year
held = sorted(df["year"].unique())[-1]
train_df = df[df["year"] != held]
test_df = df[df["year"] == held]
print(f"one LOYO fold: hold out {held}  "
      f"(train {len(train_df)}, test {len(test_df)})", flush=True)

t0 = time.monotonic()
res = train.evaluate_fold(train_df, test_df, str(held))
dt = time.monotonic() - t0

print(f"\nfold done in {dt:.1f}s", flush=True)
for k, v in res.items():
    print(f"  {k}: {v}", flush=True)

# Extrapolate: LOYO has ~5 folds, LOSO ~8 -> ~13 folds total.
print(f"\nextrapolated full run (~13 folds): ~{13 * dt:.0f}s", flush=True)
