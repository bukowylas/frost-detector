"""One-fold timing + smoke test for train.py before the full run.

Runs a single leave-one-year-out fold (hold out the latest year), times it, and
prints the fold metrics -- so we confirm the code works and can extrapolate the
full 13-fold run's cost before committing to it (execution-budget discipline).
"""

import _common  # noqa: F401  -- puts the repository root on sys.path

import train
from frostlib.progress import Timer

df = train.load_nights()
held = sorted(df["year"].unique())[-1]
train_df = df[df["year"] != held]
test_df = df[df["year"] == held]
print(f"one LOYO fold: hold out {held}  "
      f"(train {len(train_df)}, test {len(test_df)})", flush=True)

timer = Timer()
res = train.evaluate_fold(train_df, test_df, str(held))
dt = timer.elapsed()

print(f"\nfold done in {dt:.1f}s", flush=True)
for k, v in res.items():
    print(f"  {k}: {v}", flush=True)

# Extrapolate: LOYO has ~5 folds, LOSO ~8 -> ~13 folds total.
print(f"\nextrapolated full run (~13 folds): ~{13 * dt:.0f}s", flush=True)
