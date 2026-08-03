"""Verify what quality flag '9' means in OUR frost data (CS review point #1).

The concern: '9' may mean 'QC not applied' rather than 'confirmed good'. If we
keep flag-9 values, are we admitting unchecked (possibly bad) observations, or
do flag-9 rows coincide with the missing sentinel (in which case keeping '9' in
the accept-set is harmless because the sentinel check already drops them)?

Checks TMP and DEW across all frost station-years: for flag-9 rows, how many
carry the missing sentinel vs a real value, and are those real values
physically plausible?
"""

import csv as csvmod
from collections import Counter

import _common  # noqa: F401  -- puts the repository root on sys.path

from frostlib import isd, paths

for field in ("TMP", "DEW"):
    sentinel = isd.MISSING_TEMP
    flag_counts = Counter()
    flag9_sentinel = 0
    flag9_realvalue = 0
    flag9_real_examples = []
    total = 0
    for path in paths.raw_station_years():
        for row in csvmod.DictReader(path.open()):
            parts = isd.packed(row.get(field, ""))
            if len(parts) < 2:
                continue
            total += 1
            flag = parts[1]
            flag_counts[flag] += 1
            if flag == "9":
                if parts[0] == sentinel:
                    flag9_sentinel += 1
                else:
                    flag9_realvalue += 1
                    if len(flag9_real_examples) < 8:
                        flag9_real_examples.append(int(parts[0]) / 10.0)
    print(f"\n=== {field} (n={total}) ===")
    print("flag distribution:", dict(flag_counts.most_common()))
    print(f"flag '9' with missing sentinel : {flag9_sentinel}")
    print(f"flag '9' with a REAL value     : {flag9_realvalue}")
    if flag9_real_examples:
        print(f"  example flag-9 real values (C): {flag9_real_examples}")
        print("  -> if these look physical, keeping '9' admits UNCHECKED data")
        print("  -> if flag9_realvalue==0, keeping '9' is harmless (sentinel drops them)")
