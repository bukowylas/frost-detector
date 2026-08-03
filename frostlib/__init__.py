"""Shared utilities for the Frost Detector pipeline.

The pipeline scripts (``fetch_data.py``, ``prepare.py``, ``train.py``,
``predict.py``) and the exploratory scripts under ``tests/`` all touch the same
three things: NOAA ISD packed fields, the NCEI access URLs, and the frost
physics that turns an evening observation into a feature. Those definitions live
here once, so a decoding rule or a feature formula cannot drift between the
script that builds the training data and the script that serves a forecast.

Submodules:
    paths        -- repository data locations and the raw-CSV naming convention
    isd          -- ISD packed-field decoding, quality flags, LST offsets
    isd_history  -- NOAA's station-history file (pandas)
    net          -- NCEI access URLs and HTTP helpers (requests only)
    physics      -- frost feature formulas shared by prepare and predict
    progress     -- elapsed-time progress logging
"""
