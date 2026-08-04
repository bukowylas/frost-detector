"""Frost Detector live service: nightly job, database, API, and SMS delivery.

The service turns the frozen model artifact into a running system: each evening
in the frost-risk window it fetches live weather for the serviceable stations,
runs the model, stores the forecast richly, and can notify subscribers. Every
component reuses the one validated feature path (``prepare.build_feature_row``
via ``frostlib.live``) so the deployed system stays the system that was
evaluated.
"""
