"""Entry point for the nightly run: forecast, store, then notify.

Wires the pieces the scheduler invokes each evening:

    fetch (parity-validated) -> validate -> predict -> store richly -> notify

Run for tonight (the default), or for an explicit date (e.g. a backfill):

    python3 -m service.run_nightly
    python3 -m service.run_nightly --date 2026-04-15

Everything runs against the configured DATABASE_URL (SQLite by default) and the
log-stub SMS sender, so a local run needs no Postgres and sends no real texts.
Seasonal guard, validate-never-impute, idempotency and the feature-name assertion
all live in the components this calls -- see ``service.nightly_job`` /
``service.notify``.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from frostlib import model_io
from service import db, nightly_job, notify


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", help="evening date YYYY-MM-DD (default: today)")
    ap.add_argument("--no-notify", action="store_true",
                    help="forecast and store, but do not send notifications")
    args = ap.parse_args()

    date = pd.Timestamp(args.date).date() if args.date else pd.Timestamp.now().normalize().date()

    engine = db.make_engine()
    db.create_all(engine)   # bootstrap; production applies Alembic migrations first
    session = db.make_session_factory(engine)()

    results = nightly_job.run(session, dates=[date])
    stored = sum(1 for r in results if r.stored)
    logging.info("nightly run for %s: %d stored, %d skipped",
                 date, stored, len(results) - stored)

    if not args.no_notify:
        # The typical-error band shown to growers is the LIVE MAE (cloud-blank),
        # the accuracy they actually receive -- read from the frozen artifact.
        try:
            artifact = model_io.load_model()
            typical_error = artifact.mae_c
        except model_io.ModelContractError:
            typical_error = None
        # Notify iterates (station, subscriber), so a skipped station still
        # heartbeats a nightly subscriber -- run even when nothing stored.
        sent = notify.notify_for_date(session, date, typical_error_c=typical_error)
        resent = notify.retry_pending(session, typical_error_c=typical_error)
        # Drain unsubscribe confirmations that failed to send at opt-out time.
        confirmed = notify.drain_confirmations(session)
        logging.info("notified %d; retried %d pending; drained %d unsub confirmations",
                     len(sent), len(resent), len(confirmed))

    # Health: a single expected skip is a warning (logged, exit 0); a station that
    # never ran, a send failure, or a station skipping two nights running is an
    # alert (exit non-zero). A cron that always exits 0 is a cron nobody watches;
    # a cron that alerts on every expected skip is one whose exit code gets ignored.
    health = notify.notify_health(session, date)
    if health["skips"]:
        logging.warning("skips for %s: %s", date, health["skips"])
    if health["healthy"]:
        logging.info("health OK for %s: %s", date, health["send_status"])
    else:
        logging.error("UNHEALTHY %s: unwarned=%s missing=%s repeated_skips=%s",
                      date, health["unwarned"], health["missing"],
                      health["repeated_skips"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
