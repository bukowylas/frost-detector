"""SMS delivery: a small interface and a default log-stub implementation.

The whole service runs end-to-end with no paid account and no real texts: the
default ``LogSmsSender`` writes each message to the log. A real provider (e.g.
Twilio) is added by implementing ``SmsSender.send`` -- callers depend on the
interface, not the provider, so nothing else changes.

Message language is deliberate. At the default threshold the model's precision is
~0.64 (roughly one alarm in three is a false alarm), so a message is a
**forecast**, not a warning, and it reports the predicted temperature (the
magnitude), because different crop stages are damaged at different temperatures.
Every message carries a STOP line -- one-tap unsubscribe, legally required and
correct regardless.
"""

from __future__ import annotations

import logging
from typing import Protocol

log = logging.getLogger("frost.sms")

STOP_LINE = "Reply STOP to unsubscribe"


class SmsSender(Protocol):
    """Anything that can deliver a text to a phone number."""

    def send(self, to: str, message: str) -> None:
        ...


class LogSmsSender:
    """Default sender: logs the message instead of sending it (no cost, no texts).

    Records every message on ``self.sent`` as (to, message) so tests and a local
    run can assert what *would* have been delivered.
    """

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send(self, to: str, message: str) -> None:
        self.sent.append((to, message))
        log.info("SMS to %s:\n%s", to, message)


class SmsBudgetExceeded(RuntimeError):
    """The global daily send budget is exhausted; the send is refused."""


class BudgetedSender:
    """Wraps any sender with a hard GLOBAL daily send cap.

    The per-phone cooldown bounds the rate to one number, but ``/api/subscribe``
    can still issue one SMS per *distinct* number, so an attacker cycling numbers
    is unbounded across numbers. This cap is the backstop: a fixed number of sends
    per UTC day, refused past that. It lives at the sender so it protects EVERY
    path, and so that swapping the log stub for a paid provider is not the step
    that arms an unbounded-spend problem -- the budget is already there.
    """

    def __init__(self, inner, daily_limit: int = 500) -> None:
        self._inner = inner
        self._limit = daily_limit
        self._day = None
        self._count = 0

    def send(self, to: str, message: str) -> None:
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).date()
        if today != self._day:
            self._day, self._count = today, 0
        if self._count >= self._limit:
            raise SmsBudgetExceeded(
                f"daily SMS budget of {self._limit} reached; refusing further sends")
        self._inner.send(to, message)
        self._count += 1


def format_forecast_sms(station_label: str, date_label: str,
                        predicted_tmin_c: float, frost_likely: bool,
                        typical_error_c: float | None = None) -> str:
    """The grower-facing forecast message.

    Reports the predicted minimum (the magnitude), names it a forecast, and adds
    'frost likely' only when the forecast crosses the subscriber's threshold --
    honest language that reads as weather, not a broken product. When known, the
    typical error is shown, because growers act on margins: a predicted +1.6 with
    a ±1.8 error is roughly a coin flip on frost, and the number should say so.
    """
    verdict = " -- frost likely" if frost_likely else ""
    band = f" (typical error +/-{typical_error_c:.1f} C)" if typical_error_c else ""
    return (f"{station_label}, night of {date_label}\n"
            f"Predicted min: {predicted_tmin_c:+.1f} C{band}{verdict}\n"
            f"{STOP_LINE}")


def format_no_forecast_sms(station_label: str, date_label: str) -> str:
    """The heartbeat message when a station produced no forecast.

    A nightly subscriber must never receive silence -- silence is ambiguous
    between "clear tonight" and "the system is down". An explicit "no forecast"
    turns the gap into information.
    """
    return (f"{station_label}, night of {date_label}\n"
            f"No forecast tonight -- weather data unavailable. "
            f"Check local conditions.\n"
            f"{STOP_LINE}")


def format_settings_summary(station_label: str, mode: str, threshold_c: float) -> str:
    """A one-line description of what a subscription will do, for the code SMS."""
    if mode == "frost":
        return f"{station_label}, frost alerts below {threshold_c:+.1f} C"
    return f"{station_label}, a forecast every night"


def format_verification_sms(code: str, settings_summary: str) -> str:
    """The verification SMS. It NAMES the settings the code will confirm, so the
    code round-trip authenticates *what the subscription does*, not merely the
    number -- a request that tried to poison the settings is visible before the
    grower types the code."""
    return (f"Frost Detector code {code} -- confirms: {settings_summary}.\n"
            f"Enter it to activate. Ignore this if you didn't request it.")


def format_unsubscribe_sms(station_label: str) -> str:
    """Sent when a subscription is deactivated. This confirmation IS the
    authentication: an attacker who deactivates a grower cannot stop the grower
    being told, and the grower resumes with one word."""
    return (f"You've been unsubscribed from {station_label} frost forecasts.\n"
            f"Reply START to resume.")
