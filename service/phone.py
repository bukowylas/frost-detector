"""Phone-number normalisation to E.164, so one phone is one identity.

Backed by the ``phonenumbers`` library (Google's libphonenumber port): a
hand-rolled parser gets real numbers subtly wrong, and the failure mode here is a
grower who never receives a warning -- the exact harm the service exists to
prevent. The library handles the cases that matter even for the UK-only live
service today: the ``+44 (0)7911...`` business-card format, and distinguishing a
mobile (can receive SMS) from a landline (SMS silently vanishes).

The parsing region comes from the station (``uk_`` -> GB, ``pl_`` -> PL), not a
global default, so the service is correct the day a non-UK station becomes
serviceable -- but note that today every serviceable station is UK.
"""

from __future__ import annotations

import phonenumbers

# Map a station key prefix to its ISO region, for parsing national-format numbers.
_STATION_REGION = {"uk": "GB", "pl": "PL"}
_DEFAULT_REGION = "GB"


class InvalidPhone(ValueError):
    pass


def region_for_station(station: str) -> str:
    """The dialling region to parse a subscriber's number in, from the station key.

    Every serviceable station is currently UK, so this returns GB in practice; it
    is derived from the station so a future non-UK station is handled correctly
    without an API change."""
    return _STATION_REGION.get(station[:2].lower(), _DEFAULT_REGION)


def normalize_e164(raw: str, region: str = _DEFAULT_REGION) -> str:
    """Return ``raw`` as an E.164 string, or raise InvalidPhone.

    Rejects numbers that are not valid, and numbers that cannot receive SMS
    (fixed-line): texting a landline fails silently, so it is caught at signup
    rather than at 2am on a frost night.
    """
    if not raw or not raw.strip():
        raise InvalidPhone("empty phone")
    try:
        parsed = phonenumbers.parse(raw, region)
    except phonenumbers.NumberParseException as exc:
        raise InvalidPhone(f"cannot parse {raw!r}: {exc}") from exc
    if not phonenumbers.is_valid_number(parsed):
        raise InvalidPhone(f"not a valid number: {raw!r}")
    ntype = phonenumbers.number_type(parsed)
    if ntype == phonenumbers.PhoneNumberType.FIXED_LINE:
        raise InvalidPhone("landline numbers cannot receive SMS; use a mobile")
    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
