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

# Map a station key's country prefix to its ISO region, for parsing national-
# format numbers. Matched on the first '_'-delimited segment, not a slice, and
# with NO silent default -- an unknown prefix fails closed (a number valid in two
# plans would otherwise normalise to the wrong country and text a stranger).
_STATION_REGION = {"uk": "GB", "pl": "PL"}
_DEFAULT_REGION = "GB"

# SMS-capable number types (allowlist, not blocklist): everything else -- premium
# rate (09xx, billable to us), UAN (03xx), shared-cost (084x), pager, voicemail,
# fixed line -- cannot receive an SMS or shouldn't be texted. FIXED_LINE_OR_MOBILE
# is included because libphonenumber returns it where the two are indistinguishable
# (e.g. the US); it doesn't arise for GB/PL, so today this costs nothing and is
# lenient in the right direction the day a US number appears.
_SMS_CAPABLE = {
    phonenumbers.PhoneNumberType.MOBILE,
    phonenumbers.PhoneNumberType.FIXED_LINE_OR_MOBILE,
}


class InvalidPhone(ValueError):
    pass


def region_for_station(station: str) -> str:
    """The dialling region to parse a subscriber's number in, from the station key.

    Fails closed on an unknown prefix rather than defaulting to GB, so a future
    non-UK station cannot silently misroute a number to the wrong country. (Every
    serviceable station is UK today; config.py refuses to start on an unvalidated
    station, so this only fires on a genuine config gap.)"""
    prefix = station.split("_", 1)[0].lower()
    try:
        return _STATION_REGION[prefix]
    except KeyError:
        raise InvalidPhone(f"no dialling region configured for station {station!r}")


def normalize_e164(raw: str, region: str = _DEFAULT_REGION) -> str:
    """Return ``raw`` as an E.164 string, or raise InvalidPhone.

    Rejects numbers that are not valid, and numbers that cannot receive SMS (only
    mobiles are accepted): texting a landline/premium/UAN number fails silently or
    bills us, so it is caught at signup rather than at 2am on a frost night.
    """
    if not raw or not raw.strip():
        raise InvalidPhone("empty phone")
    try:
        parsed = phonenumbers.parse(raw, region)
    except phonenumbers.NumberParseException as exc:
        raise InvalidPhone(f"cannot parse {raw!r}: {exc}") from exc
    if not phonenumbers.is_valid_number(parsed):
        raise InvalidPhone(f"not a valid number: {raw!r}")
    if phonenumbers.number_type(parsed) not in _SMS_CAPABLE:
        raise InvalidPhone("this number cannot receive SMS; use a mobile")
    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
