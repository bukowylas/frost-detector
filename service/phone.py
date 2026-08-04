"""Phone-number normalisation to E.164, so one phone is one identity.

``+441234567890``, ``00441234567890`` and ``01234567890`` are the same subscriber
and must not become three rows (three texts a night, an unsubscribe that 404s).
The phone IS the identity here, so it is normalised at the API boundary and only
the normalised form is stored.

This is a deliberately small normaliser for the service's fixed UK/PL stations,
not a full libphonenumber. It handles the ``+``/``00`` international prefixes and a
national ``0`` for a configured default region. If the service ever spans many
countries, swap this for the ``phonenumbers`` library behind the same function.
"""

from __future__ import annotations

# Default region dialling codes for turning a national "0..." number into E.164.
_DEFAULT_CC = {"GB": "44", "PL": "48"}


class InvalidPhone(ValueError):
    pass


def normalize_e164(raw: str, default_region: str = "GB") -> str:
    """Return ``raw`` as an E.164 string (``+<cc><national>``), or raise InvalidPhone.

    Rules, in order:
      ``+CC...``    -> kept (digits only)
      ``00CC...``   -> ``+CC...``
      ``0N...``     -> ``+<default cc><N...>`` (drop the national trunk 0)
    """
    if raw is None:
        raise InvalidPhone("empty phone")
    s = "".join(ch for ch in raw.strip() if ch.isdigit() or ch == "+")

    if s.startswith("+"):
        digits = s[1:]
    elif s.startswith("00"):
        digits = s[2:]
    elif s.startswith("0"):
        cc = _DEFAULT_CC.get(default_region)
        if cc is None:
            raise InvalidPhone(f"unknown default region {default_region!r}")
        digits = cc + s[1:]
    else:
        raise InvalidPhone(f"cannot parse {raw!r}: no +, 00, or leading 0")

    if not digits.isdigit() or not (8 <= len(digits) <= 15):
        raise InvalidPhone(f"implausible number after normalising {raw!r}")
    return "+" + digits
