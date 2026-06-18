"""
Airline policy predicate checkers — v2 (predicate-heavy).

This variant absorbs *all* guard logic into predicates so the YAML file
contains only structural templates and predicate calls — no inline
comparisons.  Every predicate is a small, single-responsibility function.
"""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_if_str(value: Any) -> Any:
    """Parse a JSON string into a Python object; return as-is if already parsed."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def _get_db():
    """Lazily load the airline FlightDB."""
    try:
        from tau2.domains.airline.data_model import get_db
        return get_db()
    except Exception:
        import os
        from pathlib import Path
        db_path = Path(os.environ.get(
            "TAU2_AIRLINE_DB_PATH",
            Path(__file__).resolve().parents[3] / "db.json",
        ))
        if db_path.exists():
            from tau2.domains.airline.data_model import FlightDB
            return FlightDB.load(str(db_path))
        return None


def _extract_payment_id(item: Any) -> str:
    """Extract payment_id string from a dict, object, or raw string."""
    if isinstance(item, dict):
        return str(item.get("payment_id", ""))
    if isinstance(item, str):
        return item
    return str(getattr(item, "payment_id", ""))


# ===========================================================================
# Simple value-only predicates (no DB required)
# ===========================================================================

def same_airport(a: Any, b: Any) -> bool:
    """Return True when two airport codes are identical.

    Replaces the inline ``origin == destination`` comparison.
    """
    return str(a).strip().upper() == str(b).strip().upper()


def is_basic_economy(cabin: Any) -> bool:
    """Return True when the cabin class is basic_economy.

    Replaces ``cabin == "basic_economy"``.
    """
    return str(cabin).strip().lower() == "basic_economy"


def invalid_baggage_counts(total: Any, nonfree: Any) -> bool:
    """Return True when baggage counts violate invariants.

    Consolidates three original inline checks:
      - total_baggages < 0
      - nonfree_baggages < 0
      - nonfree_baggages > total_baggages
    """
    try:
        t, n = int(total), int(nonfree)
    except (ValueError, TypeError):
        return True  # Non-numeric → invalid
    return t < 0 or n < 0 or n > t


def invalid_certificate_amount(amount: Any) -> bool:
    """Return True when the certificate amount is out of valid range.

    Consolidates:
      - amount <= 0
      - amount > 500  (max 5 passengers × $100)
    """
    try:
        a = int(amount)
    except (ValueError, TypeError):
        return True
    return a <= 0 or a > 500


def is_certificate_payment(payment_id: Any) -> bool:
    """Return True when the payment ID refers to a travel certificate.

    Policy: certificates cannot be used to pay for reservation updates.
    """
    return str(payment_id).strip().lower().startswith("certificate")


def too_many_passengers(passengers: Any) -> bool:
    """Return True if the passenger list exceeds 5 entries.

    Policy: "Each reservation can have at most five passengers."
    """
    parsed = _parse_json_if_str(passengers)
    if isinstance(parsed, list):
        return len(parsed) > 5
    return False


def exceeds_payment_limits(payment_methods: Any) -> bool:
    """Return True if payment methods exceed the allowed limits.

    Policy: at most 1 certificate, 1 credit card, 3 gift cards.
    """
    methods = _parse_json_if_str(payment_methods)
    if not isinstance(methods, list):
        return False

    cc, gc, cert = 0, 0, 0
    for m in methods:
        pid = _extract_payment_id(m).lower()
        if pid.startswith("credit_card"):
            cc += 1
        elif pid.startswith("gift_card"):
            gc += 1
        elif pid.startswith("certificate"):
            cert += 1
    return cc > 1 or gc > 3 or cert > 1


# ===========================================================================
# DB-aware predicates — single lookup each
# ===========================================================================

def invalid_payment_methods(payment_methods: Any, user_id: Any) -> bool:
    """Return True if any payment method is not in the user's profile."""
    methods = _parse_json_if_str(payment_methods)
    if not isinstance(methods, list):
        return False

    db = _get_db()
    if db is None:
        return False

    uid = str(user_id)
    if uid not in db.users:
        return True

    known_ids = set(db.users[uid].payment_methods.keys())
    for m in methods:
        pid = _extract_payment_id(m)
        if pid and pid not in known_ids:
            return True
    return False


def passenger_count_changed(reservation_id: Any, passengers: Any) -> bool:
    """Return True if the new passenger count differs from the reservation."""
    parsed = _parse_json_if_str(passengers)
    if not isinstance(parsed, list):
        return False

    db = _get_db()
    if db is None:
        return False

    rid = str(reservation_id)
    if rid not in db.reservations:
        return False
    return len(parsed) != len(db.reservations[rid].passengers)


def baggage_decreased(reservation_id: Any, total_baggages: Any) -> bool:
    """Return True if the new total baggage count is less than the current."""
    db = _get_db()
    if db is None:
        return False

    rid = str(reservation_id)
    if rid not in db.reservations:
        return False

    try:
        new_total = int(total_baggages)
    except (ValueError, TypeError):
        return False
    return new_total < db.reservations[rid].total_baggages


def has_flown_flights(reservation_id: Any) -> bool:
    """Return True if any flight in the reservation has status 'flying' or 'landed'."""
    db = _get_db()
    if db is None:
        return False

    rid = str(reservation_id)
    if rid not in db.reservations:
        return False

    for fi in db.reservations[rid].flights:
        if fi.flight_number in db.flights and fi.date in db.flights[fi.flight_number].dates:
            if db.flights[fi.flight_number].dates[fi.date].status in ("flying", "landed"):
                return True
    return False


def route_changed(reservation_id: Any, flights: Any) -> bool:
    """Return True if the flight update changes origin, destination, or trip type."""
    db = _get_db()
    if db is None:
        return False

    rid = str(reservation_id)
    if rid not in db.reservations:
        return False

    res = db.reservations[rid]
    flights_parsed = _parse_json_if_str(flights)
    if not isinstance(flights_parsed, list) or not flights_parsed:
        return False

    new_origins, new_dests = [], []
    for f in flights_parsed:
        fn = f.get("flight_number", "") if isinstance(f, dict) else str(getattr(f, "flight_number", ""))
        if fn and fn in db.flights:
            new_origins.append(db.flights[fn].origin)
            new_dests.append(db.flights[fn].destination)

    if not new_origins or not new_dests:
        return False

    if new_origins[0] != res.origin or new_dests[-1] != res.destination:
        return True

    if res.flight_type == "round_trip" and len(flights_parsed) < 2:
        return True

    return False


def compensation_eligible(user_id: Any) -> bool:
    """Return True if the user qualifies for compensation.

    Eligible if silver/gold, or any reservation has insurance or business cabin.
    """
    db = _get_db()
    if db is None:
        return True

    uid = str(user_id)
    if uid not in db.users:
        return False

    user = db.users[uid]
    if user.membership in ("silver", "gold"):
        return True

    for rid in user.reservations:
        if rid in db.reservations:
            r = db.reservations[rid]
            if r.insurance == "yes" or r.cabin == "business":
                return True
    return False


def valid_certificate_amount(user_id: Any, amount: Any) -> bool:
    """Return True if the certificate amount matches $100/pax (cancelled) or $50/pax (delayed)."""
    db = _get_db()
    if db is None:
        return True

    try:
        amt = int(amount)
    except (ValueError, TypeError):
        return False

    if amt <= 0:
        return False

    uid = str(user_id)
    if uid not in db.users:
        return False

    valid = set()
    for rid in db.users[uid].reservations:
        if rid not in db.reservations:
            continue
        res = db.reservations[rid]
        npax = len(res.passengers)
        cancelled, delayed = False, False

        for fi in res.flights:
            if fi.flight_number in db.flights and fi.date in db.flights[fi.flight_number].dates:
                st = db.flights[fi.flight_number].dates[fi.date].status
                if st == "cancelled":
                    cancelled = True
                elif st == "delayed":
                    delayed = True
        if res.status == "cancelled":
            cancelled = True

        if cancelled:
            valid.add(100 * npax)
        if delayed:
            valid.add(50 * npax)

    return amt in valid if valid else True
