"""
Airline policy predicate checkers — v3 (minimal, YAML-heavy).

Each predicate is a trivial single-purpose function.  Where possible it
returns a *scalar* (int, str, bool) so the YAML where-clause can do the
comparison itself.  Only functions that genuinely require Python (JSON
parsing, DB iteration) live here — all other logic is inline in
policy_v3.yaml.
"""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_if_str(value: Any) -> Any:
    """Parse a JSON string; return as-is if already a Python object."""
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
    """Extract payment_id from a dict, object, or raw string."""
    if isinstance(item, dict):
        return str(item.get("payment_id", ""))
    if isinstance(item, str):
        return item
    return str(getattr(item, "payment_id", ""))


# ===========================================================================
# Scalar-returning predicates — used in YAML comparisons like:
#   count_items(passengers) > 5
#   count_credit_cards(payment_methods) > 1
# ===========================================================================

def count_items(collection: Any) -> int:
    """Return the number of items in a JSON list (or 0 if not a list)."""
    parsed = _parse_json_if_str(collection)
    return len(parsed) if isinstance(parsed, list) else 0


def count_credit_cards(payment_methods: Any) -> int:
    """Count payment entries whose ID starts with 'credit_card'."""
    methods = _parse_json_if_str(payment_methods)
    if not isinstance(methods, list):
        return 0
    return sum(1 for m in methods if _extract_payment_id(m).lower().startswith("credit_card"))


def count_gift_cards(payment_methods: Any) -> int:
    """Count payment entries whose ID starts with 'gift_card'."""
    methods = _parse_json_if_str(payment_methods)
    if not isinstance(methods, list):
        return 0
    return sum(1 for m in methods if _extract_payment_id(m).lower().startswith("gift_card"))


def count_certificates(payment_methods: Any) -> int:
    """Count payment entries whose ID starts with 'certificate'."""
    methods = _parse_json_if_str(payment_methods)
    if not isinstance(methods, list):
        return 0
    return sum(1 for m in methods if _extract_payment_id(m).lower().startswith("certificate"))


# ===========================================================================
# Scalar-returning DB lookups — return a single value for YAML comparison
# ===========================================================================

def reservation_passenger_count(reservation_id: Any) -> int:
    """Return the current number of passengers on a reservation (DB lookup)."""
    db = _get_db()
    if db is None:
        return 0
    rid = str(reservation_id)
    if rid not in db.reservations:
        return 0
    return len(db.reservations[rid].passengers)


def reservation_baggage_count(reservation_id: Any) -> int:
    """Return the current total_baggages on a reservation (DB lookup)."""
    db = _get_db()
    if db is None:
        return 0
    rid = str(reservation_id)
    if rid not in db.reservations:
        return 0
    return db.reservations[rid].total_baggages


def reservation_cabin(reservation_id: Any) -> str:
    """Return the current cabin class of a reservation (DB lookup).

    Used by no-basic-economy-flight-modification to check the *source*
    reservation cabin rather than the target cabin argument, so that
    upgrades FROM basic_economy are correctly blocked while downgrades
    TO basic_economy from a higher cabin are correctly allowed.
    Returns an empty string if the reservation is not found.
    """
    db = _get_db()
    if db is None:
        return ""
    rid = str(reservation_id)
    if rid not in db.reservations:
        return ""
    return db.reservations[rid].cabin


# ===========================================================================
# Bool predicates — minimal set for checks that cannot be decomposed
# into a single scalar + YAML comparison
# ===========================================================================

def has_unknown_payment(payment_methods: Any, user_id: Any) -> bool:
    """Return True if any payment_id is not in the user's profile."""
    methods = _parse_json_if_str(payment_methods)
    if not isinstance(methods, list):
        return False
    db = _get_db()
    if db is None:
        return False
    uid = str(user_id)
    if uid not in db.users:
        return True
    known = set(db.users[uid].payment_methods.keys())
    return any(_extract_payment_id(m) not in known for m in methods if _extract_payment_id(m))


def has_flown_flights(reservation_id: Any) -> bool:
    """Return True if any flight in the reservation has status 'flying' or 'landed'."""
    db = _get_db()
    if db is None:
        return False
    rid = str(reservation_id)
    if rid not in db.reservations:
        return False
    for fi in db.reservations[rid].flights:
        fn, dt = fi.flight_number, fi.date
        if fn in db.flights and dt in db.flights[fn].dates:
            if db.flights[fn].dates[dt].status in ("flying", "landed"):
                return True
    return False


def route_changed(reservation_id: Any, flights: Any) -> bool:
    """Return True if new flights change origin, destination, or trip type."""
    db = _get_db()
    if db is None:
        return False
    rid = str(reservation_id)
    if rid not in db.reservations:
        return False

    res = db.reservations[rid]
    parsed = _parse_json_if_str(flights)
    if not isinstance(parsed, list) or not parsed:
        return False

    origins, dests = [], []
    for f in parsed:
        fn = f.get("flight_number", "") if isinstance(f, dict) else str(getattr(f, "flight_number", ""))
        if fn and fn in db.flights:
            origins.append(db.flights[fn].origin)
            dests.append(db.flights[fn].destination)

    if not origins or not dests:
        return False
    if origins[0] != res.origin or dests[-1] != res.destination:
        return True
    if res.flight_type == "round_trip" and len(parsed) < 2:
        return True
    return False


def compensation_eligible(user_id: Any) -> bool:
    """Return True if the user qualifies for compensation.

    Eligible when:
      - silver/gold member, OR
      - the user has at least one reservation with a disrupted flight
        (delayed or cancelled status, or reservation.status == "cancelled")
        AND that specific reservation has travel insurance or is business class.

    This prevents a user's unrelated business-cabin reservation from granting
    compensation eligibility for a disrupted basic-economy flight on a
    different reservation.
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
    # Check only reservations that actually have a disrupted flight
    for rid in user.reservations:
        if rid not in db.reservations:
            continue
        res = db.reservations[rid]
        disrupted = res.status == "cancelled"
        if not disrupted:
            for fi in res.flights:
                fn, dt = fi.flight_number, fi.date
                if fn in db.flights and dt in db.flights[fn].dates:
                    if db.flights[fn].dates[dt].status in ("delayed", "cancelled"):
                        disrupted = True
                        break
        if disrupted and (res.insurance == "yes" or res.cabin == "business"):
            return True
    return False


def valid_certificate_amount(user_id: Any, amount: Any) -> bool:
    """Return True if the amount matches $100/pax (cancelled) or $50/pax (delayed)."""
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
            fn, dt = fi.flight_number, fi.date
            if fn in db.flights and dt in db.flights[fn].dates:
                st = db.flights[fn].dates[dt].status
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
