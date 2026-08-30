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
    # Always check origin
    if origins[0] != res.origin:
        return True
    # Trip-type check: round_trip requires >= 2 submitted flights
    if res.flight_type == "round_trip":
        if len(parsed) < 2:
            return True
        # For round-trip, the outbound destination must appear in the submitted
        # flights. (The last flight returns to origin, so dests[-1] != res.destination
        # is expected and must NOT be used as the destination check here.)
        if res.destination not in dests:
            return True
    else:
        # One-way: the last flight must land at the booking destination
        if dests[-1] != res.destination:
            return True
    return False

def reservation_user_id(reservation_id: Any) -> str:
    """Return the user_id associated with a reservation (DB lookup)."""
    db = _get_db()
    if db is None:
        return ""
    rid = str(reservation_id)
    if rid not in db.reservations:
        return ""
    return db.reservations[rid].user_id

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


def all_flights_available(flights: Any) -> bool:
    """Return True if all flights in the list have status 'available'."""
    db = _get_db()
    if db is None:
        return True
    parsed = _parse_json_if_str(flights)
    if not isinstance(parsed, list):
        return True
    for f in parsed:
        fn = f.get("flight_number", "") if isinstance(f, dict) else str(getattr(f, "flight_number", ""))
        dt = f.get("date", "") if isinstance(f, dict) else str(getattr(f, "date", ""))
        if not fn or not dt:
            continue
        if fn not in db.flights or dt not in db.flights[fn].dates:
            continue
        if db.flights[fn].dates[dt].status != "available":
            return False
    return True

def reservation_destination(reservation_id: Any) -> str:
    """Return the destination of a reservation (DB lookup)."""
    db = _get_db()
    if db is None:
        return ""
    rid = str(reservation_id)
    if rid not in db.reservations:
        return ""
    return db.reservations[rid].destination
    
def reservation_origin(reservation_id: Any) -> str:
    """Return the origin of a reservation (DB lookup)."""
    db = _get_db()
    if db is None:
        return ""
    rid = str(reservation_id)
    if rid not in db.reservations:
        return ""
    return db.reservations[rid].origin


# ===========================================================================
# Flight-update helpers — resolve field values from a submitted flight list
# ===========================================================================

def flight_update_origin(flights: Any) -> str:
    """Return the origin airport of the first flight in the submitted list.

    Correctly identifies the departure point for both one-way and round-trip
    updates (the journey always starts at the first flight's origin).
    Used by the no-change-origin rule.
    """
    parsed = _parse_json_if_str(flights)
    if not isinstance(parsed, list) or not parsed:
        return ""
    db = _get_db()
    if db is None:
        return ""
    first = parsed[0]
    fn = first.get("flight_number", "") if isinstance(first, dict) else str(getattr(first, "flight_number", ""))
    if fn and fn in db.flights:
        return db.flights[fn].origin
    return ""

def flight_update_destination(flights: Any) -> str:
    """Return the booking destination of the submitted flight list.

    For one-way trips this is the destination of the last flight.
    For round trips (last flight returns to first flight's origin) this is
    the destination of the first (outbound) flight — matching how
    reservation.destination is stored for round trips.
    Used by the no-change-destination rule.
    """
    parsed = _parse_json_if_str(flights)
    if not isinstance(parsed, list) or not parsed:
        return ""
    db = _get_db()
    if db is None:
        return ""

    def _fn(f: Any) -> str:
        return f.get("flight_number", "") if isinstance(f, dict) else str(getattr(f, "flight_number", ""))

    first_fn = _fn(parsed[0])
    last_fn = _fn(parsed[-1])
    if not first_fn or not last_fn:
        return ""
    if first_fn not in db.flights or last_fn not in db.flights:
        return ""
    # Round-trip: the return leg lands back at the departure airport.
    # The booking destination is the outbound (first) flight's destination.
    if db.flights[first_fn].origin == db.flights[last_fn].destination:
        return db.flights[first_fn].destination
    # One-way (or multi-leg): destination is where the last flight lands.
    return db.flights[last_fn].destination


def flight_update_trip_type(flights: Any) -> str:
    """Infer the trip type of the submitted flight list.

    Returns 'round_trip' when the first flight's origin equals the last
    flight's destination (the journey returns to its starting point).
    Returns 'one_way' otherwise.  Returns '' on lookup failure.
    Used by the no-change-type rule.
    """
    parsed = _parse_json_if_str(flights)
    if not isinstance(parsed, list) or not parsed:
        return ""
    if len(parsed) == 1:
        return "one_way"
    db = _get_db()
    if db is None:
        return ""

    def _fn(f: Any) -> str:
        return f.get("flight_number", "") if isinstance(f, dict) else str(getattr(f, "flight_number", ""))

    first_fn = _fn(parsed[0])
    last_fn = _fn(parsed[-1])
    if not first_fn or not last_fn:
        return ""
    if first_fn not in db.flights or last_fn not in db.flights:
        return ""
    if db.flights[first_fn].origin == db.flights[last_fn].destination:
        return "round_trip"
    return "one_way"


def reservation_trip_type(reservation_id: Any) -> str:
    """Return the flight_type of a reservation ('one_way' or 'round_trip').

    Used by the no-change-type rule together with flight_update_trip_type.
    """
    db = _get_db()
    if db is None:
        return ""
    rid = str(reservation_id)
    if rid not in db.reservations:
        return ""
    return db.reservations[rid].flight_type


# ===========================================================================
# Cancellation eligibility
# ===========================================================================

def cancellation_eligible(reservation_id: Any) -> bool:
    """Return True if the reservation is eligible for cancellation per policy.

    Eligible when any of the following hold:
      - Business class reservation (always refundable/cancellable)
      - Travel insurance is present (health/weather cancellations covered)
      - At least one flight in the reservation was cancelled by the airline

    NOTE: The 24-hour booking window criterion is deliberately NOT checked here
    because predicates execute against the live DB clock, which would always
    fail for historical simulation scenarios.  Agents must verify the 24h
    window themselves per policy.md.
    """
    db = _get_db()
    if db is None:
        return True  # fail open; agent must still verify
    rid = str(reservation_id)
    if rid not in db.reservations:
        return False
    res = db.reservations[rid]
    # Business class is always cancellable with a refund
    if res.cabin == "business":
        return True
    # Travel insurance covers health/weather cancellations
    if res.insurance == "yes":
        return True
    # Airline-cancelled flight: customer is entitled to a full refund
    for fi in res.flights:
        fn, dt = fi.flight_number, fi.date
        if fn in db.flights and dt in db.flights[fn].dates:
            if db.flights[fn].dates[dt].status == "cancelled":
                return True
    return False


def user_membership_level(user_id: Any) -> str:
    """Return the membership level of a user ('regular', 'silver', 'gold')."""
    db = _get_db()
    if db is None:
        return "regular"
    uid = str(user_id)
    if uid not in db.users:
        return "regular"
    return db.users[uid].membership


def pass_info_complete(passengers: Any) -> bool:
    """Return true if all passengers have complete information (first name, last name, date of birth)."""
    parsed = _parse_json_if_str(passengers)
    if not isinstance(parsed, list) or not parsed:
        return False
    for p in parsed:
        if not isinstance(p, dict):
            return False
        if not p.get("first_name") or not p.get("last_name") or not p.get("dob"):
            return False
    return True

def all_flights_same_cabin_class(flights: Any) -> bool:
    """Return True if all flights in the submitted list share a single cabin class.

    Individual flight dicts in update_reservation_flights carry flight_number
    and date but no per-flight cabin; cabin is set at the reservation level by
    the separate `cabin` argument.  When no per-flight cabin field is present
    the constraint is vacuously satisfied (all flights inherit the same cabin).
    """
    parsed = _parse_json_if_str(flights)
    if not isinstance(parsed, list) or not parsed:
        return True
    cabins = set()
    for f in parsed:
        cabin = f.get("cabin") if isinstance(f, dict) else getattr(f, "cabin", None)
        if cabin is not None:
            cabins.add(cabin)
    return len(cabins) <= 1