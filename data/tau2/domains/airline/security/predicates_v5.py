"""
Airline policy predicate checkers — v3 (minimal, YAML-heavy).

Companion to policy_v5.yaml, which binds facts from the results of the
lookups it already requires (`get_user_details -> membership`,
`get_reservation_details -> total_baggages, passengers, user_id`).  Ten v4
predicates became unnecessary that way and were removed:

    user_membership_level, reservation_baggage_count, reservation_passenger_count,
    reservation_origin, reservation_destination, reservation_trip_type,
    flight_update_origin, flight_update_destination, flight_update_trip_type,
    all_flights_same_cabin_class

What is left is exactly what the where-grammar cannot express:

  * generic operators (count_items, count_credit_cards, count_gift_cards,
    count_certificates, payment_method_type_ok) — domain-independent;
  * quantification over a collection (pass_info_complete, has_unknown_payment,
    all_flights_available);
  * comparison of two collections (flights_changed, route_changed);
  * a stored fact no lookup result carries (has_flown_flights, reservation_cabin);
  * membership in a profile / cross-entity search (payment_in_profile,
    reservation_user_id, cancellation_eligible, compensation_eligible,
    valid_certificate_amount).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

# Matches dcr2_data_resolver.py's _SIMULATED_NOW: predicates run against the
# live DB clock, which would always fail the 24h check for historical
# simulation scenarios, so both formalisms pin "now" to the same fixed
# instant instead of using datetime.now().
_SIMULATED_NOW = "2024-05-15T15:00:00"


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
    """Return True if the amount matches $100/pax (cancelled) or $50/pax (delayed).

    Fail-closed: a user with no disrupted reservation has an empty
    valid-amounts set, so no amount is valid -- previously this fell back to
    `True` ("nothing to check against"), which is exactly the gap
    dcr2_predicates.valid_certificate_amount's docstring calls out as wrong
    for a guardrail (its fix is mirrored here).
    """
    db = _get_db()
    if db is None:
        return False
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
    return amt in valid


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

def booking_within_24h(reservation_id: Any) -> bool:
    """Return True if the reservation was created within the last 24 hours
    of the simulated "now" instant (see _SIMULATED_NOW), mirroring dcr2's
    booking_within_24h (dcr2_data_resolver.py / dcr2_predicates.py)."""
    db = _get_db()
    if db is None:
        return False  # fail closed; nothing to check against
    rid = str(reservation_id)
    if rid not in db.reservations:
        return False
    res = db.reservations[rid]
    created = datetime.fromisoformat(res.created_at)
    now = datetime.fromisoformat(_SIMULATED_NOW)
    return 0 <= (now - created).total_seconds() <= 86400


def cancellation_eligible(reservation_id: Any) -> bool:
    """Return True if the reservation is eligible for cancellation per policy.

    Eligible when any of the following hold:
      - Booked within the last 24 hours (see booking_within_24h)
      - Business class reservation (always refundable/cancellable)
      - Travel insurance is present (health/weather cancellations covered)
      - At least one flight in the reservation was cancelled by the airline

    NOTE: this still doesn't verify the cancellation *reason* matches
    "health/weather" for the insurance branch -- dcr2's cancellation_eligible_base
    has the identical gap (rules-translation-dcr-analysis.md, R35), so this
    is not a place where Declare is behind DCR; both are equally permissive
    here.
    """
    db = _get_db()
    if db is None:
        return False  # fail closed; agent must still verify
    rid = str(reservation_id)
    if rid not in db.reservations:
        return False
    if booking_within_24h(reservation_id):
        return True
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



def flights_changed(reservation_id: Any, flights: Any) -> bool:
    """Return True if the submitted segments differ from the reservation's current
    ones (by flight_number+date, order-independent).

    Scopes the basic-economy rule to actual flight changes: policy.md allows a
    basic-economy reservation to change cabin as long as the segments are
    resubmitted unchanged.  Fails closed (True) when the reservation is unknown.
    """
    db = _get_db()
    if db is None:
        return True
    rid = str(reservation_id)
    if rid not in db.reservations:
        return True
    parsed = _parse_json_if_str(flights)
    if not isinstance(parsed, list):
        return True

    def _key(f: Any) -> tuple[str, str]:
        if isinstance(f, dict):
            return (str(f.get("flight_number", "")), str(f.get("date", "")))
        return (str(getattr(f, "flight_number", "")), str(getattr(f, "date", "")))

    submitted = sorted(_key(f) for f in parsed)
    current = sorted((fi.flight_number, fi.date) for fi in db.reservations[rid].flights)
    return submitted != current


def payment_in_profile(payment_id: Any, user_id: Any) -> bool:
    """Return True if payment_id is one of the user's stored payment methods."""
    db = _get_db()
    if db is None:
        return False
    uid = str(user_id)
    if uid not in db.users:
        return False
    return str(payment_id) in db.users[uid].payment_methods


def payment_method_type_ok(payment_id: Any) -> bool:
    """Return True if payment_id is a gift card or credit card (not a certificate)."""
    pid = str(payment_id).lower()
    return pid.startswith("credit_card") or pid.startswith("gift_card")
