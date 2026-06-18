"""
Airline policy predicate checkers.

Each top-level callable is available by name in YAML ``where`` clauses.
Predicates receive the *runtime values* of the referenced trace-event
attributes.  Complex arguments (lists, dicts) arrive as JSON strings when
coming from the DECLARE trace; this module handles both parsed objects and
raw JSON strings transparently.

Database-aware predicates load the current FlightDB snapshot so they can
cross-reference reservation / user state that is not available in the
tool-call arguments alone.
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
    """Lazily load the airline FlightDB.

    Deferred import avoids circular-import issues and keeps the module
    importable even when tau2 internals are not on ``sys.path`` (the
    Declare4Py loader only needs the callables).
    """
    try:
        from tau2.domains.airline.data_model import FlightDB, get_db
        return get_db()
    except Exception:
        # Fallback: try loading directly from the well-known JSON path.
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


# ===========================================================================
# J. Booking — max 5 passengers
# ===========================================================================

def too_many_passengers(passengers: Any) -> bool:
    """Return True if the passenger list exceeds 5 entries.

    Policy: "Each reservation can have at most five passengers."
    """
    passengers = _parse_json_if_str(passengers)
    if isinstance(passengers, list):
        return len(passengers) > 5
    return False


# ===========================================================================
# K. Booking — payment method limits
# ===========================================================================

def exceeds_payment_limits(payment_methods: Any) -> bool:
    """Return True if payment methods exceed the allowed limits.

    Policy: "Each reservation can use at most one travel certificate,
    at most one credit card, and at most three gift cards."

    Payment objects have a ``payment_id`` whose prefix reveals the type:
    ``credit_card_*``, ``gift_card_*``, ``certificate_*``.
    """
    methods = _parse_json_if_str(payment_methods)
    if not isinstance(methods, list):
        return False

    credit_cards = 0
    gift_cards = 0
    certificates = 0

    for m in methods:
        pid = ""
        if isinstance(m, dict):
            pid = str(m.get("payment_id", ""))
        elif isinstance(m, str):
            pid = m
        else:
            pid = str(getattr(m, "payment_id", ""))

        pid_lower = pid.lower()
        if pid_lower.startswith("credit_card"):
            credit_cards += 1
        elif pid_lower.startswith("gift_card"):
            gift_cards += 1
        elif pid_lower.startswith("certificate"):
            certificates += 1

    return credit_cards > 1 or gift_cards > 3 or certificates > 1


# ===========================================================================
# L. Booking — payment methods must be in user profile
# ===========================================================================

def invalid_payment_methods(payment_methods: Any, user_id: Any) -> bool:
    """Return True if any payment method is not in the user's profile.

    Policy: "All payment methods must already be in user profile for
    safety reasons."
    """
    methods = _parse_json_if_str(payment_methods)
    if not isinstance(methods, list):
        return False

    db = _get_db()
    if db is None:
        return False  # Cannot verify without DB — allow optimistically

    user_id = str(user_id)
    if user_id not in db.users:
        return True  # Unknown user — flag as invalid

    user = db.users[user_id]
    user_payment_ids = set(user.payment_methods.keys())

    for m in methods:
        pid = ""
        if isinstance(m, dict):
            pid = str(m.get("payment_id", ""))
        elif isinstance(m, str):
            pid = m
        else:
            pid = str(getattr(m, "payment_id", ""))

        if pid and pid not in user_payment_ids:
            return True

    return False


# ===========================================================================
# M. Passengers — count cannot change on update
# ===========================================================================

def passenger_count_changed(reservation_id: Any, passengers: Any) -> bool:
    """Return True if the new passenger count differs from the reservation.

    Policy: "The user can modify passengers but cannot modify the number
    of passengers."
    """
    passengers = _parse_json_if_str(passengers)
    if not isinstance(passengers, list):
        return False

    db = _get_db()
    if db is None:
        return False

    reservation_id = str(reservation_id)
    if reservation_id not in db.reservations:
        return False

    reservation = db.reservations[reservation_id]
    return len(passengers) != len(reservation.passengers)


# ===========================================================================
# N. Baggage — can only add, not remove
# ===========================================================================

def baggage_decreased(reservation_id: Any, total_baggages: Any) -> bool:
    """Return True if the new total baggage count is less than the current.

    Policy: "The user can add but not remove checked bags."
    """
    db = _get_db()
    if db is None:
        return False

    reservation_id = str(reservation_id)
    if reservation_id not in db.reservations:
        return False

    reservation = db.reservations[reservation_id]
    try:
        new_total = int(total_baggages)
    except (ValueError, TypeError):
        return False

    return new_total < reservation.total_baggages


# ===========================================================================
# P. Flight modification — origin/destination/trip type must not change
# ===========================================================================

def route_changed(reservation_id: Any, flights: Any) -> bool:
    """Return True if the flight update changes origin, destination, or
    trip type relative to the existing reservation.

    Policy: "Other reservations can be modified without changing the
    origin, destination, and trip type."

    The origin is determined by the first flight's origin and the
    destination by the last flight's destination.  For round trips,
    the return leg's destination must equal the original origin.
    """
    db = _get_db()
    if db is None:
        return False

    reservation_id = str(reservation_id)
    if reservation_id not in db.reservations:
        return False

    reservation = db.reservations[reservation_id]
    flights_parsed = _parse_json_if_str(flights)
    if not isinstance(flights_parsed, list) or len(flights_parsed) == 0:
        return False

    # Resolve new flight origin/destination from the flight DB
    new_origins = []
    new_destinations = []
    for f in flights_parsed:
        if isinstance(f, dict):
            fn = f.get("flight_number", "")
        else:
            fn = str(getattr(f, "flight_number", ""))

        if fn and fn in db.flights:
            flight_obj = db.flights[fn]
            new_origins.append(flight_obj.origin)
            new_destinations.append(flight_obj.destination)

    if not new_origins or not new_destinations:
        return False

    new_origin = new_origins[0]
    new_destination = new_destinations[-1]

    # Check trip type consistency
    old_origin = reservation.origin
    old_destination = reservation.destination
    old_type = reservation.flight_type

    # Origin or destination changed
    if new_origin != old_origin or new_destination != old_destination:
        return True

    # For round trips, verify the structure is preserved
    if old_type == "round_trip":
        # A round trip should return to origin
        # The last flight destination should be the origin
        if new_destination != old_origin:
            # Already caught above, but double-check round-trip semantics
            pass
        # Number of legs should imply a return
        if len(flights_parsed) < 2:
            return True  # Round trip needs at least 2 flights

    return False


# ===========================================================================
# Q. Cancellation — no flown flights
# ===========================================================================

def has_flown_flights(reservation_id: Any) -> bool:
    """Return True if any flight in the reservation has already been flown.

    Policy: "If any portion of the flight has already been flown, the
    agent cannot help and transfer is needed."

    Also used for cabin change guard: "Cabin cannot be changed if any
    flight in the reservation has already been flown."

    Flight status values that indicate a flight has been flown or is
    currently flying: 'flying', 'landed'.
    """
    db = _get_db()
    if db is None:
        return False

    reservation_id = str(reservation_id)
    if reservation_id not in db.reservations:
        return False

    reservation = db.reservations[reservation_id]
    for flight_info in reservation.flights:
        fn = flight_info.flight_number
        date = flight_info.date
        if fn in db.flights and date in db.flights[fn].dates:
            status = db.flights[fn].dates[date].status
            if status in ("flying", "landed"):
                return True

    return False


# ===========================================================================
# S. Compensation eligibility
# ===========================================================================

def compensation_eligible(user_id: Any) -> bool:
    """Return True if the user is eligible for compensation.

    Policy: "Only compensate if the user is a silver/gold member or has
    travel insurance or flies business."

    Eligibility criteria (any one is sufficient):
    - User is silver or gold member
    - Any of the user's reservations has travel insurance
    - Any of the user's reservations is business class

    "Do not compensate if the user is regular member and has no travel
    insurance and flies (basic) economy."
    """
    db = _get_db()
    if db is None:
        return True  # Cannot verify — allow optimistically

    user_id = str(user_id)
    if user_id not in db.users:
        return False

    user = db.users[user_id]

    # Silver/gold membership is sufficient
    if user.membership in ("silver", "gold"):
        return True

    # Check reservations for insurance or business class
    for res_id in user.reservations:
        if res_id in db.reservations:
            res = db.reservations[res_id]
            if res.insurance == "yes":
                return True
            if res.cabin == "business":
                return True

    return False


# ===========================================================================
# T. Certificate amount validation
# ===========================================================================

def valid_certificate_amount(user_id: Any, amount: Any) -> bool:
    """Return True if the certificate amount matches policy guidelines.

    Policy:
    - Cancelled flights: $100 × number of passengers
    - Delayed flights: $50 × number of passengers

    The function checks whether the amount is a valid multiple for any
    of the user's reservations.
    """
    db = _get_db()
    if db is None:
        return True  # Cannot verify — allow optimistically

    try:
        amount = int(amount)
    except (ValueError, TypeError):
        return False

    if amount <= 0:
        return False

    user_id = str(user_id)
    if user_id not in db.users:
        return False

    user = db.users[user_id]
    valid_amounts = set()

    for res_id in user.reservations:
        if res_id not in db.reservations:
            continue
        res = db.reservations[res_id]
        n_passengers = len(res.passengers)

        # Check if any flight is cancelled → $100/passenger
        has_cancelled = False
        has_delayed = False
        for flight_info in res.flights:
            fn = flight_info.flight_number
            date = flight_info.date
            if fn in db.flights and date in db.flights[fn].dates:
                status = db.flights[fn].dates[date].status
                if status == "cancelled":
                    has_cancelled = True
                elif status == "delayed":
                    has_delayed = True

        if res.status == "cancelled":
            # Reservation itself was cancelled (by airline or user)
            has_cancelled = True

        if has_cancelled:
            valid_amounts.add(100 * n_passengers)
        if has_delayed:
            valid_amounts.add(50 * n_passengers)

    return amount in valid_amounts if valid_amounts else True
