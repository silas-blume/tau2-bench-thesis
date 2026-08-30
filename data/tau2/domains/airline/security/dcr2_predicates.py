"""Predicate library for dcr2.yaml's data-event resolver.

Unlike ``predicates_v4.py`` (written for the Declare-policy path, where a
predicate that can't determine an answer should fail *open* -- "agent must
still verify" -- because Declare only flags violations after the fact),
every function here is a pure computation over already-looked-up domain
objects (``Reservation``, ``User``, ``FlightDB``). It never does its own DB
lookups and never needs a fail-open/fail-closed default, because
``dcr2_data_resolver.py`` (the only caller) is responsible for looking the
entities up and returning ``UNRESOLVED`` itself when one doesn't exist --
these functions are only ever called with real, present objects.

This keeps the DCR-specific "no answer must mean the gate stays blocked"
requirement local to the resolver's plumbing, instead of being smuggled in
as inversions of another file's fail-open defaults.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_if_str(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def _extract_payment_id(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("payment_id", ""))
    if isinstance(item, str):
        return item
    return str(getattr(item, "payment_id", ""))


def _flight_status(db: Any, flight_number: str, date: str) -> str | None:
    if flight_number in db.flights and date in db.flights[flight_number].dates:
        return db.flights[flight_number].dates[date].status
    return None


# ---------------------------------------------------------------------------
# Booking parameter counts
# ---------------------------------------------------------------------------

def count_items(collection: Any) -> int:
    parsed = _parse_json_if_str(collection)
    return len(parsed) if isinstance(parsed, list) else 0


def count_credit_cards(payment_methods: Any) -> int:
    methods = _parse_json_if_str(payment_methods)
    if not isinstance(methods, list):
        return 0
    return sum(1 for m in methods if _extract_payment_id(m).lower().startswith("credit_card"))


def count_gift_cards(payment_methods: Any) -> int:
    methods = _parse_json_if_str(payment_methods)
    if not isinstance(methods, list):
        return 0
    return sum(1 for m in methods if _extract_payment_id(m).lower().startswith("gift_card"))


def count_certificates(payment_methods: Any) -> int:
    methods = _parse_json_if_str(payment_methods)
    if not isinstance(methods, list):
        return 0
    return sum(1 for m in methods if _extract_payment_id(m).lower().startswith("certificate"))


def has_unknown_payment(payment_methods: Any, user: Any) -> bool:
    """True if any submitted payment_id is not in ``user``'s profile."""
    methods = _parse_json_if_str(payment_methods)
    if not isinstance(methods, list):
        return False
    known = set(user.payment_methods.keys())
    return any(_extract_payment_id(m) not in known for m in methods if _extract_payment_id(m))


def payment_in_profile(user: Any, payment_id: Any) -> bool:
    return str(payment_id) in user.payment_methods


def pass_info_complete(passengers: Any) -> bool:
    """True if every passenger has a first name, last name, and DOB."""
    parsed = _parse_json_if_str(passengers)
    if not isinstance(parsed, list) or not parsed:
        return False
    for p in parsed:
        if isinstance(p, dict):
            first, last, dob = p.get("first_name"), p.get("last_name"), p.get("dob")
        else:
            first = getattr(p, "first_name", None)
            last = getattr(p, "last_name", None)
            dob = getattr(p, "dob", None)
        if not first or not last or not dob:
            return False
    return True


_MEMBERSHIP_CODE = {"regular": 0, "silver": 1, "gold": 2}
_CABIN_CODE = {"basic_economy": 0, "economy": 1, "business": 2}


def membership_code(user: Any) -> int:
    return _MEMBERSHIP_CODE.get(user.membership, 0)


def cabin_code(cabin: Any) -> int:
    return _CABIN_CODE.get(str(cabin), 0)


def payment_method_type_ok(payment_id: Any) -> bool:
    """True if payment_id is a gift card or credit card (not a certificate)."""
    pid = str(payment_id).lower()
    return pid.startswith("credit_card") or pid.startswith("gift_card")


# ---------------------------------------------------------------------------
# Reservation facts
# ---------------------------------------------------------------------------

def is_basic_economy(reservation: Any) -> bool:
    return reservation.cabin == "basic_economy"


def has_flown(reservation: Any, db: Any) -> bool:
    for fi in reservation.flights:
        status = _flight_status(db, fi.flight_number, fi.date)
        if status in ("flying", "landed"):
            return True
    return False


def _flight_key(f: Any) -> tuple[str, str]:
    if isinstance(f, dict):
        return (str(f.get("flight_number", "")), str(f.get("date", "")))
    return (str(getattr(f, "flight_number", "")), str(getattr(f, "date", "")))


def flights_changed(reservation: Any, flights: Any) -> bool:
    """True if the submitted flight segments differ from the reservation's
    current ones (by flight_number+date pair, order-independent). Used to
    scope the basic-economy "flights cannot be modified" rule to actual
    flight-segment changes, not cabin-only updates (policy.md explicitly
    allows cabin changes from basic economy as long as flights don't change).
    """
    parsed = _parse_json_if_str(flights)
    if not isinstance(parsed, list):
        return True
    submitted = sorted(_flight_key(f) for f in parsed)
    current = sorted((fi.flight_number, fi.date) for fi in reservation.flights)
    return submitted != current


def route_changed(reservation: Any, flights: Any, db: Any) -> bool:
    """True if the submitted flights would change origin, destination, or
    trip type relative to the reservation's stored route."""
    parsed = _parse_json_if_str(flights)
    if not isinstance(parsed, list) or not parsed:
        return False

    origins, dests = [], []
    for f in parsed:
        fn, _ = _flight_key(f)
        if fn and fn in db.flights:
            origins.append(db.flights[fn].origin)
            dests.append(db.flights[fn].destination)
    if not origins or not dests:
        return False

    if origins[0] != reservation.origin:
        return True
    if reservation.flight_type == "round_trip":
        if len(parsed) < 2:
            return True
        if reservation.destination not in dests:
            return True
    else:
        if dests[-1] != reservation.destination:
            return True
    return False


def cancellation_eligible_base(reservation: Any, db: Any) -> bool:
    """True on grounds other than the 24h window: business class, travel
    insurance (health/weather cancellations), or an airline-cancelled
    flight."""
    if reservation.cabin == "business":
        return True
    if reservation.insurance == "yes":
        return True
    for fi in reservation.flights:
        if _flight_status(db, fi.flight_number, fi.date) == "cancelled":
            return True
    return False


def booking_within_24h(reservation: Any, now_iso: str) -> bool:
    created_at = datetime.fromisoformat(reservation.created_at)
    now = datetime.fromisoformat(now_iso)
    elapsed_seconds = (now - created_at).total_seconds()
    return 0 <= elapsed_seconds <= 86400


def baggage_count(reservation: Any) -> int:
    return reservation.total_baggages


def passenger_count(reservation: Any) -> int:
    return len(reservation.passengers)


# ---------------------------------------------------------------------------
# Compensation
# ---------------------------------------------------------------------------

def compensation_eligible(user: Any, db: Any) -> bool:
    """True when ``user`` is silver/gold, or holds a reservation with a
    disrupted flight (delayed/cancelled) that is insured or business class.

    A user's unrelated business-cabin reservation must not grant eligibility
    for a disrupted basic-economy flight on a different reservation, so
    disruption and insurance/cabin are checked on the *same* reservation.
    """
    if user.membership in ("silver", "gold"):
        return True
    for rid in user.reservations:
        res = db.reservations.get(rid)
        if res is None:
            continue
        disrupted = res.status == "cancelled"
        if not disrupted:
            for fi in res.flights:
                if _flight_status(db, fi.flight_number, fi.date) in ("delayed", "cancelled"):
                    disrupted = True
                    break
        if disrupted and (res.insurance == "yes" or res.cabin == "business"):
            return True
    return False


def valid_certificate_amount(user: Any, amount: Any, db: Any) -> bool:
    """True if ``amount`` matches $100/pax for any of ``user``'s reservations
    with a cancelled flight, or $50/pax for any with a delayed flight.

    Fail-closed by construction: a user with *no* disrupted reservation has
    an empty valid-amounts set, so no amount is considered valid -- unlike
    predicates_v4.valid_certificate_amount, which treats an empty set as
    "nothing to check against" and returns True. That fail-open default is
    correct for Declare (flag only what's provably wrong) but wrong for a
    DCR gate, where "nothing to check against" must mean "stay blocked".
    """
    try:
        amt = int(amount)
    except (ValueError, TypeError):
        return False
    if amt <= 0:
        return False

    valid: set[int] = set()
    for rid in user.reservations:
        res = db.reservations.get(rid)
        if res is None:
            continue
        npax = len(res.passengers)
        cancelled = res.status == "cancelled"
        delayed = False
        for fi in res.flights:
            status = _flight_status(db, fi.flight_number, fi.date)
            if status == "cancelled":
                cancelled = True
            elif status == "delayed":
                delayed = True
        if cancelled:
            valid.add(100 * npax)
        if delayed:
            valid.add(50 * npax)
    return amt in valid
