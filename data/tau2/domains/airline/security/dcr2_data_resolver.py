"""Data-event resolver for dcr2.yaml.

dcr2.yaml gates every state-changing airline tool behind data events
(``reservation_has_flown``, ``booking_payment_methods_valid``, etc.) that must
be *executed* in the DCR graph before the write action is enabled. Nothing
about a plain LLM tool call produces those events on its own -- this module
is the "environment" that supplies them, by looking up the real values from
the FlightDB and from the pending tool call's own arguments.

Loaded dynamically (not a normal package import) by
``tau2.agent.secure_airline_agent`` via the ``<stem>_data_resolver.py``
sibling-file convention. Must expose a top-level
``resolve(event_id, event, graph)`` function matching
``thesis_dpm_secure_langgraph.constraints.data_resolver.DataEventResolver``.

This file owns all DB access and event-argument extraction (the "is the
data even available yet" plumbing); the actual predicate logic lives in
``dcr2_predicates.py`` (also loaded dynamically, from this same directory)
as pure functions over already-looked-up domain objects -- see that file's
docstring for why fail-closed behavior lives here rather than there.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from thesis_dpm_secure_langgraph import UNRESOLVED

_MODULE_DIR = Path(__file__).resolve().parent

# Fixed simulated "now" for the airline domain -- the only notion of "now"
# used anywhere in this domain today (see
# tau2.domains.airline.tools.AirlineTools._get_datetime(), an instance
# method not freely importable here). Keep the two literals in sync.
_SIMULATED_NOW = "2024-05-15T15:00:00"


def _load_sibling(stem: str):
    spec = importlib.util.spec_from_file_location(
        f"tau2_dcr2_{stem}", _MODULE_DIR / f"{stem}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_p = _load_sibling("dcr2_predicates")


def _db():
    from tau2.domains.airline.data_model import get_db
    return get_db()


def _reservation(reservation_id: Any):
    if not reservation_id:
        return None
    return _db().reservations.get(str(reservation_id))


def _user(user_id: Any):
    if not user_id:
        return None
    return _db().users.get(str(user_id))


# ---------------------------------------------------------------------------
# Per-event resolution handlers
# ---------------------------------------------------------------------------

def _booking_num_passengers(event: Any) -> Any:
    return _p.count_items(event.get("passengers"))


def _booking_num_credit_cards(event: Any) -> Any:
    return _p.count_credit_cards(event.get("payment_methods"))


def _booking_num_gift_cards(event: Any) -> Any:
    return _p.count_gift_cards(event.get("payment_methods"))


def _booking_num_certificates(event: Any) -> Any:
    return _p.count_certificates(event.get("payment_methods"))


def _booking_payment_methods_valid(event: Any) -> Any:
    user = _user(event.get("user_id"))
    if user is None:
        return UNRESOLVED
    return not _p.has_unknown_payment(event.get("payment_methods"), user)


def _booking_passengers_info_complete(event: Any) -> Any:
    return _p.pass_info_complete(event.get("passengers"))


def _booking_membership_code(event: Any) -> Any:
    user = _user(event.get("user_id"))
    if user is None:
        return UNRESOLVED
    return _p.membership_code(user)


def _booking_cabin_code(event: Any) -> Any:
    cabin = event.get("cabin")
    if not cabin:
        return UNRESOLVED
    return _p.cabin_code(cabin)


def _booking_total_baggages(event: Any) -> Any:
    value = event.get("total_baggages")
    return UNRESOLVED if value is None else value


def _booking_nonfree_baggages(event: Any) -> Any:
    value = event.get("nonfree_baggages")
    return UNRESOLVED if value is None else value


def _update_payment_method_type_valid(event: Any) -> Any:
    payment_id = event.get("payment_id")
    if not payment_id:
        return UNRESOLVED
    return _p.payment_method_type_ok(payment_id)


def _reservation_flights_changed(event: Any) -> Any:
    reservation = _reservation(event.get("reservation_id"))
    if reservation is None:
        return UNRESOLVED
    return _p.flights_changed(reservation, event.get("flights"))


def _reservation_route_changed(event: Any) -> Any:
    reservation = _reservation(event.get("reservation_id"))
    if reservation is None:
        return UNRESOLVED
    return _p.route_changed(reservation, event.get("flights"), _db())


def _reservation_is_basic_economy(event: Any) -> Any:
    reservation = _reservation(event.get("reservation_id"))
    if reservation is None:
        return UNRESOLVED
    return _p.is_basic_economy(reservation)


def _reservation_has_flown(event: Any) -> Any:
    reservation = _reservation(event.get("reservation_id"))
    if reservation is None:
        return UNRESOLVED
    return _p.has_flown(reservation, _db())


def _reservation_cancellation_eligible_base(event: Any) -> Any:
    reservation = _reservation(event.get("reservation_id"))
    if reservation is None:
        return UNRESOLVED
    return _p.cancellation_eligible_base(reservation, _db())


def _booking_within_24h(event: Any) -> Any:
    reservation = _reservation(event.get("reservation_id"))
    if reservation is None or not reservation.created_at:
        return UNRESOLVED
    try:
        return _p.booking_within_24h(reservation, _SIMULATED_NOW)
    except ValueError:
        return UNRESOLVED


def _reservation_current_bag_count(event: Any) -> Any:
    reservation = _reservation(event.get("reservation_id"))
    if reservation is None:
        return UNRESOLVED
    return _p.baggage_count(reservation)


def _reservation_num_passengers(event: Any) -> Any:
    reservation = _reservation(event.get("reservation_id"))
    if reservation is None:
        return UNRESOLVED
    return _p.passenger_count(reservation)


def _update_payment_in_profile(event: Any) -> Any:
    reservation = _reservation(event.get("reservation_id"))
    payment_id = event.get("payment_id")
    if reservation is None or not payment_id:
        return UNRESOLVED
    user = _user(reservation.user_id)
    if user is None:
        return UNRESOLVED
    return _p.payment_in_profile(user, payment_id)


def _new_total_baggages(event: Any) -> Any:
    value = event.get("total_baggages")
    return UNRESOLVED if value is None else value


def _update_num_passengers(event: Any) -> Any:
    passengers = event.get("passengers")
    if passengers is None:
        return UNRESOLVED
    return _p.count_items(passengers)


def _user_compensation_eligible(event: Any) -> Any:
    user = _user(event.get("user_id"))
    if user is None:
        return UNRESOLVED
    return _p.compensation_eligible(user, _db())


def _certificate_amount_valid(event: Any) -> Any:
    user = _user(event.get("user_id"))
    amount = event.get("amount")
    if user is None or amount is None:
        return UNRESOLVED
    return _p.valid_certificate_amount(user, amount, _db())


_HANDLERS = {
    "booking_num_passengers": _booking_num_passengers,
    "booking_num_credit_cards": _booking_num_credit_cards,
    "booking_num_gift_cards": _booking_num_gift_cards,
    "booking_num_certificates": _booking_num_certificates,
    "booking_payment_methods_valid": _booking_payment_methods_valid,
    "booking_passengers_info_complete": _booking_passengers_info_complete,
    "booking_membership_code": _booking_membership_code,
    "booking_cabin_code": _booking_cabin_code,
    "booking_total_baggages": _booking_total_baggages,
    "booking_nonfree_baggages": _booking_nonfree_baggages,
    "update_payment_method_type_valid": _update_payment_method_type_valid,
    "reservation_flights_changed": _reservation_flights_changed,
    "reservation_route_changed": _reservation_route_changed,
    "reservation_is_basic_economy": _reservation_is_basic_economy,
    "reservation_has_flown": _reservation_has_flown,
    "reservation_cancellation_eligible_base": _reservation_cancellation_eligible_base,
    "booking_within_24h": _booking_within_24h,
    "reservation_current_bag_count": _reservation_current_bag_count,
    "reservation_num_passengers": _reservation_num_passengers,
    "update_payment_in_profile": _update_payment_in_profile,
    "new_total_baggages": _new_total_baggages,
    "update_num_passengers": _update_num_passengers,
    "user_compensation_eligible": _user_compensation_eligible,
    "certificate_amount_valid": _certificate_amount_valid,
}


def resolve(event_id: str, event: Any, graph: Any) -> Any:
    """Resolve a dcr2.yaml input event's value from the pending tool call's
    own arguments plus a FlightDB lookup. Returns
    :data:`thesis_dpm_secure_langgraph.UNRESOLVED` for anything not
    (yet) determinable, which leaves the corresponding DCR gate blocked."""
    handler = _HANDLERS.get(event_id)
    if handler is None:
        return UNRESOLVED
    try:
        return handler(event)
    except Exception:
        return UNRESOLVED
