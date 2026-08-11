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

Built on ``thesis_dpm_secure_langgraph``'s ``ResolverRegistry``: 20 of the 24
input events are a plain "extract a tool-call argument, optionally look up
one DB entity by it, call a predicate" shape, expressed below as one-line
``@registry.input_event(...)`` registrations. The remaining 4
(``booking_cabin_code``, ``update_payment_method_type_valid``,
``booking_within_24h``, ``update_payment_in_profile``) don't fit that shape
exactly -- either their missing-value check isn't a plain ``is None`` (a
falsy check that also rejects an empty string), or they need a value derived
from an already-looked-up entity's own attribute rather than from a second
raw event field -- and are registered directly against the raw ``event``
instead, preserving their exact original behavior. Either way, this file
owns DB access and event-argument extraction (the "is the data even
available yet" plumbing); the actual predicate logic lives in
``dcr2_predicates.py`` (also loaded dynamically, from this same directory)
as pure functions over already-looked-up domain objects -- see that file's
docstring for why fail-closed behavior lives here rather than there.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from thesis_dpm_secure_langgraph import UNRESOLVED, ResolverRegistry

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


registry = ResolverRegistry(entity_resolvers={"reservation": _reservation, "user": _user})


# ---------------------------------------------------------------------------
# Sugared handlers -- extract args / look up one entity / call a predicate
# ---------------------------------------------------------------------------

@registry.input_event("booking_num_passengers", args=("passengers",))
def booking_num_passengers(passengers):
    return _p.count_items(passengers)


@registry.input_event("booking_num_credit_cards", args=("payment_methods",))
def booking_num_credit_cards(payment_methods):
    return _p.count_credit_cards(payment_methods)


@registry.input_event("booking_num_gift_cards", args=("payment_methods",))
def booking_num_gift_cards(payment_methods):
    return _p.count_gift_cards(payment_methods)


@registry.input_event("booking_num_certificates", args=("payment_methods",))
def booking_num_certificates(payment_methods):
    return _p.count_certificates(payment_methods)


@registry.input_event("booking_payment_methods_valid", args=("payment_methods",), lookup="user")
def booking_payment_methods_valid(user, payment_methods):
    return not _p.has_unknown_payment(payment_methods, user)


@registry.input_event("booking_passengers_info_complete", args=("passengers",))
def booking_passengers_info_complete(passengers):
    return _p.pass_info_complete(passengers)


@registry.input_event("booking_membership_code", lookup="user")
def booking_membership_code(user):
    return _p.membership_code(user)


@registry.input_event("booking_total_baggages", args=("total_baggages",), require=("total_baggages",))
def booking_total_baggages(total_baggages):
    return total_baggages


@registry.input_event("booking_nonfree_baggages", args=("nonfree_baggages",), require=("nonfree_baggages",))
def booking_nonfree_baggages(nonfree_baggages):
    return nonfree_baggages


@registry.input_event("reservation_flights_changed", args=("flights",), lookup="reservation")
def reservation_flights_changed(reservation, flights):
    return _p.flights_changed(reservation, flights)


@registry.input_event("reservation_route_changed", args=("flights",), lookup="reservation")
def reservation_route_changed(reservation, flights):
    return _p.route_changed(reservation, flights, _db())


@registry.input_event("reservation_is_basic_economy", lookup="reservation")
def reservation_is_basic_economy(reservation):
    return _p.is_basic_economy(reservation)


@registry.input_event("reservation_has_flown", lookup="reservation")
def reservation_has_flown(reservation):
    return _p.has_flown(reservation, _db())


@registry.input_event("reservation_cancellation_eligible_base", lookup="reservation")
def reservation_cancellation_eligible_base(reservation):
    return _p.cancellation_eligible_base(reservation, _db())


@registry.input_event("reservation_current_bag_count", lookup="reservation")
def reservation_current_bag_count(reservation):
    return _p.baggage_count(reservation)


@registry.input_event("reservation_num_passengers", lookup="reservation")
def reservation_num_passengers(reservation):
    return _p.passenger_count(reservation)


@registry.input_event("new_total_baggages", args=("total_baggages",), require=("total_baggages",))
def new_total_baggages(total_baggages):
    return total_baggages


@registry.input_event("update_num_passengers", args=("passengers",), require=("passengers",))
def update_num_passengers(passengers):
    return _p.count_items(passengers)


@registry.input_event("user_compensation_eligible", lookup="user")
def user_compensation_eligible(user):
    return _p.compensation_eligible(user, _db())


@registry.input_event("certificate_amount_valid", args=("amount",), require=("amount",), lookup="user")
def certificate_amount_valid(user, amount):
    return _p.valid_certificate_amount(user, amount, _db())


# ---------------------------------------------------------------------------
# Escape-hatch handlers -- don't fit the sugar exactly, kept explicit rather
# than bending the registry API to special-case them (see module docstring)
# ---------------------------------------------------------------------------

@registry.input_event("booking_cabin_code")
def booking_cabin_code(event: Any) -> Any:
    # Falsy check (not a plain `is None`) -- an empty-string cabin is also
    # treated as missing, matching the original handler exactly.
    cabin = event.get("cabin")
    if not cabin:
        return UNRESOLVED
    return _p.cabin_code(cabin)


@registry.input_event("update_payment_method_type_valid")
def update_payment_method_type_valid(event: Any) -> Any:
    # Falsy check, same reasoning as booking_cabin_code above.
    payment_id = event.get("payment_id")
    if not payment_id:
        return UNRESOLVED
    return _p.payment_method_type_ok(payment_id)


@registry.input_event("booking_within_24h")
def booking_within_24h(event: Any) -> Any:
    # Post-lookup attribute check (reservation.created_at) beyond a plain
    # entity-found check -- not expressible via lookup= alone.
    reservation = _reservation(event.get("reservation_id"))
    if reservation is None or not reservation.created_at:
        return UNRESOLVED
    return _p.booking_within_24h(reservation, _SIMULATED_NOW)


@registry.input_event("update_payment_in_profile")
def update_payment_in_profile(event: Any) -> Any:
    # Chained lookup: `user` is derived from `reservation.user_id` (an
    # already-looked-up entity's own attribute), not from a raw event
    # field -- the one shape ResolverRegistry's lookup= mode deliberately
    # doesn't cover (see resolver_registry.py's docstring).
    reservation = _reservation(event.get("reservation_id"))
    payment_id = event.get("payment_id")
    if reservation is None or not payment_id:
        return UNRESOLVED
    user = _user(reservation.user_id)
    if user is None:
        return UNRESOLVED
    return _p.payment_in_profile(user, payment_id)


resolve = registry.resolve
