"""Integration tests for the dcr2.yaml data-event resolver.

dcr2.yaml gates every write tool behind data events (reservation_has_flown,
booking_payment_methods_valid, etc.) that no ordinary tool call ever
executes on its own. dcr2_data_resolver.py supplies those values from the
real FlightDB + the pending tool call's own arguments, wired through
thesis_dpm_secure_langgraph's data-event resolver hook.

These tests load the *real* dcr2.yaml + dcr2_data_resolver.py against the
*real* db.json (same code path as SecureAirlineAgent.__init__) and replay
tool-call sequences end-to-end, asserting write actions are enabled when the
underlying facts justify it and blocked when they don't -- including the
adversarial cases that would have silently bypassed the pre-fix "last
resolved event wins" bug in the guarded relations.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pm4py.objects.dcr.data.obj import DataDcrGraph
from pm4py.objects.log.obj import Event
from thesis_dpm_secure_langgraph import AgentDCRConstraints, DCRStateValidator, ValidationDecision

from tau2.agent.secure_airline_agent import _load_data_event_resolver
from tau2.domains.airline.data_model import get_db

_SEC_DIR = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "tau2"
    / "domains"
    / "airline"
    / "security"
)


@pytest.fixture()
def resolver():
    return _load_data_event_resolver(_SEC_DIR / "dcr2_data_resolver.py")


@pytest.fixture()
def validator_and_tracker(resolver):
    constraints = AgentDCRConstraints().parse_from_yaml(
        str(_SEC_DIR / "dcr2.yaml"),
        data_event_resolver=resolver,
        predicate_file_path=_SEC_DIR / "dcr2_expr_predicates.py",
    )
    assert isinstance(constraints.to_dcr_graph(), DataDcrGraph)
    return DCRStateValidator(constraints), constraints.get_state_tracker()


def _call(validator, tracker, name, args):
    decision, error = validator.validate({"name": name, "args": args})
    if decision == ValidationDecision.ALLOW:
        tracker.on_trace_event_published(
            Event({"concept:name": name, "status": "complete", **args})
        )
    return decision, error


class TestCancelReservation:
    def test_business_class_unflown_reservation_can_be_cancelled(self, validator_and_tracker):
        """Regression test for the original bug report: MZDDS4 is business
        class and unflown, so cancel_reservation must be reachable once the
        agent has called the required read tools."""
        validator, tracker = validator_and_tracker
        db = get_db()
        res = db.reservations["MZDDS4"]

        _call(validator, tracker, "get_user_details", {"user_id": res.user_id})
        _call(validator, tracker, "get_reservation_details", {"reservation_id": "MZDDS4"})
        for f in res.flights:
            _call(validator, tracker, "get_flight_status", {"flight_number": f.flight_number, "date": f.date})

        decision, error = _call(
            validator, tracker, "cancel_reservation",
            {"reservation_id": "MZDDS4", "reason": "business class cancellation"},
        )
        assert decision == ValidationDecision.ALLOW, error

    def test_ineligible_economy_reservation_stays_blocked(self, validator_and_tracker):
        validator, tracker = validator_and_tracker
        db = get_db()
        res = db.reservations["VAAOXJ"]
        assert res.cabin != "business" and res.insurance != "yes"

        _call(validator, tracker, "get_user_details", {"user_id": res.user_id})
        _call(validator, tracker, "get_reservation_details", {"reservation_id": "VAAOXJ"})
        for f in res.flights:
            _call(validator, tracker, "get_flight_status", {"flight_number": f.flight_number, "date": f.date})

        decision, error = _call(
            validator, tracker, "cancel_reservation",
            {"reservation_id": "VAAOXJ", "reason": "changed my mind"},
        )
        assert decision == ValidationDecision.DECLINE
        assert error is not None

    def test_cannot_cancel_before_calling_read_tools(self, validator_and_tracker):
        """The unguarded conditions (call get_user_details/get_reservation_details/
        get_flight_status first) must still be enforced independently of the
        data-event resolver."""
        validator, tracker = validator_and_tracker
        decision, error = _call(
            validator, tracker, "cancel_reservation",
            {"reservation_id": "MZDDS4", "reason": "change of plan"},
        )
        assert decision == ValidationDecision.DECLINE


class TestBookReservation:
    """Also exercises the fix for the "last resolved event wins" bug: each
    of these adversarial cases has exactly one check failing while the other
    passes, which would have been silently allowed by the pre-fix graph
    depending on the (arbitrary) resolution order of the two independent
    guarded relations."""

    def _valid_payment_id(self, user_id: str) -> str:
        db = get_db()
        return next(iter(db.users[user_id].payment_methods))

    def test_valid_params_and_payment_method_allowed(self, validator_and_tracker):
        validator, tracker = validator_and_tracker
        user_id = "lei_rossi_3206"
        _call(validator, tracker, "get_user_details", {"user_id": user_id})

        decision, error = _call(
            validator, tracker, "book_reservation",
            {
                "user_id": user_id, "origin": "AAA", "destination": "BBB",
                "flight_type": "one_way", "cabin": "economy", "flights": [],
                "passengers": [{"first_name": "A", "last_name": "B", "dob": "2000-01-01"}],
                "payment_methods": [{"payment_id": self._valid_payment_id(user_id), "amount": 100}],
                "total_baggages": 0, "nonfree_baggages": 0, "insurance": "no",
            },
        )
        assert decision == ValidationDecision.ALLOW, error

    def test_unknown_payment_method_blocked_even_with_valid_params(self, validator_and_tracker):
        validator, tracker = validator_and_tracker
        user_id = "lei_rossi_3206"
        _call(validator, tracker, "get_user_details", {"user_id": user_id})

        decision, _ = _call(
            validator, tracker, "book_reservation",
            {
                "user_id": user_id, "origin": "AAA", "destination": "BBB",
                "flight_type": "one_way", "cabin": "economy", "flights": [],
                "passengers": [{"first_name": "A", "last_name": "B", "dob": "2000-01-01"}],
                "payment_methods": [{"payment_id": "credit_card_doesnotexist", "amount": 100}],
                "total_baggages": 0, "nonfree_baggages": 0, "insurance": "no",
            },
        )
        assert decision == ValidationDecision.DECLINE

    def test_too_many_passengers_blocked_even_with_valid_payment(self, validator_and_tracker):
        validator, tracker = validator_and_tracker
        user_id = "lei_rossi_3206"
        _call(validator, tracker, "get_user_details", {"user_id": user_id})

        decision, _ = _call(
            validator, tracker, "book_reservation",
            {
                "user_id": user_id, "origin": "AAA", "destination": "BBB",
                "flight_type": "one_way", "cabin": "economy", "flights": [],
                "passengers": [{"first_name": "A", "last_name": "B", "dob": "2000-01-01"}] * 6,
                "payment_methods": [{"payment_id": self._valid_payment_id(user_id), "amount": 100}],
                "total_baggages": 0, "nonfree_baggages": 0, "insurance": "no",
            },
        )
        assert decision == ValidationDecision.DECLINE


class TestSendCertificate:
    def test_correct_amount_for_eligible_user_allowed(self, validator_and_tracker):
        validator, tracker = validator_and_tracker
        db = get_db()
        # Find a user with a cancelled- or delayed-flight reservation so
        # valid_certificate_amount() has a non-empty valid-amounts set.
        target_user_id = None
        target_reservation_id = None
        target_amount = None
        for user_id, user in db.users.items():
            for rid in user.reservations:
                res = db.reservations.get(rid)
                if res is None:
                    continue
                npax = len(res.passengers)
                for fi in res.flights:
                    fn, dt = fi.flight_number, fi.date
                    status = db.flights[fn].dates[dt].status if fn in db.flights and dt in db.flights[fn].dates else None
                    if status == "cancelled":
                        target_user_id, target_reservation_id, target_amount = user_id, rid, 100 * npax
                    elif status == "delayed" and target_user_id is None:
                        target_user_id, target_reservation_id, target_amount = user_id, rid, 50 * npax
        if target_user_id is None:
            pytest.skip("No disrupted-flight reservation in fixture db.json")

        res = db.reservations[target_reservation_id]
        _call(validator, tracker, "get_user_details", {"user_id": target_user_id})
        _call(validator, tracker, "get_reservation_details", {"reservation_id": target_reservation_id})
        for f in res.flights:
            _call(validator, tracker, "get_flight_status", {"flight_number": f.flight_number, "date": f.date})
        decision, error = _call(
            validator, tracker, "send_certificate",
            {"user_id": target_user_id, "amount": target_amount},
        )
        assert decision == ValidationDecision.ALLOW, error

    def test_wrong_amount_blocked(self, validator_and_tracker):
        validator, tracker = validator_and_tracker
        user_id = "lei_rossi_3206"
        _call(validator, tracker, "get_user_details", {"user_id": user_id})

        decision, _ = _call(
            validator, tracker, "send_certificate",
            {"user_id": user_id, "amount": 999999},
        )
        assert decision == ValidationDecision.DECLINE
