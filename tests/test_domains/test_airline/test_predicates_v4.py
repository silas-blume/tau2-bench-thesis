"""Tests for predicates_v4.py — one test per predicate, real DB objects."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from tau2.domains.airline.data_model import FlightDB


# ---------------------------------------------------------------------------
# Minimal DB fixture
# ---------------------------------------------------------------------------

MODULE = "tau2.domains.airline.security.predicates_v4"

# Re-export from the src copy so imports resolve
import sys as _sys
import importlib as _importlib
_src_mod = _importlib.import_module("tau2.domains.airline.security.predicates_v4")
_sys.modules.setdefault("tau2.domains.airline.security.predicates_v4", _src_mod)


@pytest.fixture
def db() -> FlightDB:
    """Minimal FlightDB covering all predicate paths."""
    return FlightDB(
        flights={
            "F001": {
                "flight_number": "F001",
                "origin": "JFK",
                "destination": "LAX",
                "scheduled_departure_time_est": "08:00:00",
                "scheduled_arrival_time_est": "11:00:00",
                "dates": {
                    "2024-06-01": {
                        "status": "available",
                        "available_seats": {"basic_economy": 10, "economy": 10, "business": 5},
                        "prices": {"basic_economy": 100, "economy": 200, "business": 500},
                    },
                    "2024-06-02": {"status": "landed", "actual_departure_time_est": "2024-06-02T08:05:00", "actual_arrival_time_est": "2024-06-02T11:10:00"},
                    "2024-06-03": {"status": "flying", "actual_departure_time_est": "2024-06-03T08:05:00", "estimated_arrival_time_est": "2024-06-03T11:10:00"},
                    "2024-06-04": {"status": "cancelled"},
                    "2024-06-05": {"status": "delayed", "estimated_departure_time_est": "2024-06-05T09:00:00", "estimated_arrival_time_est": "2024-06-05T12:00:00"},
                },
            },
            "F002": {
                "flight_number": "F002",
                "origin": "LAX",
                "destination": "JFK",
                "scheduled_departure_time_est": "14:00:00",
                "scheduled_arrival_time_est": "22:00:00",
                "dates": {
                    "2024-06-01": {
                        "status": "available",
                        "available_seats": {"basic_economy": 10, "economy": 10, "business": 5},
                        "prices": {"basic_economy": 100, "economy": 200, "business": 500},
                    },
                },
            },
        },
        users={
            "user_regular": {
                "user_id": "user_regular",
                "name": {"first_name": "John", "last_name": "Doe"},
                "address": {
                    "address1": "1 Main St", "address2": "",
                    "city": "NY", "country": "USA", "state": "NY", "zip": "10001",
                },
                "email": "john@example.com",
                "dob": "1990-01-01",
                "payment_methods": {
                    "credit_card_001": {"source": "credit_card", "brand": "visa", "last_four": "1234", "id": "credit_card_001"},
                    "gift_card_001": {"source": "gift_card", "brand": "generic", "amount": 200, "id": "gift_card_001"},
                    "certificate_001": {"source": "certificate", "amount": 100, "id": "certificate_001"},
                },
                "saved_passengers": [],
                "membership": "regular",
                "reservations": ["RES001", "RES_FLOWN", "RES_CANCELLED", "RES_INSURED"],
            },
            "user_silver": {
                "user_id": "user_silver",
                "name": {"first_name": "Jane", "last_name": "Doe"},
                "address": {
                    "address1": "2 Oak Ave", "address2": "",
                    "city": "LA", "country": "USA", "state": "CA", "zip": "90001",
                },
                "email": "jane@example.com",
                "dob": "1985-05-15",
                "payment_methods": {},
                "saved_passengers": [],
                "membership": "silver",
                "reservations": ["RES_SILVER_DELAY"],
            },
            "user_gold": {
                "user_id": "user_gold",
                "name": {"first_name": "Bob", "last_name": "Smith"},
                "address": {
                    "address1": "3 Pine Rd", "address2": "",
                    "city": "Chicago", "country": "USA", "state": "IL", "zip": "60601",
                },
                "email": "bob@example.com",
                "dob": "1975-11-30",
                "payment_methods": {},
                "saved_passengers": [],
                "membership": "gold",
                "reservations": [],
            },
            # Regular member with only an active (non-disrupted) uninsured economy reservation
            "user_clean": {
                "user_id": "user_clean",
                "name": {"first_name": "Clean", "last_name": "Slate"},
                "address": {
                    "address1": "4 Clear St", "address2": "",
                    "city": "NY", "country": "USA", "state": "NY", "zip": "10002",
                },
                "email": "clean@example.com",
                "dob": "1995-03-10",
                "payment_methods": {},
                "saved_passengers": [],
                "membership": "regular",
                "reservations": ["RES_CLEAN"],
            },
        },
        reservations={
            # One-way, economy, 2 pax, active flights
            "RES001": {
                "reservation_id": "RES001",
                "user_id": "user_regular",
                "origin": "JFK",
                "destination": "LAX",
                "flight_type": "one_way",
                "cabin": "economy",
                "flights": [
                    {"flight_number": "F001", "date": "2024-06-01", "origin": "JFK", "destination": "LAX", "price": 200},
                ],
                "passengers": [
                    {"first_name": "John", "last_name": "Doe", "dob": "1990-01-01"},
                    {"first_name": "Jane", "last_name": "Doe", "dob": "1985-05-15"},
                ],
                "payment_history": [{"payment_id": "credit_card_001", "amount": 400}],
                "created_at": "2024-05-01T10:00:00",
                "total_baggages": 4,
                "nonfree_baggages": 2,
                "insurance": "no",
            },
            # Flight already landed — has_flown_flights → True
            "RES_FLOWN": {
                "reservation_id": "RES_FLOWN",
                "user_id": "user_regular",
                "origin": "JFK",
                "destination": "LAX",
                "flight_type": "one_way",
                "cabin": "economy",
                "flights": [
                    {"flight_number": "F001", "date": "2024-06-02", "origin": "JFK", "destination": "LAX", "price": 200},
                ],
                "passengers": [{"first_name": "John", "last_name": "Doe", "dob": "1990-01-01"}],
                "payment_history": [],
                "created_at": "2024-05-01T10:00:00",
                "total_baggages": 1,
                "nonfree_baggages": 0,
                "insurance": "no",
            },
            # Airline-cancelled flight → cancellation_eligible True
            "RES_CANCELLED": {
                "reservation_id": "RES_CANCELLED",
                "user_id": "user_regular",
                "origin": "JFK",
                "destination": "LAX",
                "flight_type": "one_way",
                "cabin": "economy",
                "flights": [
                    {"flight_number": "F001", "date": "2024-06-04", "origin": "JFK", "destination": "LAX", "price": 200},
                ],
                "passengers": [
                    {"first_name": "John", "last_name": "Doe", "dob": "1990-01-01"},
                    {"first_name": "Jane", "last_name": "Doe", "dob": "1985-05-15"},
                ],
                "payment_history": [],
                "created_at": "2024-05-01T10:00:00",
                "total_baggages": 2,
                "nonfree_baggages": 0,
                "insurance": "no",
            },
            # Insured + delayed flight → compensation_eligible (regular+insured)
            "RES_INSURED": {
                "reservation_id": "RES_INSURED",
                "user_id": "user_regular",
                "origin": "JFK",
                "destination": "LAX",
                "flight_type": "one_way",
                "cabin": "economy",
                "flights": [
                    {"flight_number": "F001", "date": "2024-06-05", "origin": "JFK", "destination": "LAX", "price": 200},
                ],
                "passengers": [{"first_name": "John", "last_name": "Doe", "dob": "1990-01-01"}],
                "payment_history": [],
                "created_at": "2024-05-01T10:00:00",
                "total_baggages": 1,
                "nonfree_baggages": 0,
                "insurance": "yes",
            },
            # Business cabin → cancellation_eligible True
            "RES_BUSINESS": {
                "reservation_id": "RES_BUSINESS",
                "user_id": "user_regular",
                "origin": "JFK",
                "destination": "LAX",
                "flight_type": "one_way",
                "cabin": "business",
                "flights": [
                    {"flight_number": "F001", "date": "2024-06-01", "origin": "JFK", "destination": "LAX", "price": 500},
                ],
                "passengers": [{"first_name": "John", "last_name": "Doe", "dob": "1990-01-01"}],
                "payment_history": [],
                "created_at": "2024-05-01T10:00:00",
                "total_baggages": 2,
                "nonfree_baggages": 0,
                "insurance": "no",
            },
            # Round-trip
            "RES_ROUND": {
                "reservation_id": "RES_ROUND",
                "user_id": "user_regular",
                "origin": "JFK",
                "destination": "LAX",
                "flight_type": "round_trip",
                "cabin": "economy",
                "flights": [
                    {"flight_number": "F001", "date": "2024-06-01", "origin": "JFK", "destination": "LAX", "price": 200},
                    {"flight_number": "F002", "date": "2024-06-01", "origin": "LAX", "destination": "JFK", "price": 200},
                ],
                "passengers": [{"first_name": "John", "last_name": "Doe", "dob": "1990-01-01"}],
                "payment_history": [],
                "created_at": "2024-05-01T10:00:00",
                "total_baggages": 1,
                "nonfree_baggages": 0,
                "insurance": "no",
            },
            # Silver user + delayed flight → compensation (delayed)
            "RES_SILVER_DELAY": {
                "reservation_id": "RES_SILVER_DELAY",
                "user_id": "user_silver",
                "origin": "JFK",
                "destination": "LAX",
                "flight_type": "one_way",
                "cabin": "economy",
                "flights": [
                    {"flight_number": "F001", "date": "2024-06-05", "origin": "JFK", "destination": "LAX", "price": 200},
                ],
                "passengers": [
                    {"first_name": "Jane", "last_name": "Doe", "dob": "1985-05-15"},
                    {"first_name": "Bob", "last_name": "Smith", "dob": "1975-11-30"},
                ],
                "payment_history": [],
                "created_at": "2024-05-01T10:00:00",
                "total_baggages": 2,
                "nonfree_baggages": 0,
                "insurance": "no",
            },
            # Flight currently in the air — has_flown_flights → True
            "RES_FLYING": {
                "reservation_id": "RES_FLYING",
                "user_id": "user_regular",
                "origin": "JFK",
                "destination": "LAX",
                "flight_type": "one_way",
                "cabin": "economy",
                "flights": [
                    {"flight_number": "F001", "date": "2024-06-03", "origin": "JFK", "destination": "LAX", "price": 200},
                ],
                "passengers": [{"first_name": "John", "last_name": "Doe", "dob": "1990-01-01"}],
                "payment_history": [],
                "created_at": "2024-05-01T10:00:00",
                "total_baggages": 1,
                "nonfree_baggages": 0,
                "insurance": "no",
            },
            # Regular, economy, no insurance, available flight — no disruption
            "RES_CLEAN": {
                "reservation_id": "RES_CLEAN",
                "user_id": "user_clean",
                "origin": "JFK",
                "destination": "LAX",
                "flight_type": "one_way",
                "cabin": "economy",
                "flights": [
                    {"flight_number": "F001", "date": "2024-06-01", "origin": "JFK", "destination": "LAX", "price": 200},
                ],
                "passengers": [{"first_name": "Clean", "last_name": "Slate", "dob": "1995-03-10"}],
                "payment_history": [],
                "created_at": "2024-05-01T10:00:00",
                "total_baggages": 1,
                "nonfree_baggages": 0,
                "insurance": "no",
            },
        },
    )


# ===========================================================================
# Passenger predicates
# ===========================================================================

class TestCountItems:
    def test_list_of_three(self):
        from tau2.domains.airline.security.predicates_v4 import count_items
        assert count_items([1, 2, 3]) == 3

    def test_json_encoded_list(self):
        from tau2.domains.airline.security.predicates_v4 import count_items
        assert count_items(json.dumps(["a", "b"])) == 2

    def test_empty_list(self):
        from tau2.domains.airline.security.predicates_v4 import count_items
        assert count_items([]) == 0

    def test_non_list_returns_zero(self):
        from tau2.domains.airline.security.predicates_v4 import count_items
        assert count_items("not a list") == 0
        assert count_items(42) == 0
        assert count_items(None) == 0


class TestPassInfoComplete:
    def test_complete_passengers(self):
        from tau2.domains.airline.security.predicates_v4 import pass_info_complete
        pax = [{"first_name": "A", "last_name": "B", "dob": "1990-01-01"}]
        assert pass_info_complete(pax) is True

    def test_multiple_complete(self):
        from tau2.domains.airline.security.predicates_v4 import pass_info_complete
        pax = [
            {"first_name": "A", "last_name": "B", "dob": "1990-01-01"},
            {"first_name": "C", "last_name": "D", "dob": "1985-06-15"},
        ]
        assert pass_info_complete(pax) is True

    def test_missing_dob(self):
        from tau2.domains.airline.security.predicates_v4 import pass_info_complete
        pax = [{"first_name": "A", "last_name": "B"}]
        assert pass_info_complete(pax) is False

    def test_missing_last_name(self):
        from tau2.domains.airline.security.predicates_v4 import pass_info_complete
        pax = [{"first_name": "A", "dob": "1990-01-01"}]
        assert pass_info_complete(pax) is False

    def test_empty_list_returns_false(self):
        from tau2.domains.airline.security.predicates_v4 import pass_info_complete
        assert pass_info_complete([]) is False

    def test_json_encoded(self):
        from tau2.domains.airline.security.predicates_v4 import pass_info_complete
        pax = json.dumps([{"first_name": "A", "last_name": "B", "dob": "1990-01-01"}])
        assert pass_info_complete(pax) is True


# ===========================================================================
# Payment predicates
# ===========================================================================

class TestCountCreditCards:
    def test_one_credit_card(self):
        from tau2.domains.airline.security.predicates_v4 import count_credit_cards
        methods = [{"payment_id": "credit_card_001"}]
        assert count_credit_cards(methods) == 1

    def test_two_credit_cards(self):
        from tau2.domains.airline.security.predicates_v4 import count_credit_cards
        methods = [{"payment_id": "credit_card_001"}, {"payment_id": "credit_card_002"}]
        assert count_credit_cards(methods) == 2

    def test_no_credit_cards(self):
        from tau2.domains.airline.security.predicates_v4 import count_credit_cards
        methods = [{"payment_id": "gift_card_001"}, {"payment_id": "certificate_001"}]
        assert count_credit_cards(methods) == 0

    def test_json_encoded(self):
        from tau2.domains.airline.security.predicates_v4 import count_credit_cards
        methods = json.dumps([{"payment_id": "credit_card_999"}])
        assert count_credit_cards(methods) == 1

    def test_raw_string_id(self):
        from tau2.domains.airline.security.predicates_v4 import count_credit_cards
        assert count_credit_cards(["credit_card_x"]) == 1

    def test_empty_list(self):
        from tau2.domains.airline.security.predicates_v4 import count_credit_cards
        assert count_credit_cards([]) == 0


class TestCountGiftCards:
    def test_two_gift_cards(self):
        from tau2.domains.airline.security.predicates_v4 import count_gift_cards
        methods = [{"payment_id": "gift_card_1"}, {"payment_id": "gift_card_2"}, {"payment_id": "credit_card_1"}]
        assert count_gift_cards(methods) == 2

    def test_none(self):
        from tau2.domains.airline.security.predicates_v4 import count_gift_cards
        assert count_gift_cards([{"payment_id": "credit_card_1"}]) == 0


class TestCountCertificates:
    def test_one_certificate(self):
        from tau2.domains.airline.security.predicates_v4 import count_certificates
        assert count_certificates([{"payment_id": "certificate_abc"}]) == 1

    def test_none(self):
        from tau2.domains.airline.security.predicates_v4 import count_certificates
        assert count_certificates([{"payment_id": "credit_card_001"}]) == 0


class TestHasUnknownPayment:
    def test_known_payment(self, db):
        from tau2.domains.airline.security.predicates_v4 import has_unknown_payment
        with patch(f"{MODULE}._get_db", return_value=db):
            methods = [{"payment_id": "credit_card_001"}]
            assert has_unknown_payment(methods, "user_regular") is False

    def test_unknown_payment(self, db):
        from tau2.domains.airline.security.predicates_v4 import has_unknown_payment
        with patch(f"{MODULE}._get_db", return_value=db):
            methods = [{"payment_id": "credit_card_UNKNOWN"}]
            assert has_unknown_payment(methods, "user_regular") is True

    def test_unknown_user(self, db):
        from tau2.domains.airline.security.predicates_v4 import has_unknown_payment
        with patch(f"{MODULE}._get_db", return_value=db):
            methods = [{"payment_id": "credit_card_001"}]
            assert has_unknown_payment(methods, "no_such_user") is True

    def test_empty_methods(self, db):
        from tau2.domains.airline.security.predicates_v4 import has_unknown_payment
        with patch(f"{MODULE}._get_db", return_value=db):
            assert has_unknown_payment([], "user_regular") is False


# ===========================================================================
# Baggage predicates
# ===========================================================================

class TestUserMembershipLevel:
    def test_regular(self, db):
        from tau2.domains.airline.security.predicates_v4 import user_membership_level
        with patch(f"{MODULE}._get_db", return_value=db):
            assert user_membership_level("user_regular") == "regular"

    def test_silver(self, db):
        from tau2.domains.airline.security.predicates_v4 import user_membership_level
        with patch(f"{MODULE}._get_db", return_value=db):
            assert user_membership_level("user_silver") == "silver"

    def test_gold(self, db):
        from tau2.domains.airline.security.predicates_v4 import user_membership_level
        with patch(f"{MODULE}._get_db", return_value=db):
            assert user_membership_level("user_gold") == "gold"

    def test_unknown_user_defaults_to_regular(self, db):
        from tau2.domains.airline.security.predicates_v4 import user_membership_level
        with patch(f"{MODULE}._get_db", return_value=db):
            assert user_membership_level("no_such_user") == "regular"


class TestReservationBaggageCount:
    def test_existing_reservation(self, db):
        from tau2.domains.airline.security.predicates_v4 import reservation_baggage_count
        with patch(f"{MODULE}._get_db", return_value=db):
            assert reservation_baggage_count("RES001") == 4

    def test_missing_reservation(self, db):
        from tau2.domains.airline.security.predicates_v4 import reservation_baggage_count
        with patch(f"{MODULE}._get_db", return_value=db):
            assert reservation_baggage_count("NOPE") == 0


# ===========================================================================
# Reservation lookups
# ===========================================================================

class TestReservationUserId:
    def test_existing(self, db):
        from tau2.domains.airline.security.predicates_v4 import reservation_user_id
        with patch(f"{MODULE}._get_db", return_value=db):
            assert reservation_user_id("RES001") == "user_regular"

    def test_missing(self, db):
        from tau2.domains.airline.security.predicates_v4 import reservation_user_id
        with patch(f"{MODULE}._get_db", return_value=db):
            assert reservation_user_id("NOPE") == ""


class TestReservationCabin:
    def test_economy(self, db):
        from tau2.domains.airline.security.predicates_v4 import reservation_cabin
        with patch(f"{MODULE}._get_db", return_value=db):
            assert reservation_cabin("RES001") == "economy"

    def test_business(self, db):
        from tau2.domains.airline.security.predicates_v4 import reservation_cabin
        with patch(f"{MODULE}._get_db", return_value=db):
            assert reservation_cabin("RES_BUSINESS") == "business"

    def test_missing(self, db):
        from tau2.domains.airline.security.predicates_v4 import reservation_cabin
        with patch(f"{MODULE}._get_db", return_value=db):
            assert reservation_cabin("NOPE") == ""


class TestReservationOrigin:
    def test_existing(self, db):
        from tau2.domains.airline.security.predicates_v4 import reservation_origin
        with patch(f"{MODULE}._get_db", return_value=db):
            assert reservation_origin("RES001") == "JFK"

    def test_missing(self, db):
        from tau2.domains.airline.security.predicates_v4 import reservation_origin
        with patch(f"{MODULE}._get_db", return_value=db):
            assert reservation_origin("NOPE") == ""


class TestReservationDestination:
    def test_existing(self, db):
        from tau2.domains.airline.security.predicates_v4 import reservation_destination
        with patch(f"{MODULE}._get_db", return_value=db):
            assert reservation_destination("RES001") == "LAX"

    def test_missing(self, db):
        from tau2.domains.airline.security.predicates_v4 import reservation_destination
        with patch(f"{MODULE}._get_db", return_value=db):
            assert reservation_destination("NOPE") == ""


class TestReservationTripType:
    def test_one_way(self, db):
        from tau2.domains.airline.security.predicates_v4 import reservation_trip_type
        with patch(f"{MODULE}._get_db", return_value=db):
            assert reservation_trip_type("RES001") == "one_way"

    def test_round_trip(self, db):
        from tau2.domains.airline.security.predicates_v4 import reservation_trip_type
        with patch(f"{MODULE}._get_db", return_value=db):
            assert reservation_trip_type("RES_ROUND") == "round_trip"

    def test_missing(self, db):
        from tau2.domains.airline.security.predicates_v4 import reservation_trip_type
        with patch(f"{MODULE}._get_db", return_value=db):
            assert reservation_trip_type("NOPE") == ""


class TestReservationPassengerCount:
    def test_two_pax(self, db):
        from tau2.domains.airline.security.predicates_v4 import reservation_passenger_count
        with patch(f"{MODULE}._get_db", return_value=db):
            assert reservation_passenger_count("RES001") == 2

    def test_one_pax(self, db):
        from tau2.domains.airline.security.predicates_v4 import reservation_passenger_count
        with patch(f"{MODULE}._get_db", return_value=db):
            assert reservation_passenger_count("RES_FLOWN") == 1

    def test_missing(self, db):
        from tau2.domains.airline.security.predicates_v4 import reservation_passenger_count
        with patch(f"{MODULE}._get_db", return_value=db):
            assert reservation_passenger_count("NOPE") == 0


# ===========================================================================
# Flight predicates
# ===========================================================================

class TestAllFlightsAvailable:
    def test_available_flight(self, db):
        from tau2.domains.airline.security.predicates_v4 import all_flights_available
        flights = [{"flight_number": "F001", "date": "2024-06-01"}]
        with patch(f"{MODULE}._get_db", return_value=db):
            assert all_flights_available(flights) is True

    def test_landed_flight_is_not_available(self, db):
        from tau2.domains.airline.security.predicates_v4 import all_flights_available
        flights = [{"flight_number": "F001", "date": "2024-06-02"}]
        with patch(f"{MODULE}._get_db", return_value=db):
            assert all_flights_available(flights) is False

    def test_cancelled_flight_is_not_available(self, db):
        from tau2.domains.airline.security.predicates_v4 import all_flights_available
        flights = [{"flight_number": "F001", "date": "2024-06-04"}]
        with patch(f"{MODULE}._get_db", return_value=db):
            assert all_flights_available(flights) is False

    def test_delayed_flight_is_not_available(self, db):
        """Policy: delayed flights have not taken off but cannot be booked."""
        from tau2.domains.airline.security.predicates_v4 import all_flights_available
        flights = [{"flight_number": "F001", "date": "2024-06-05"}]
        with patch(f"{MODULE}._get_db", return_value=db):
            assert all_flights_available(flights) is False

    def test_flying_flight_is_not_available(self, db):
        """Policy: flying flights cannot be booked."""
        from tau2.domains.airline.security.predicates_v4 import all_flights_available
        flights = [{"flight_number": "F001", "date": "2024-06-03"}]
        with patch(f"{MODULE}._get_db", return_value=db):
            assert all_flights_available(flights) is False

    def test_json_encoded_flights(self, db):
        from tau2.domains.airline.security.predicates_v4 import all_flights_available
        flights = json.dumps([{"flight_number": "F001", "date": "2024-06-01"}])
        with patch(f"{MODULE}._get_db", return_value=db):
            assert all_flights_available(flights) is True

    def test_unknown_flight_is_treated_as_available(self, db):
        from tau2.domains.airline.security.predicates_v4 import all_flights_available
        # Unknown flights are skipped — no evidence they're unavailable
        flights = [{"flight_number": "UNKNOWN", "date": "2024-06-01"}]
        with patch(f"{MODULE}._get_db", return_value=db):
            assert all_flights_available(flights) is True


class TestHasFlownFlights:
    def test_landed_flight(self, db):
        from tau2.domains.airline.security.predicates_v4 import has_flown_flights
        with patch(f"{MODULE}._get_db", return_value=db):
            assert has_flown_flights("RES_FLOWN") is True

    def test_available_flight_not_flown(self, db):
        from tau2.domains.airline.security.predicates_v4 import has_flown_flights
        with patch(f"{MODULE}._get_db", return_value=db):
            assert has_flown_flights("RES001") is False

    def test_flying_status_counts_as_flown(self, db):
        """Policy: 'flying' (taken off, not yet landed) is treated as flown."""
        from tau2.domains.airline.security.predicates_v4 import has_flown_flights
        with patch(f"{MODULE}._get_db", return_value=db):
            assert has_flown_flights("RES_FLYING") is True

    def test_missing_reservation(self, db):
        from tau2.domains.airline.security.predicates_v4 import has_flown_flights
        with patch(f"{MODULE}._get_db", return_value=db):
            assert has_flown_flights("NOPE") is False


class TestFlightUpdateOrigin:
    def test_first_flight_origin(self, db):
        from tau2.domains.airline.security.predicates_v4 import flight_update_origin
        flights = [{"flight_number": "F001", "date": "2024-06-01"}]
        with patch(f"{MODULE}._get_db", return_value=db):
            assert flight_update_origin(flights) == "JFK"

    def test_round_trip_origin_is_first_flight(self, db):
        from tau2.domains.airline.security.predicates_v4 import flight_update_origin
        flights = [
            {"flight_number": "F001", "date": "2024-06-01"},
            {"flight_number": "F002", "date": "2024-06-01"},
        ]
        with patch(f"{MODULE}._get_db", return_value=db):
            assert flight_update_origin(flights) == "JFK"

    def test_empty_list(self, db):
        from tau2.domains.airline.security.predicates_v4 import flight_update_origin
        with patch(f"{MODULE}._get_db", return_value=db):
            assert flight_update_origin([]) == ""

    def test_unknown_flight(self, db):
        from tau2.domains.airline.security.predicates_v4 import flight_update_origin
        flights = [{"flight_number": "UNKNOWN", "date": "2024-06-01"}]
        with patch(f"{MODULE}._get_db", return_value=db):
            assert flight_update_origin(flights) == ""


class TestFlightUpdateTripType:
    def test_single_flight_is_one_way(self, db):
        from tau2.domains.airline.security.predicates_v4 import flight_update_trip_type
        flights = [{"flight_number": "F001", "date": "2024-06-01"}]
        with patch(f"{MODULE}._get_db", return_value=db):
            assert flight_update_trip_type(flights) == "one_way"

    def test_round_trip_detected(self, db):
        # F001: JFK→LAX, F002: LAX→JFK — first origin == last destination
        from tau2.domains.airline.security.predicates_v4 import flight_update_trip_type
        flights = [
            {"flight_number": "F001", "date": "2024-06-01"},
            {"flight_number": "F002", "date": "2024-06-01"},
        ]
        with patch(f"{MODULE}._get_db", return_value=db):
            assert flight_update_trip_type(flights) == "round_trip"

    def test_two_flights_not_returning_is_one_way(self, db):
        # F001: JFK→LAX, F001: JFK→LAX — doesn't return to start
        from tau2.domains.airline.security.predicates_v4 import flight_update_trip_type
        flights = [
            {"flight_number": "F001", "date": "2024-06-01"},
            {"flight_number": "F001", "date": "2024-06-01"},
        ]
        with patch(f"{MODULE}._get_db", return_value=db):
            assert flight_update_trip_type(flights) == "one_way"

    def test_empty_list(self, db):
        from tau2.domains.airline.security.predicates_v4 import flight_update_trip_type
        with patch(f"{MODULE}._get_db", return_value=db):
            assert flight_update_trip_type([]) == ""


class TestRouteChanged:
    def test_same_route_one_way(self, db):
        from tau2.domains.airline.security.predicates_v4 import route_changed
        # RES001 is one_way JFK→LAX; new flights same route
        flights = [{"flight_number": "F001", "date": "2024-06-01"}]
        with patch(f"{MODULE}._get_db", return_value=db):
            assert route_changed("RES001", flights) is False

    def test_different_origin(self, db):
        from tau2.domains.airline.security.predicates_v4 import route_changed
        # F002 originates at LAX, but RES001 expects JFK origin
        flights = [{"flight_number": "F002", "date": "2024-06-01"}]
        with patch(f"{MODULE}._get_db", return_value=db):
            assert route_changed("RES001", flights) is True

    def test_same_route_round_trip(self, db):
        from tau2.domains.airline.security.predicates_v4 import route_changed
        # RES_ROUND is round_trip JFK→LAX; new flights preserve same structure
        flights = [
            {"flight_number": "F001", "date": "2024-06-01"},
            {"flight_number": "F002", "date": "2024-06-01"},
        ]
        with patch(f"{MODULE}._get_db", return_value=db):
            assert route_changed("RES_ROUND", flights) is False

    def test_round_trip_changed_to_one_way(self, db):
        from tau2.domains.airline.security.predicates_v4 import route_changed
        # RES_ROUND expects round_trip; submitting only one flight
        flights = [{"flight_number": "F001", "date": "2024-06-01"}]
        with patch(f"{MODULE}._get_db", return_value=db):
            assert route_changed("RES_ROUND", flights) is True

    def test_missing_reservation(self, db):
        from tau2.domains.airline.security.predicates_v4 import route_changed
        with patch(f"{MODULE}._get_db", return_value=db):
            assert route_changed("NOPE", [{"flight_number": "F001", "date": "2024-06-01"}]) is False


# ===========================================================================
# Cancellation predicates
# ===========================================================================

class TestCancellationEligible:
    def test_business_class_always_eligible(self, db):
        from tau2.domains.airline.security.predicates_v4 import cancellation_eligible
        with patch(f"{MODULE}._get_db", return_value=db):
            assert cancellation_eligible("RES_BUSINESS") is True

    def test_insured_economy_eligible(self, db):
        from tau2.domains.airline.security.predicates_v4 import cancellation_eligible
        with patch(f"{MODULE}._get_db", return_value=db):
            assert cancellation_eligible("RES_INSURED") is True

    def test_airline_cancelled_flight_eligible(self, db):
        from tau2.domains.airline.security.predicates_v4 import cancellation_eligible
        with patch(f"{MODULE}._get_db", return_value=db):
            assert cancellation_eligible("RES_CANCELLED") is True

    def test_regular_economy_no_insurance_not_eligible(self, db):
        from tau2.domains.airline.security.predicates_v4 import cancellation_eligible
        # RES001: regular, economy, no insurance, active flight
        with patch(f"{MODULE}._get_db", return_value=db):
            assert cancellation_eligible("RES001") is False

    def test_missing_reservation(self, db):
        from tau2.domains.airline.security.predicates_v4 import cancellation_eligible
        with patch(f"{MODULE}._get_db", return_value=db):
            assert cancellation_eligible("NOPE") is False


# ===========================================================================
# Compensation predicates
# ===========================================================================

class TestCompensationEligible:
    def test_silver_member_always_eligible(self, db):
        from tau2.domains.airline.security.predicates_v4 import compensation_eligible
        with patch(f"{MODULE}._get_db", return_value=db):
            assert compensation_eligible("user_silver") is True

    def test_gold_member_always_eligible(self, db):
        from tau2.domains.airline.security.predicates_v4 import compensation_eligible
        with patch(f"{MODULE}._get_db", return_value=db):
            assert compensation_eligible("user_gold") is True

    def test_regular_with_insured_delayed_flight_eligible(self, db):
        # user_regular has RES_INSURED: insured + delayed flight
        from tau2.domains.airline.security.predicates_v4 import compensation_eligible
        with patch(f"{MODULE}._get_db", return_value=db):
            assert compensation_eligible("user_regular") is True

    def test_unknown_user_not_eligible(self, db):
        from tau2.domains.airline.security.predicates_v4 import compensation_eligible
        with patch(f"{MODULE}._get_db", return_value=db):
            assert compensation_eligible("no_such_user") is False

    def test_regular_no_disruption_not_eligible(self, db):
        """Policy: regular member, economy, no insurance, no disrupted flight → not eligible."""
        from tau2.domains.airline.security.predicates_v4 import compensation_eligible
        with patch(f"{MODULE}._get_db", return_value=db):
            assert compensation_eligible("user_clean") is False


class TestValidCertificateAmount:
    def test_valid_delay_amount(self, db):
        # RES_SILVER_DELAY: 2 pax, delayed → valid $50 * 2 = $100
        from tau2.domains.airline.security.predicates_v4 import valid_certificate_amount
        with patch(f"{MODULE}._get_db", return_value=db):
            assert valid_certificate_amount("user_silver", 100) is True

    def test_wrong_amount(self, db):
        from tau2.domains.airline.security.predicates_v4 import valid_certificate_amount
        with patch(f"{MODULE}._get_db", return_value=db):
            assert valid_certificate_amount("user_silver", 75) is False

    def test_cancelled_amount(self, db):
        # user_regular has RES_CANCELLED: 2 pax, airline-cancelled → $100 * 2 = $200
        from tau2.domains.airline.security.predicates_v4 import valid_certificate_amount
        with patch(f"{MODULE}._get_db", return_value=db):
            assert valid_certificate_amount("user_regular", 200) is True

    def test_zero_amount_is_invalid(self, db):
        from tau2.domains.airline.security.predicates_v4 import valid_certificate_amount
        with patch(f"{MODULE}._get_db", return_value=db):
            assert valid_certificate_amount("user_regular", 0) is False

    def test_non_numeric_is_invalid(self, db):
        from tau2.domains.airline.security.predicates_v4 import valid_certificate_amount
        with patch(f"{MODULE}._get_db", return_value=db):
            assert valid_certificate_amount("user_regular", "abc") is False

    def test_cancelled_rate_invalid_for_delay_only_reservation(self, db):
        """Policy: $100/pax only for cancelled; $50/pax only for delayed.
        user_silver has only a delayed reservation (2 pax) → valid = {100}.
        The cancelled rate $200 (100×2) must be rejected.
        """
        from tau2.domains.airline.security.predicates_v4 import valid_certificate_amount
        with patch(f"{MODULE}._get_db", return_value=db):
            assert valid_certificate_amount("user_silver", 200) is False

    def test_no_disrupted_flights_accepts_any_amount(self, db):
        """Edge case: when a user has no disrupted flights, valid set is empty
        and the predicate returns True for any positive amount.
        This is the documented fail-open behaviour (predicate note: TODO check).
        """
        from tau2.domains.airline.security.predicates_v4 import valid_certificate_amount
        with patch(f"{MODULE}._get_db", return_value=db):
            # user_gold has no reservations at all
            assert valid_certificate_amount("user_gold", 999) is True


# ===========================================================================
# all_flights_same_cabin_class — documents the broken predicate
# ===========================================================================

class TestAllFlightsSameCabinClass:
    def test_empty_list_returns_true(self, db):
        from tau2.domains.airline.security.predicates_v4 import all_flights_same_cabin_class
        with patch(f"{MODULE}._get_db", return_value=db):
            assert all_flights_same_cabin_class([]) is True

    def test_flights_without_cabin_field_returns_true(self, db):
        """Normal case: flight dicts have no cabin field — vacuously consistent."""
        from tau2.domains.airline.security.predicates_v4 import all_flights_same_cabin_class
        flights = [
            {"flight_number": "F001", "date": "2024-06-01"},
            {"flight_number": "F002", "date": "2024-06-01"},
        ]
        with patch(f"{MODULE}._get_db", return_value=db):
            assert all_flights_same_cabin_class(flights) is True

    def test_flights_with_same_cabin_returns_true(self, db):
        from tau2.domains.airline.security.predicates_v4 import all_flights_same_cabin_class
        flights = [
            {"flight_number": "F001", "date": "2024-06-01", "cabin": "economy"},
            {"flight_number": "F002", "date": "2024-06-01", "cabin": "economy"},
        ]
        with patch(f"{MODULE}._get_db", return_value=db):
            assert all_flights_same_cabin_class(flights) is True

    def test_flights_with_mixed_cabins_returns_false(self, db):
        from tau2.domains.airline.security.predicates_v4 import all_flights_same_cabin_class
        flights = [
            {"flight_number": "F001", "date": "2024-06-01", "cabin": "economy"},
            {"flight_number": "F002", "date": "2024-06-01", "cabin": "business"},
        ]
        with patch(f"{MODULE}._get_db", return_value=db):
            assert all_flights_same_cabin_class(flights) is False


# ===========================================================================
# flight_update_destination
# ===========================================================================

class TestFlightUpdateDestination:
    def test_one_way_returns_last_flight_destination(self, db):
        # F001: JFK→LAX
        from tau2.domains.airline.security.predicates_v4 import flight_update_destination
        flights = [{"flight_number": "F001", "date": "2024-06-01"}]
        with patch(f"{MODULE}._get_db", return_value=db):
            assert flight_update_destination(flights) == "LAX"

    def test_round_trip_returns_outbound_destination(self, db):
        # F001: JFK→LAX, F002: LAX→JFK — round trip, booking destination is LAX
        from tau2.domains.airline.security.predicates_v4 import flight_update_destination
        flights = [
            {"flight_number": "F001", "date": "2024-06-01"},
            {"flight_number": "F002", "date": "2024-06-01"},
        ]
        with patch(f"{MODULE}._get_db", return_value=db):
            assert flight_update_destination(flights) == "LAX"

    def test_empty_list(self, db):
        from tau2.domains.airline.security.predicates_v4 import flight_update_destination
        with patch(f"{MODULE}._get_db", return_value=db):
            assert flight_update_destination([]) == ""

    def test_unknown_flight(self, db):
        from tau2.domains.airline.security.predicates_v4 import flight_update_destination
        flights = [{"flight_number": "UNKNOWN", "date": "2024-06-01"}]
        with patch(f"{MODULE}._get_db", return_value=db):
            assert flight_update_destination(flights) == ""

    def test_consistent_with_reservation_destination_one_way(self, db):
        """flight_update_destination must equal reservation_destination for a valid update."""
        from tau2.domains.airline.security.predicates_v4 import (
            flight_update_destination,
            reservation_destination,
        )
        flights = [{"flight_number": "F001", "date": "2024-06-01"}]
        with patch(f"{MODULE}._get_db", return_value=db):
            assert flight_update_destination(flights) == reservation_destination("RES001")

    def test_consistent_with_reservation_destination_round_trip(self, db):
        """For a round-trip update, flight_update_destination must equal reservation_destination."""
        from tau2.domains.airline.security.predicates_v4 import (
            flight_update_destination,
            reservation_destination,
        )
        flights = [
            {"flight_number": "F001", "date": "2024-06-01"},
            {"flight_number": "F002", "date": "2024-06-01"},
        ]
        with patch(f"{MODULE}._get_db", return_value=db):
            assert flight_update_destination(flights) == reservation_destination("RES_ROUND")
