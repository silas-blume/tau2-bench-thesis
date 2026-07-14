
## Domain Basic

Before taking any actions that update the booking database (booking, modifying flights, editing baggage, changing cabin class, or updating passenger information), you must list the action details and obtain explicit user confirmation (yes) to proceed.





You should not provide any information, knowledge, or procedures not provided by the user or available tools, or give subjective recommendations or comments.




You should only make one tool call at a time, and if you make a tool call, you should not respond to the user simultaneously. If you respond to the user, you should not make a tool call at the same time.




You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions.



## Book flight

The agent must first obtain the user id from the user.





The agent should then ask for the trip type, origin, destination.




Each reservation can have at most five passengers. 




The agent needs to collect the first name, last name, and date of birth for each passenger. 



All passengers must fly the same flights in the same cabin.

NN: covered by api, cannot be verified


Each reservation can use at most one travel certificate, at most one credit card, and at most three gift cards. 

  - name: book-too-many-credit-cards
    template: Absence
    first:
      event: book_reservation(user_id, origin, destination, flight_type, cabin, flights, passengers, payment_methods, total_baggages, nonfree_baggages, insurance)
      where:
        - count_credit_cards(payment_methods) > 1
    description: "Booking can use at most 1 credit card."

  - name: book-too-many-gift-cards
    template: Absence
    first:
      event: book_reservation(user_id, origin, destination, flight_type, cabin, flights, passengers, payment_methods, total_baggages, nonfree_baggages, insurance)
      where:
        - count_gift_cards(payment_methods) > 3
    description: "Booking can use at most 3 gift cards."

  - name: book-too-many-certificates
    template: Absence
    first:
      event: book_reservation(user_id, origin, destination, flight_type, cabin, flights, passengers, payment_methods, total_baggages, nonfree_baggages, insurance)
      where:
        - count_certificates(payment_methods) > 1
    description: "Booking can use at most 1 travel certificate."

DI


The remaining amount of a travel certificate is not refundable. 

IN

no direct refund action & certificate has no origin to check


All payment methods must already be in user profile for safety reasons.

  - name: book-payment-in-profile
    template: Absence
    first:
      event: book_reservation(user_id, origin, destination, flight_type, cabin, flights, passengers, payment_methods, total_baggages, nonfree_baggages, insurance)
      where:
        - has_unknown_payment(payment_methods, user_id)
    description: "All payment methods must be in the user profile beforehand."

DI

----

Checked bag allowance: 
- If the booking user is a regular member:
  - 0 free checked bag for each basic economy passenger
  - 1 free checked bag for each economy passenger
  - 2 free checked bags for each business passenger
- If the booking user is a silver member:
  - 1 free checked bag for each basic economy passenger
  - 2 free checked bag for each economy passenger
  - 3 free checked bags for each business passenger
- If the booking user is a gold member:
  - 2 free checked bag for each basic economy passenger
  - 3 free checked bag for each economy passenger
  - 4 free checked bags for each business passenger

see file

DI
  
Each extra baggage is 50 dollars.

NN: automatically enforced by api

automatically calculated in api

The agent should ask if the user wants to buy the travel insurance.

IN: User messages not accessable + NL


The travel insurance is 30 dollars per passenger and enables full refund if the user needs to cancel the flight given health or weather reasons.

NN: automatically calculated in api

## Modify flight

First, the agent must obtain the user id and reservation id. 
- The user must provide their user id. 

  - name: user-before-flight-update
    template: Precedence
    first: get_user_details(user_id)
    second: update_reservation_flights(reservation_id, cabin, flights, payment_id)
    where:
      - user_id == reservation_user_id(reservation_id)
    description: "Before changing flights, verify user identity first."

  - name: user-before-baggage-update
    template: Precedence
    first: get_user_details(user_id)
    second: update_reservation_baggages(reservation_id, total_baggages, nonfree_baggages, payment_id)
    where:
      - user_id == reservation_user_id(reservation_id)
    description: "Before changing baggage, verify user identity first."

  - name: user-before-passenger-update
    template: Precedence
    first: get_user_details(user_id)
    second: update_reservation_passengers(reservation_id, passengers)
    where:
      - user_id == reservation_user_id(reservation_id)
    description: "Before changing passengers, verify user identity first."

WO: By enforcing user detail lookup, uid needs to be retrieved

- If the user doesn't know their reservation id, the agent should help locate it using available tools.


  - name: reservation-before-flight-update
    template: Precedence
    first: get_reservation_details(rid)
    second: update_reservation_flights(reservation_id, cabin, flights, payment_id)
    where:
      - rid == reservation_id
    description: "Before changing flights, inspect reservation details first."

  - name: reservation-before-baggage-update
    template: Precedence
    first: get_reservation_details(rid)
    second: update_reservation_baggages(reservation_id, total_baggages, nonfree_baggages, payment_id)
    where:
      - rid == reservation_id
    description: "Before changing baggage, inspect reservation details first."

  - name: reservation-before-passenger-update
    template: Precedence
    first: get_reservation_details(rid)
    second: update_reservation_passengers(reservation_id, passengers)
    where:
      - rid == reservation_id
    description: "Before changing passengers, inspect reservation details first."

WO: By enforcing user detail lookup, rid needs to be retrieved

Basic economy flights cannot be modified.

  - name: no-basic-economy-flight-modification
    template: Absence
    first:
      event: update_reservation_flights(reservation_id, cabin, flights, payment_id)
      where:
        - reservation_cabin(reservation_id) == "basic_economy"
    description: "Cannot modify flights on a reservation that is currently in basic economy cabin."

DI

Other reservations can be modified without changing the origin, destination, and trip type.

  - name: no-change-origin
    template: Absence
    first:
      event: update_reservation_flights(reservation_id, cabin, flights, payment_id)
      where:
        - reservation_origin(reservation_id) != flight_update_origin(flights)
    description: "Flight modifications cannot change the origin airport."

  - name: no-change-destination
    template: Absence
    first:
      event: update_reservation_flights(reservation_id, cabin, flights, payment_id)
      where:
        - reservation_destination(reservation_id) != flight_update_destination(flights)
    description: "Flight modifications cannot change the destination airport."

  - name: no-change-type
    template: Absence
    first:
      event: update_reservation_flights(reservation_id, cabin, flights, payment_id)
      where:
        - reservation_trip_type(reservation_id) != flight_update_trip_type(flights)
    description: "Flight modifications cannot change the trip type (one-way vs round-trip)."

DI

Some flight segments can be kept, but their prices will not be updated based on the current price.

???

Cabin cannot be changed if any flight in the reservation has already been flown.

  - name: cabin-change-no-flown-flights
    template: Absence
    first:
      event: update_reservation_flights(reservation_id, cabin, flights, payment_id)
      where:
        - has_flown_flights(reservation_id)
    description: "Cannot change cabin if any flight has already been flown."

  DI


In other cases, all reservations, including basic economy, can change cabin without changing the flights.

  - name: flight-update-preserves-route
    template: Absence
    first:
      event: update_reservation_flights(reservation_id, cabin, flights, payment_id)
      where:
        - route_changed(reservation_id, flights)
    description: "Flight modification must not change origin, destination, or trip type."

  DI

Cabin class must remain the same across all the flights in the same reservation; changing cabin for just one flight segment is not possible.

  - name: all-flights-same-cabin-class
    template: Absence
    first:
      event: update_reservation_flights(reservation_id, cabin, flights, payment_id)
      where:
        - not all_flights_same_cabin_class(flights) # TODO implement predicate
    description: "All flights in a reservation must have the same cabin class."

  DI

If the price after cabin change is higher than the original price, the user is required to pay for the difference.

  NN: handled in API

If the price after cabin change is lower than the original price, the user is should be refunded the difference.

  NN: handled in API

The user can add but not remove checked bags.

  - name: baggage-only-add
    template: Absence
    first:
      event: update_reservation_baggages(reservation_id, total_baggages, nonfree_baggages, payment_id)
      where:
        - total_baggages < reservation_baggage_count(reservation_id)
    description: "Baggage can only be added, not removed."

DI

The user cannot add insurance after initial booking.

NN: no tool for that

The user can modify passengers but cannot modify the number of passengers.

  - name: passenger-count-unchanged
    template: Absence
    first:
      event: update_reservation_passengers(reservation_id, passengers)
      where:
        - count_items(passengers) != reservation_passenger_count(reservation_id)
    description: "Cannot modify the number of passengers in a reservation."

    DI

If the flights are changed, the user needs to provide a single gift card or credit card for payment or refund method. The payment method must already be in user profile for safety reasons.

???

## Cancel flight

First, the agent must obtain the user id and reservation id. 
- The user must provide their user id. 

  - name: user-before-cancellation
    template: Precedence
    first: get_user_details(user_id)
    second: cancel_reservation(reservation_id)
    where:
      - user_id == reservation_user_id(reservation_id)
    description: "Before cancellation, verify user identity first."

WO: By enforcing user detail lookup, uid needs to be retrieved

- If the user doesn't know their reservation id, the agent should help locate it using available tools.

  - name: reservation-before-cancellation
    template: Precedence
    first: get_reservation_details(rid)
    second: cancel_reservation(reservation_id)
    where:
      - rid == reservation_id
    description: "Before cancellation, inspect reservation details first."

  WO: By enforcing user detail lookup, rid needs to be retrieved


The agent must also obtain the reason for cancellation (change of plan, airline cancelled flight, or other reasons)

NN: enforced by api

If any portion of the flight has already been flown, the agent cannot help and transfer is needed.

  - name: cancel-no-flown-flights
    template: Absence
    first:
      event: cancel_reservation(reservation_id, reason)
      where:
        - has_flown_flights(reservation_id)
    description: "Cannot cancel a reservation if any flight has already been flown. Transfer to human agent for special handling."

  DI

Otherwise, flight can be cancelled if any of the following is true:

WO: Disjunction not expressable in YAML, only in predicates

- The booking was made within the last 24 hrs
- The flight is cancelled by airline
- It is a business flight
- The user has travel insurance and the reason for cancellation is covered by insurance (health or weather reasons).

  - name: cancel-not-eligible
    template: Absence
    first:
      event: cancel_reservation(reservation_id, reason)
      where:
        - not cancellation_eligible(reservation_id)
    description: "Cannot cancel unless business class, airline-cancelled flight, or insured. NOTE: the 24h booking window must be verified by the agent."

The refund will go to original payment methods within 5 to 7 business days.

NN: implemented in API

## Refunds and Compensation

Do not proactively offer a compensation unless the user explicitly asks for one.

  IN: no semantic nl eval and no message access

Do not compensate if the user is regular member and has no travel insurance and flies (basic) economy.
Only compensate if the user is a silver/gold member or has travel insurance or flies business.

WO: Disjunction not expressable in YAML, only in predicates

 - name: compensation-eligibility
    template: Absence
    first:
      event: send_certificate(user_id, amount)
      where:
        - not compensation_eligible(user_id)
    description: "Compensation only for silver/gold members, insured passengers, or business class. Do not compensate if the user is regular member and has no travel insurance and flies (basic) economy."

If the user complains about cancelled flights in a reservation, the agent can offer a certificate as a gesture after confirming the facts, with the amount being $100 times the number of passengers.
If the user complains about delayed flights in a reservation and wants to change or cancel the reservation, the agent can offer a certificate as a gesture after confirming the facts and changing or cancelling the reservation, with the amount being $50 times the number of passengers.
Do not offer compensation for any other reason than the ones listed above.

WO: Disjunction not expressable in YAML, only in predicates -> no separation between reasons

 - name: certificate-amount-valid
    template: Absence
    first:
      event: send_certificate(user_id, amount)
      where:
        - not valid_certificate_amount(user_id, amount)
    description: "Certificate amount must match policy ($100/pax cancelled, $50/pax delayed)."