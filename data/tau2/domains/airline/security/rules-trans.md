
## Domain Basic

Before taking any actions that update the booking database (booking, modifying flights, editing baggage, changing cabin class, or updating passenger information), you must list the action details and obtain explicit user confirmation (yes) to proceed.

IN

User messages not accessable + NL +  obtaining confirmation not expressable



You should not provide any information, knowledge, or procedures not provided by the user or available tools, or give subjective recommendations or comments.

IN

User messages not accessable + NL evaluation not possible


You should only make one tool call at a time, and if you make a tool call, you should not respond to the user simultaneously. If you respond to the user, you should not make a tool call at the same time.

IN

User messages not accessable + NL evaluation not possible


You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions.

IN

Scope not verifyable from NL

## Book flight

The agent must first obtain the user id from the user.

  - name: user-before-booking
    template: Precedence
    first: get_user_details(uid)
    second: book_reservation(user_id, origin, destination, flight_type, cabin, flights, passengers, payment_methods, total_baggages, nonfree_baggages, insurance)
    where:
      - user_id == uid
    description: "Before booking, retrieve user details first."

WO

By enforcing user detail lookup, uid needs to be retrieved


The agent should then ask for the trip type, origin, destination.

IN

User messages not accessable + NL


Each reservation can have at most five passengers. 

  - name: book-max-five-passengers
    template: Absence
    first:
      event: book_reservation(user_id, origin, destination, flight_type, cabin, flights, passengers, payment_methods, total_baggages, nonfree_baggages, insurance)
      where:
        - count_items(passengers) > 5
    description: "Each reservation can have at most five passengers."

DI


The agent needs to collect the first name, last name, and date of birth for each passenger. 

- name: all-passengers-data
    template: Absence
    first:
      event: book_reservation(user_id, origin, destination, flight_type, cabin, flights, passengers, payment_methods, total_baggages, nonfree_baggages, insurance)
      where:
        - pass_info_complete(passengers)
    description: "For each passenger first name, second name and date of birth must be provided."

DI

All passengers must fly the same flights in the same cabin.

NN

covered by api, cannot be verified


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

NN

automatically calculated in api

The agent should ask if the user wants to buy the travel insurance.

IN

User messages not accessable + NL


The travel insurance is 30 dollars per passenger and enables full refund if the user needs to cancel the flight given health or weather reasons.

NN

automatically calculated in api

## Modify flight

First, the agent must obtain the user id and reservation id. 
- The user must provide their user id. 
- If the user doesn't know their reservation id, the agent should help locate it using available tools.

Basic economy flights cannot be modified.

Other reservations can be modified without changing the origin, destination, and trip type.

Some flight segments can be kept, but their prices will not be updated based on the current price.


Cabin cannot be changed if any flight in the reservation has already been flown.

In other cases, all reservations, including basic economy, can change cabin without changing the flights.

Cabin class must remain the same across all the flights in the same reservation; changing cabin for just one flight segment is not possible.

If the price after cabin change is higher than the original price, the user is required to pay for the difference.

If the price after cabin change is lower than the original price, the user is should be refunded the difference.

The user can add but not remove checked bags.

The user cannot add insurance after initial booking.

The user can modify passengers but cannot modify the number of passengers.

If the flights are changed, the user needs to provide a single gift card or credit card for payment or refund method. The payment method must already be in user profile for safety reasons.

## Cancel flight

First, the agent must obtain the user id and reservation id. 
- The user must provide their user id. 
- If the user doesn't know their reservation id, the agent should help locate it using available tools.

The agent must also obtain the reason for cancellation (change of plan, airline cancelled flight, or other reasons)

If any portion of the flight has already been flown, the agent cannot help and transfer is needed.

Otherwise, flight can be cancelled if any of the following is true:
- The booking was made within the last 24 hrs
- The flight is cancelled by airline
- It is a business flight
- The user has travel insurance and the reason for cancellation is covered by insurance (health or weather reasons).

The refund will go to original payment methods within 5 to 7 business days.

## Refunds and Compensation

Do not proactively offer a compensation unless the user explicitly asks for one.

Do not compensate if the user is regular member and has no travel insurance and flies (basic) economy.

Only compensate if the user is a silver/gold member or has travel insurance or flies business.

If the user complains about cancelled flights in a reservation, the agent can offer a certificate as a gesture after confirming the facts, with the amount being $100 times the number of passengers.

If the user complains about delayed flights in a reservation and wants to change or cancel the reservation, the agent can offer a certificate as a gesture after confirming the facts and changing or cancelling the reservation, with the amount being $50 times the number of passengers.

Do not offer compensation for any other reason than the ones listed above.