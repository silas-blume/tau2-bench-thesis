# Airline Policy — Declare YAML Translation Analysis

Classification of each natural-language policy rule against Declare YAML expressiveness.

**Legend**
| Code | Meaning |
|------|---------|
| **DI** | Direct — rule maps cleanly onto a Declare template + predicate |
| **WO** | Workaround — rule can be approximated but requires an indirect encoding that does not fully capture the intent |
| **IN** | Inexpressible — the constraint cannot be encoded in Declare YAML (typically because it involves NL content, message inspection, or open-world reasoning) |
| **NN** | Not Needed — constraint is automatically enforced by the API / tool layer; no YAML rule is required |

---

## 1. Domain Basic

### R01 — Confirmation Before Database-Modifying Actions

**NL-Rule:** Before taking any action that updates the booking database (booking, modifying flights, editing baggage, changing cabin class, or updating passenger information), the agent must list the action details and obtain explicit user confirmation (`yes`) to proceed.

**Classification:** IN

**Comment:** Requires reading and evaluating the agent's outgoing message to the user (to check that details were listed) and then verifying the content of the user's reply (to check for explicit confirmation). Neither outgoing agent messages nor incoming user messages are accessible to the Declare monitor. Confirmation is a conversational interaction pattern that cannot be expressed as an event constraint.

---

### R02 — Restrict to Known Information and Tools

**NL-Rule:** The agent must not provide any information, knowledge, or procedures not provided by the user or available tools. It must not give subjective recommendations or comments.

**Classification:** IN

**Comment:** Evaluating whether a response contains information that originates only from the user or tool outputs requires semantic NL understanding of agent messages. The Declare monitor has no access to message content. The distinction between factual/grounded and subjective/hallucinated output is an NL-level judgment.

---

### R03 — One Tool Call at a Time / No Simultaneous Tool Call and Response

**NL-Rule:** The agent must only make one tool call at a time. If it makes a tool call, it must not respond to the user simultaneously. If it responds to the user, it must not make a tool call at the same time.

**Classification:** IN

**Comment:** This rule governs the internal concurrency of agent execution (parallel tool invocations, simultaneous message emission). The Declare event log records completed tool events sequentially and has no notion of simultaneity or parallel dispatch within a single agent step.

---

### R04 — Human Agent Transfer Protocol

**NL-Rule:** The agent must transfer the user to a human agent if and only if the request cannot be handled within the scope of its actions. It must try everything before transferring, first explicitly refuse, and only transfer if the customer insists.

**Classification:** IN

**Comment:** Determining whether a request is "within scope" requires open-world reasoning about agent capabilities relative to the request. Verifying that the agent exhausted all options before transferring, and that the customer insisted, requires reading and interpreting NL conversation content — both inaccessible to the monitor.

---

## 2. Book Flight

### R05 — Obtain User ID Before Booking

**NL-Rule:** The agent must first obtain the user id from the user before booking.

**Classification:** WO

**Comment:** A `Precedence` constraint enforcing `get_user_details` before `book_reservation` (with `user_id == uid`) approximates the intent. The workaround works because retrieving user details is a necessary precondition — by encoding it as a mandatory preceding event, the uid is implicitly obtained. The original rule says the id must come *from the user*, which is not verifiable; only the lookup call can be enforced.

**YAML:**
```yaml
- name: user-before-booking
  template: Precedence
  first: get_user_details(uid)
  second: book_reservation(user_id, ...)
  where:
    - user_id == uid
```

---

### R06 — Ask for Trip Type, Origin, Destination

**NL-Rule:** The agent should ask the user for the trip type, origin, and destination.

**Classification:** IN

**Comment:** Requires verifying that the agent sent a message asking for specific fields. This is an NL conversation step that cannot be expressed as a tool-event constraint.

---

### R07 — At Most Five Passengers Per Booking

**NL-Rule:** Each reservation can have at most five passengers.

**Classification:** DI

**Comment:** Directly encoded as an `Absence` constraint on `book_reservation` with `count_items(passengers) > 5`.

**YAML:**
```yaml
- name: book-max-five-passengers
  template: Absence
  first:
    event: book_reservation(...)
    where:
      - count_items(passengers) > 5
```

---

### R08 — Collect Full Passenger Information

**NL-Rule:** The agent must collect the first name, last name, and date of birth for each passenger.

**Classification:** DI

**Comment:** Directly encoded as an `Absence` constraint checking `pass_info_complete(passengers)` on `book_reservation`. The predicate verifies that all required fields are present in the payload.

**YAML:**
```yaml
- name: all-passengers-data
  template: Absence
  first:
    event: book_reservation(...)
    where:
      - not pass_info_complete(passengers)
```

---

### R09 — All Passengers on Same Flights / Same Cabin

**NL-Rule:** All passengers must fly the same flights in the same cabin.

**Classification:** NN

**Comment:** The API structure enforces a single `cabin` and single `flights` list per reservation; per-passenger flight/cabin divergence is not a valid API call. No Declare rule needed.

---

### R10 — Payment Method Limits Per Booking

**NL-Rule:** Each reservation can use at most one travel certificate, at most one credit card, and at most three gift cards.

**Classification:** DI

**Comment:** Three separate `Absence` constraints, each with a count predicate on `payment_methods`, cover the three limits independently and directly.

**YAML:**
```yaml
- name: book-too-many-credit-cards
  template: Absence
  first:
    event: book_reservation(...)
    where:
      - count_credit_cards(payment_methods) > 1

- name: book-too-many-gift-cards
  template: Absence
  first:
    event: book_reservation(...)
    where:
      - count_gift_cards(payment_methods) > 3

- name: book-too-many-certificates
  template: Absence
  first:
    event: book_reservation(...)
    where:
      - count_certificates(payment_methods) > 1
```

---

### R11 — Travel Certificate Remainder Not Refundable

**NL-Rule:** The remaining amount of a travel certificate is not refundable.

**Classification:** IN

**Comment:** There is no dedicated refund action for certificate remainders in the tool set. The origin and remaining balance of a certificate cannot be traced in the event log, so the constraint cannot be expressed or checked.

---

### R12 — Payment Methods Must Be in User Profile

**NL-Rule:** All payment methods must already be in the user profile.

**Classification:** DI

**Comment:** Directly encoded via `Absence` with predicate `has_unknown_payment(payment_methods, user_id)`, which cross-references the booking payload against the user profile retrieved earlier in the trace.

**YAML:**
```yaml
- name: book-payment-in-profile
  template: Absence
  first:
    event: book_reservation(...)
    where:
      - has_unknown_payment(payment_methods, user_id)
```

---

### R13 — Checked Bag Allowance (Membership × Cabin)

**NL-Rule:** Free checked bag allowance depends on the booking user's membership level and the cabin class (Regular: 0/1/2; Silver: 1/2/3; Gold: 2/3/4 for basic economy/economy/business). Each extra bag costs $50.

**Classification:** DI

**Comment:** Encoded as an `Absence` constraint on `book_reservation` with a predicate that evaluates `nonfree_baggages` against the membership-cabin lookup table. The $50 per extra bag pricing is validated in the same predicate (or confirmed as NN — see R14).

---

### R14 — Extra Baggage Price ($50 Each)

**NL-Rule:** Each extra baggage is $50.

**Classification:** NN

**Comment:** Pricing is automatically calculated and enforced by the booking API. No agent-level constraint needed.

---

### R15 — Ask About Travel Insurance

**NL-Rule:** The agent must ask if the user wants to buy travel insurance during booking.

**Classification:** IN

**Comment:** Requires verifying that a specific NL question was posed by the agent during the conversation. Not expressible as a tool-event constraint.

---

### R16 — Travel Insurance Terms ($30/pax, Health/Weather Refund)

**NL-Rule:** The travel insurance is $30 per passenger and enables full refund for cancellations due to health or weather reasons.

**Classification:** NN

**Comment:** Pricing and coverage logic are handled entirely within the booking API. No agent-enforced rule needed.

---

## 3. Modify Flight

### R17 — Obtain User ID Before Modifications

**NL-Rule:** The user must provide their user id before any flight, baggage, or passenger modification.

**Classification:** WO

**Comment:** Three `Precedence` constraints enforce `get_user_details` before each of `update_reservation_flights`, `update_reservation_baggages`, and `update_reservation_passengers`. The workaround encodes that the lookup must precede the action, but does not verify the id came from the user rather than the agent's own state.

**YAML:**
```yaml
- name: user-before-flight-update
  template: Precedence
  first: get_user_details(user_id)
  second: update_reservation_flights(reservation_id, ...)
  where:
    - user_id == reservation_user_id(reservation_id)

- name: user-before-baggage-update
  template: Precedence
  first: get_user_details(user_id)
  second: update_reservation_baggages(reservation_id, ...)
  where:
    - user_id == reservation_user_id(reservation_id)

- name: user-before-passenger-update
  template: Precedence
  first: get_user_details(user_id)
  second: update_reservation_passengers(reservation_id, ...)
  where:
    - user_id == reservation_user_id(reservation_id)
```

---

### R18 — Help Locate Reservation ID

**NL-Rule:** If the user doesn't know their reservation id, the agent should help locate it using available tools.

**Classification:** WO

**Comment:** Three `Precedence` constraints enforce `get_reservation_details(rid)` before each modification action with `rid == reservation_id`. This ensures the reservation was looked up before modification, indirectly forcing the id to be resolved. The "help locate if unknown" framing is a conversational obligation that cannot be directly encoded; the workaround only enforces the lookup, not the dialogic assistance intent.

**YAML:**
```yaml
- name: reservation-before-flight-update
  template: Precedence
  first: get_reservation_details(rid)
  second: update_reservation_flights(reservation_id, ...)
  where:
    - rid == reservation_id
```
*(analogous constraints for baggage and passenger updates)*

---

### R19 — Basic Economy Flights Cannot Be Modified (Flight Change)

**NL-Rule:** Basic economy flights cannot have their flights changed. (Cabin changes are still allowed — see R22/R23.)

**Classification:** DI

**Comment:** Directly encoded as `Absence` on `update_reservation_flights` with `reservation_cabin(reservation_id) == "basic_economy"`.

**YAML:**
```yaml
- name: no-basic-economy-flight-modification
  template: Absence
  first:
    event: update_reservation_flights(reservation_id, ...)
    where:
      - reservation_cabin(reservation_id) == "basic_economy"
```

---

### R20 — No Change of Origin, Destination, or Trip Type

**NL-Rule:** Reservations can be modified without changing the origin, destination, or trip type.

**Classification:** DI

**Comment:** Three `Absence` constraints, one each for origin, destination, and trip-type mismatch, directly express this restriction.

**YAML:**
```yaml
- name: no-change-origin
  template: Absence
  first:
    event: update_reservation_flights(...)
    where:
      - reservation_origin(reservation_id) != flight_update_origin(flights)

- name: no-change-destination / no-change-type  # analogous
```

---

### R21 — Kept Segments Have Frozen Prices

**NL-Rule:** Some flight segments can be kept, but their prices will not be updated based on the current price.

**Classification:** NN

**Comment:** This is an API-internal pricing behavior (kept segments retain their original price). There is no agent action to enforce; the constraint is structural to the API response.

---

### R22 — No Cabin Change if Any Flight Already Flown

**NL-Rule:** Cabin cannot be changed if any flight in the reservation has already been flown.

**Classification:** DI

**Comment:** Directly encoded as `Absence` on `update_reservation_flights` with `has_flown_flights(reservation_id)`.

**YAML:**
```yaml
- name: cabin-change-no-flown-flights
  template: Absence
  first:
    event: update_reservation_flights(...)
    where:
      - has_flown_flights(reservation_id)
```

---

### R23 — All Reservations May Change Cabin Without Changing Flights

**NL-Rule:** All reservations, including basic economy, can change cabin class without changing the actual flight segments.

**Classification:** DI

**Comment:** Encoded as `Absence` on `update_reservation_flights` when `route_changed(reservation_id, flights)` is true, preventing flight-segment changes while allowing cabin-only updates.

**YAML:**
```yaml
- name: flight-update-preserves-route
  template: Absence
  first:
    event: update_reservation_flights(...)
    where:
      - route_changed(reservation_id, flights)
```

---

### R24 — Cabin Class Same Across All Flights in Reservation

**NL-Rule:** Cabin class must remain the same across all flights in the same reservation; changing cabin for just one segment is not allowed.

**Classification:** DI

**Comment:** Encoded as `Absence` with `not all_flights_same_cabin_class(flights)`.

**YAML:**
```yaml
- name: all-flights-same-cabin-class
  template: Absence
  first:
    event: update_reservation_flights(...)
    where:
      - not all_flights_same_cabin_class(flights)
```

---

### R25 — Pay Difference if New Price Is Higher

**NL-Rule:** If the price after a cabin change is higher than the original, the user must pay the difference.

**Classification:** NN

**Comment:** Price-difference charging is handled automatically by the API. No agent-enforced rule needed.

---

### R26 — Refund Difference if New Price Is Lower

**NL-Rule:** If the price after a cabin change is lower than the original, the user must be refunded the difference.

**Classification:** NN

**Comment:** Price-difference refunds are handled automatically by the API. No agent-enforced rule needed.

---

### R27 — Baggage: Add Only, No Removal

**NL-Rule:** The user can add but not remove checked bags.

**Classification:** DI

**Comment:** Directly encoded as `Absence` on `update_reservation_baggages` with `total_baggages < reservation_baggage_count(reservation_id)`.

**YAML:**
```yaml
- name: baggage-only-add
  template: Absence
  first:
    event: update_reservation_baggages(...)
    where:
      - total_baggages < reservation_baggage_count(reservation_id)
```

---

### R28 — Cannot Add Insurance After Booking

**NL-Rule:** The user cannot add insurance after the initial booking.

**Classification:** NN

**Comment:** There is no `add_insurance` tool in the agent's action set. The constraint is structurally enforced by the absence of a corresponding API endpoint.

---

### R29 — Modify Passengers but Not Passenger Count

**NL-Rule:** The user can modify passenger details but cannot change the number of passengers.

**Classification:** DI

**Comment:** Directly encoded as `Absence` on `update_reservation_passengers` with `count_items(passengers) != reservation_passenger_count(reservation_id)`.

**YAML:**
```yaml
- name: passenger-count-unchanged
  template: Absence
  first:
    event: update_reservation_passengers(...)
    where:
      - count_items(passengers) != reservation_passenger_count(reservation_id)
```

---

### R30 — Single Gift Card or Credit Card for Payment When Flights Changed

**NL-Rule:** If flights are changed, the user must provide exactly one gift card or credit card for payment or refund. The payment method must already be in the user profile.

**Classification:** WO

**Comment:** Partially expressible: the profile membership check can reuse the `has_unknown_payment` predicate pattern. The "exactly one gift card or credit card" constraint requires a disjunctive count predicate (`count_gift_cards == 1 OR count_credit_cards == 1`, with the total being exactly 1), which cannot be expressed natively in YAML — a custom predicate is needed to encode the disjunction. This is therefore a workaround rather than a direct encoding.

---

## 4. Cancel Flight

### R31 — Obtain User ID Before Cancellation

**NL-Rule:** The user must provide their user id before cancellation.

**Classification:** WO

**Comment:** `Precedence` constraint enforcing `get_user_details` before `cancel_reservation` with `user_id == reservation_user_id(reservation_id)`. Same workaround pattern as R05 and R17.

**YAML:**
```yaml
- name: user-before-cancellation
  template: Precedence
  first: get_user_details(user_id)
  second: cancel_reservation(reservation_id)
  where:
    - user_id == reservation_user_id(reservation_id)
```

---

### R32 — Help Locate Reservation ID for Cancellation

**NL-Rule:** If the user doesn't know their reservation id, the agent should help locate it.

**Classification:** WO

**Comment:** `Precedence` constraint enforcing `get_reservation_details(rid)` before `cancel_reservation`. Same workaround pattern as R18.

**YAML:**
```yaml
- name: reservation-before-cancellation
  template: Precedence
  first: get_reservation_details(rid)
  second: cancel_reservation(reservation_id)
  where:
    - rid == reservation_id
```

---

### R33 — Obtain Cancellation Reason

**NL-Rule:** The agent must obtain the reason for cancellation (change of plan, airline cancelled, or other reasons).

**Classification:** NN

**Comment:** The `cancel_reservation` API requires a `reason` parameter; the tool call itself enforces that a reason is provided.

---

### R34 — No Cancellation if Any Flight Already Flown

**NL-Rule:** If any portion of the flight has already been flown, the agent cannot cancel and must transfer to a human agent.

**Classification:** DI

**Comment:** Directly encoded as `Absence` on `cancel_reservation` with `has_flown_flights(reservation_id)`.

**YAML:**
```yaml
- name: cancel-no-flown-flights
  template: Absence
  first:
    event: cancel_reservation(reservation_id, reason)
    where:
      - has_flown_flights(reservation_id)
```

---

### R35 — Cancellation Eligibility (Disjunctive Conditions)

**NL-Rule:** Cancellation is allowed only if at least one of the following holds: (a) booking was made within the last 24 hours; (b) the airline cancelled the flight; (c) it is a business class flight; (d) the user has travel insurance and the cancellation reason is covered (health or weather).

**Classification:** WO

**Comment:** The four conditions form a disjunction that cannot be expressed natively in YAML (which supports only conjunctive `where` predicates). The workaround is to encapsulate the disjunction inside a single custom predicate `cancellation_eligible(reservation_id, reason)` and encode an `Absence` on its negation. The predicate must also handle the 24-hour window check, which requires a timestamp comparison. Classification is WO because the constraint is encodable only by offloading all logic to a black-box predicate rather than composing YAML constructs.

**YAML:**
```yaml
- name: cancel-not-eligible
  template: Absence
  first:
    event: cancel_reservation(reservation_id, reason)
    where:
      - not cancellation_eligible(reservation_id, reason)
```

---

### R36 — Refund to Original Payment Methods (5–7 Business Days)

**NL-Rule:** The refund goes to the original payment methods within 5 to 7 business days.

**Classification:** NN

**Comment:** Refund routing and timeline are handled by the API backend. No agent-enforced rule needed.

---

## 5. Refunds and Compensation

### R37 — Do Not Proactively Offer Compensation

**NL-Rule:** The agent must not proactively offer compensation unless the user explicitly asks for one.

**Classification:** IN

**Comment:** Requires semantic evaluation of whether the agent's message constitutes a proactive offer versus a response to a user request. This requires both message-content access and NL intent classification — neither available in the Declare monitor.

---

### R38 — Compensation Eligibility Gate

**NL-Rule:** Do not compensate if the user is a regular member with no travel insurance flying basic economy or economy. Only compensate if the user is a silver/gold member, has travel insurance, or flies business class.

**Classification:** WO

**Comment:** The eligibility condition is a disjunction (silver/gold member OR insured OR business) that cannot be expressed natively in YAML. Encoded as `Absence` on `send_certificate` with a custom predicate `compensation_eligible(user_id)` that encapsulates the disjunction.

**YAML:**
```yaml
- name: compensation-eligibility
  template: Absence
  first:
    event: send_certificate(user_id, amount)
    where:
      - not compensation_eligible(user_id)
```

---

### R39 — Certificate Amounts ($100 Cancelled / $50 Delayed / No Other Reasons)

**NL-Rule:** For cancelled flights: offer a certificate of $100 × number of passengers. For delayed flights (when the user changes/cancels): offer $50 × number of passengers. No compensation for any other reason.

**Classification:** WO

**Comment:** The rule fuses three sub-constraints: (a) the amount formula for cancelled flights, (b) the amount formula for delayed flights, (c) prohibition of compensation for other reasons. The separation by reason requires a disjunctive condition (reason = cancelled → amount = 100×pax; reason = delayed → amount = 50×pax; else → prohibited). This cannot be expressed as separate YAML branches; all logic must be pushed into a single predicate `valid_certificate_amount(user_id, amount)`, making this a workaround. The reason-type distinction (cancelled vs. delayed) is further lost in a flat predicate.

**YAML:**
```yaml
- name: certificate-amount-valid
  template: Absence
  first:
    event: send_certificate(user_id, amount)
    where:
      - not valid_certificate_amount(user_id, amount)
```

---

## 6. Summary

### 6.1 Classification Counts

| Classification | Count | Rules |
|----------------|-------|-------|
| **DI** — Direct | 13 | R07, R08, R10, R12, R13, R19, R20, R22, R23, R24, R27, R29, R34 |
| **WO** — Workaround | 9 | R05, R17, R18, R30, R31, R32, R35, R38, R39 |
| **IN** — Inexpressible | 8 | R01, R02, R03, R04, R06, R11, R15, R37 |
| **NN** — Not Needed | 9 | R09, R14, R16, R21, R25, R26, R28, R33, R36 |
| **Total** | 39 | |

### 6.2 Classification Distribution by Section

| Section | DI | WO | IN | NN | Total |
|---------|----|----|----|----|-------|
| Domain Basic | 0 | 0 | 4 | 0 | 4 |
| Book Flight | 5 | 1 | 3 | 3 | 12 |
| Modify Flight | 7 | 2 | 0 | 5 | 14 |
| Cancel Flight | 1 | 3 | 0 | 2 | 6 |
| Compensation | 0 | 3 | 1 | 0 | 4 (not counting R21) |

> *Note: R21 (frozen prices) is counted under NN and assigned to Modify Flight.*

### 6.3 Pattern Analysis

#### Direct (DI) — What Works Well

All DI rules share a common structure: a discrete API action carries a parameter payload that fully encodes the relevant state. Constraints on that payload (count checks, equality checks, flag checks) map directly onto `Absence` templates with predicate conditions. The Declare framework is well-suited to **parameter-level constraints on individual tool calls**.

- Count-based limits (passengers, payment methods, baggage): straightforward cardinality predicates.
- Precedence-based identity verification: `Precedence` template with binding constraint.
- State-derived restrictions (already-flown, basic-economy cabin, route preservation): predicates that query derived state from prior events.

#### Workaround (WO) — Indirect Encoding Required

All WO rules share one of two underlying causes:

1. **Disjunctive eligibility conditions** (R35, R38, R39, R30): YAML `where` clauses are conjunctive only. Any rule with `OR` branching must be collapsed into a single opaque predicate, losing the ability to independently verify each branch and preventing structured fault attribution.

2. **Identity/provenance verification via lookup enforcement** (R05, R17, R18, R31, R32): Rules that require the agent to *ask the user* for an id are approximated by enforcing that a *lookup call* must have occurred. This conflates "the id was retrieved" with "the id was obtained from the user" — a subtle but meaningful loss of intent.

#### Inexpressible (IN) — Structural Limits

All IN rules involve one or more of:

- **Outgoing message content** (R01, R02, R37): verifying what the agent said, or that the agent asked something specific.
- **Incoming user message content** (R01, R03, R06, R15): verifying what the user said or that they confirmed/answered something.
- **Open-world reasoning about scope or knowledge** (R02, R04): determining whether information is grounded, whether a request is within scope, or whether a recommendation is subjective.
- **Concurrency / simultaneity** (R03): the event log is sequential; parallelism within an agent step is invisible.

These represent a hard expressiveness boundary: Declare YAML can only observe *which tool was called with which arguments*, not *what was said* in the surrounding conversation.

#### Not Needed (NN) — API Enforcement

NN rules cover pricing mechanics (R14, R25, R26), structural API constraints (R09, R28, R33), and policy side-effects automatically applied by the backend (R16, R21, R36). These are not limitations of Declare expressiveness; they reflect appropriate separation of concerns between agent-level and service-level enforcement.

### 6.4 Expressiveness Boundary

The Declare YAML format is expressive for **event-argument constraints** on a fixed action vocabulary. Its boundary is reached when a rule requires:

| Requirement | Expressible? |
|-------------|-------------|
| Count / cardinality check on tool argument | Yes (DI) |
| Ordering / precedence between tool calls | Yes (DI/WO) |
| Derived-state predicate on prior events | Yes (DI) |
| Disjunctive condition in `where` clause | No — requires custom predicate (WO) |
| Reading agent's outgoing NL message | No (IN) |
| Reading user's incoming NL message | No (IN) |
| Reasoning about scope, intent, or subjectivity | No (IN) |
| Concurrent / simultaneous events | No (IN) |
