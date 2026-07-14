# Airline Policy → Data-Aware DCR (dcr2.yaml) — Rule-by-Rule Draft

Analogue of `rules-trans.md` (which does this for the Declare-YAML policy),
but against `dcr2.yaml`, the data-aware DCR-YAML policy. Goes through
`policy.md` top to bottom; each NL paragraph is followed by the DCR-YAML
construct(s) that encode it (event + condition + guarded exclude/include),
a classification tag, and a short comment. Where no construct existed in
`dcr2.yaml` before this pass, it is marked **ADDED** and the new
construct is shown.

**Legend:** `DI` direct · `WO` workaround · `IN` inexpressible · `NN` not needed
(same codes as `rules-trans.md`/`rules-translation-analysis.md`, for
comparability). See `rules-translation-dcr-analysis.md` for the polished,
numbered version of this pass with full classification rationale and the
Declare-vs-DCR comparison.

---

## Domain Basic

Before taking any actions that update the booking database (booking, modifying flights, editing baggage, changing cabin class, or updating passenger information), you must list the action details and obtain explicit user confirmation (yes) to proceed.

IN

Same limitation as Declare: requires reading the agent's outgoing message (were details listed?) and the user's reply (was there an explicit "yes"?). DCR's graph only sees which tool executed with which arguments, never message content — identical blindness to Declare here, no difference in expressiveness.

---

You should not provide any information, knowledge, or procedures not provided by the user or available tools, or give subjective recommendations or comments.

IN

Same as Declare — NL semantic judgment, no message access.

---

You should only make one tool call at a time, and if you make a tool call, you should not respond to the user simultaneously. If you respond to the user, you should not make a tool call at the same time.

IN

Same as Declare — concurrency/simultaneity is invisible to a sequential event log regardless of formalism.

---

You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions.

IN (trigger condition) — but the **consequence** is DI and already implemented:

```yaml
# dcr2.yaml
excludes:
  transfer_to_human_agents:
    - book_reservation
    - update_reservation_flights
    - update_reservation_baggages
    - update_reservation_passengers
    - cancel_reservation
    - send_certificate
```

The "iff" trigger condition (is this request truly out of scope?) is open-world reasoning, same IN limitation as Declare. But *once* transfer happens, DCR's `excludes` relation lets the *consequence* — "nothing else is allowed after this" — be expressed directly and structurally as a first-class relation. Declare's Absence/Precedence/Response vocabulary has no equally direct primitive for "this event permanently disables that one"; it would need something like a NotSuccession template layered on top, if the grammar even supports it. This is a genuine DCR expressiveness edge that isn't really about *this* rule's IN core, but about a structural pattern DCR happens to make cheap.

---

## Book flight

The agent must first obtain the user id from the user.

```yaml
# dcr2.yaml
conditions:
  get_user_details:
    - book_reservation
```

WO

Same workaround as Declare's Precedence: enforces that the lookup *happened*, not that the id came *from the user*. DCR's unguarded `conditions` relation and Declare's `Precedence` template express the identical semantics here (source must have executed before target is enabled) — no difference.

---

The agent should then ask for the trip type, origin, destination.

IN

Same as Declare — NL conversational step.

---

Each reservation can have at most five passengers.

```yaml
# dcr2.yaml
booking_num_passengers:
  type: int
  description: "Provide the number of passengers in the booking request (policy limit: <= 5)."
booking_params_valid:
  type: bool
  expr: "[booking_num_passengers] <= 5 and [booking_num_credit_cards] <= 1 and [booking_num_gift_cards] <= 3 and [booking_num_certificates] <= 1"
```

DI

Already present. Folded into the single combined `booking_params_valid` gate together with the payment-method-count checks below (§4.5 of `DCR_DATA_ARCHITECTURE.md` explains why a *single* combined gate, not four separate ones, is required).

---

The agent needs to collect the first name, last name, and date of birth for each passenger.

**ADDED** — was missing from `dcr2.yaml` entirely (confirmed via `tools.py`: `book_reservation` never validates passenger field completeness itself):

```yaml
# dcr2.yaml
booking_passengers_info_complete:
  type: bool
  description: "Provide whether every passenger has first name, last name, and date of birth."
book_reservation_valid:
  type: bool
  expr: "[booking_params_valid] and [booking_payment_methods_valid] and [booking_passengers_info_complete] and [booking_baggage_allowance_valid]"
```
Resolver: `dcr2_data_resolver._booking_passengers_info_complete` → `dcr2_predicates.pass_info_complete`.

DI

Direct, resolver-hook based, same mechanism as every other input event already in `dcr2.yaml`.

---

All passengers must fly the same flights in the same cabin.

NN

`book_reservation`'s API signature takes one `cabin` and one `flights` list for the whole reservation — per-passenger divergence isn't a constructible call. Same as Declare's classification.

---

Each reservation can use at most one travel certificate, at most one credit card, and at most three gift cards.

```yaml
# dcr2.yaml
booking_num_credit_cards:
  type: int
booking_num_gift_cards:
  type: int
booking_num_certificates:
  type: int
booking_params_valid:
  expr: "[booking_num_passengers] <= 5 and [booking_num_credit_cards] <= 1 and [booking_num_gift_cards] <= 3 and [booking_num_certificates] <= 1"
```

DI

Already present, same combined-gate pattern.

---

The remaining amount of a travel certificate is not refundable.

IN

Same as Declare — no refund action, no certificate-origin tracking in the event log.

---

All payment methods must already be in user profile for safety reasons.

```yaml
# dcr2.yaml
booking_payment_methods_valid:
  type: bool
  description: "Provide whether all payment methods in the booking are in the user profile."
```
Resolver: `dcr2_data_resolver._booking_payment_methods_valid` → `dcr2_predicates.has_unknown_payment`.

DI

Already present.

---

Checked bag allowance: [membership × cabin table] Each extra baggage is 50 dollars.

**ADDED** (the allowance table was missing; the $50 pricing itself is NN, see below) — this is dcr2's first genuine use of `FunctionCallExpression`/`predicate_registry` rather than the resolver hook:

```yaml
# dcr2.yaml
booking_membership_code:
  type: int
  description: "regular=0, silver=1, gold=2"
booking_cabin_code:
  type: int
  description: "basic_economy=0, economy=1, business=2"
booking_total_baggages:
  type: int
booking_nonfree_baggages:
  type: int
booking_baggage_allowance_valid:
  type: bool
  expr: "baggage_allowance_ok([booking_membership_code], [booking_cabin_code], [booking_num_passengers], [booking_total_baggages], [booking_nonfree_baggages])"
```
```python
# dcr2_expr_predicates.py — registered via predicate_file_path=, NOT the resolver hook
def baggage_allowance_ok(membership_code, cabin_code, num_passengers,
                          total_baggages, nonfree_baggages) -> bool:
    free_per_passenger = membership_code + cabin_code
    free_total = free_per_passenger * num_passengers
    return nonfree_baggages >= max(0, total_baggages - free_total)
```

DI

This is the case flagged in `DCR_DATA_ARCHITECTURE.md` §6 as `FunctionCallExpression`'s actual niche: a 3×3 lookup table times passenger count is a pure function of already-resolved int event values, more naturally written as real Python than nested `if/then/else`. `booking_membership_code`/`booking_cabin_code`/`booking_total_baggages`/`booking_nonfree_baggages` still need the resolver hook to get the raw facts (user membership, cabin argument, baggage counts) into the graph as event values in the first place — `FunctionCallExpression` only composes over what's already there. Both mechanisms genuinely cooperate here.

---

Each extra baggage is 50 dollars.

NN

Pricing calculated automatically in `AirlineTools.book_reservation` (`total_price += 50 * nonfree_baggages`). Same as Declare.

---

The agent should ask if the user wants to buy the travel insurance.

IN

Same as Declare.

---

The travel insurance is 30 dollars per passenger and enables full refund if the user needs to cancel the flight given health or weather reasons.

NN

Pricing calculated automatically. (The "enables cancellation" half is captured separately under Cancel flight, via `reservation_cancellation_eligible_base` checking `insurance == "yes"`.)

---

## Modify flight

First, the agent must obtain the user id and reservation id.

```yaml
# dcr2.yaml
conditions:
  get_user_details:
    - update_reservation_flights
    - update_reservation_baggages
    - update_reservation_passengers
  get_reservation_details:
    - update_reservation_flights
    - update_reservation_baggages
    - update_reservation_passengers
```

WO

Same as Declare's paired Precedence constraints — enforces the lookups happened, not that the ids came from the user.

---

Change flights: Basic economy flights cannot be modified.

**REFINED this session** — was present but over-restrictive (blocked *any* `update_reservation_flights` call on a basic-economy reservation, including a pure cabin upgrade with unchanged flights, contradicting the very next rule below):

```yaml
# dcr2.yaml
reservation_flights_changed:
  type: bool
  description: "Provide whether the submitted flight segments differ from the reservation's current ones."
update_reservation_flights_valid:
  expr: "(not ([reservation_is_basic_economy] and [reservation_flights_changed])) and ..."
```
Resolver: `dcr2_data_resolver._reservation_flights_changed` → `dcr2_predicates.flights_changed` (order-independent flight_number+date set comparison against the reservation's current flights).

DI

Now correctly scoped: blocks only when *both* basic-economy *and* the flight segments actually changed, matching the next rule's explicit cabin-only exception.

---

Other reservations can be modified without changing the origin, destination, and trip type. ... In other cases, all reservations, including basic economy, can change cabin without changing the flights.

**ADDED** — origin/destination/trip-type preservation was missing entirely (`update_reservation_flights_valid` previously had no route check at all):

```yaml
# dcr2.yaml
reservation_route_changed:
  type: bool
  description: "Provide whether the submitted flights would change the reservation's origin, destination, or trip type."
update_reservation_flights_valid:
  expr: "... and (not [reservation_route_changed]) and ..."
```
Resolver: `dcr2_data_resolver._reservation_route_changed` → `dcr2_predicates.route_changed` (ported from `predicates_v4.route_changed`'s round-trip-aware origin/destination logic, reused near-verbatim for consistency with the Declare policy's semantics).

DI

Together with the refined basic-economy check above, this is exactly the pair Declare encoded as three separate Absence constraints (`no-change-origin`/`no-change-destination`/`no-change-type`) plus a fourth (`flight-update-preserves-route`); DCR's version folds all four into one resolver-computed bool, same combined-gate reasoning as everywhere else in `dcr2.yaml`.

---

Some flight segments can be kept, but their prices will not be updated based on the current price.

NN

API-internal pricing behavior, same as Declare.

---

Cabin cannot be changed if any flight in the reservation has already been flown.

```yaml
# dcr2.yaml
reservation_has_flown:
  type: bool
update_reservation_flights_valid:
  expr: "... and (not [reservation_has_flown]) and ..."
```

DI

Already present.

---

Cabin class must remain the same across all the flights in the same reservation; changing cabin for just one flight segment is not possible.

NN

`update_reservation_flights(reservation_id, cabin, flights, payment_id)` takes a single `cabin` for the whole call — no per-flight cabin field exists to diverge. (Declare's version encoded a technically-present but structurally-vacuous predicate for this; DCR skips encoding a check that can never fire rather than carry dead code — see the analysis doc for the classification-refinement note.)

---

If the price after cabin change is higher/lower than the original price, the user is required to pay/be refunded the difference.

NN

API-internal, same as Declare.

---

The user can add but not remove checked bags.

```yaml
# dcr2.yaml
bags_add_only_valid:
  type: bool
  expr: "[new_total_baggages] >= [reservation_current_bag_count]"
```

DI

Already present.

---

The user cannot add insurance after initial booking.

NN

No `add_insurance`/`update_insurance` tool exists. Same as Declare.

---

The user can modify passengers but cannot modify the number of passengers.

```yaml
# dcr2.yaml
passenger_count_unchanged_valid:
  type: bool
  expr: "[update_num_passengers] == [reservation_num_passengers]"
```

DI

Already present.

---

If the flights are changed, the user needs to provide a single gift card or credit card for payment or refund method. The payment method must already be in user profile for safety reasons.

`update_reservation_flights(reservation_id, cabin, flights, payment_id)` takes a single `payment_id: str`, not a list — so "a single" is **NN** (structurally guaranteed) where Declare needed a WO (a custom disjunctive-count predicate, since its `payment_methods` argument is a list). "Must be in profile" was already **DI** (`update_payment_in_profile`). "Must be gift card or credit card, not a certificate" was **ADDED**:

```yaml
# dcr2.yaml
update_payment_method_type_valid:
  type: bool
  description: "Provide whether the flight-update payment_id is a gift card or credit card (not a travel certificate)."
update_reservation_flights_valid:
  expr: "... and [update_payment_in_profile] and [update_payment_method_type_valid]"
```
Resolver: `dcr2_data_resolver._update_payment_method_type_valid` → `dcr2_predicates.payment_method_type_ok` (checks the `payment_id` string prefix).

DI

The single-string API shape structurally eliminates the cardinality problem that made this WO in Declare; the type-exclusion is a plain resolver-hook bool.

---

## Cancel flight

First, the agent must obtain the user id and reservation id.

```yaml
# dcr2.yaml
conditions:
  get_user_details:
    - cancel_reservation
  get_reservation_details:
    - cancel_reservation
```

WO

Same as Declare.

---

If any portion of the flight has already been flown, the agent cannot help and transfer is needed.

```yaml
# dcr2.yaml
reservation_has_flown:
  type: bool
cancel_reservation_valid:
  expr: "(not [reservation_has_flown]) and [reservation_cancellation_eligible]"
```

DI

Already present.

---

Otherwise, flight can be cancelled if any of the following is true: booking within 24h / airline-cancelled / business class / insured with covered reason.

```yaml
# dcr2.yaml
reservation_cancellation_eligible_base:
  type: bool
  description: "business class, airline-cancelled flight, or travel insurance with covered reason"
booking_within_24h:
  type: bool
reservation_cancellation_eligible:
  type: bool
  expr: "[reservation_cancellation_eligible_base] or [booking_within_24h]"
```

WO (partial improvement over Declare)

The *outer* disjunction (`base OR 24h`) is now **directly visible** as a native `or` in the expr string — Declare had to hide the entire four-way disjunction inside one opaque `cancellation_eligible()` predicate with zero partial structure. But the *inner* three-way disjunction (business / insured / airline-cancelled) still has to be resolved by `dcr2_predicates.cancellation_eligible_base`, an opaque Python function — because those facts (cabin, insurance flag, per-flight cancellation status) require DB lookups that can't be expressed in the grammar regardless of how expressive it is. So this stays WO overall, but with a real, demonstrable finer-grained diagnostic: a DECLINE can now say specifically "`reservation_cancellation_eligible_base` failed AND `booking_within_24h` failed" as two distinguishable inputs, rather than Declare's single all-or-nothing `cancellation_eligible` verdict.

Also note: neither this dcr2 predicate nor Declare's `cancellation_eligible_base`/`cancellation_eligible` actually checks that the *cancellation reason argument* matches "health or weather" — both just check `insurance == "yes"` unconditionally. This is a pre-existing shared simplification, not a DCR-specific gap, and out of scope to fix here (the tool's `reason` field is free text, not a matchable enum).

---

The refund will go to original payment methods within 5 to 7 business days.

NN

API-internal, same as Declare.

---

## Refunds and Compensation

Do not proactively offer a compensation unless the user explicitly asks for one.

IN

Same as Declare.

---

Do not compensate if the user is regular member and has no travel insurance and flies (basic) economy. Only compensate if the user is a silver/gold member or has travel insurance or flies business.

```yaml
# dcr2.yaml
user_compensation_eligible:
  type: bool
  description: "silver/gold member, travel insurance holder, or business-class passenger"
send_certificate_valid:
  expr: "[user_compensation_eligible] and [certificate_amount_valid]"
```

WO

Same as Declare — the full disjunction requires cross-reservation DB traversal (checking every reservation of the user for a disrupted, insured/business flight), which can't be expressed in the grammar regardless of native `or` support; entirely opaque to a single resolver-hook bool, same as Declare's single opaque predicate.

---

If the user complains about cancelled flights ... $100 × pax. If ... delayed flights ... $50 × pax. Do not offer compensation for any other reason.

```yaml
# dcr2.yaml
certificate_amount_valid:
  type: bool
  description: "$100 x passengers for any cancelled-flight reservation, or $50 x passengers for any delayed-flight reservation"
send_certificate_valid:
  expr: "[user_compensation_eligible] and [certificate_amount_valid]"
```

WO

Same as Declare — `send_certificate(user_id, amount)` carries no `reservation_id`, so the resolver must search across *all* of the user's reservations for one that justifies the submitted amount; the reason-type separation (cancelled vs. delayed) is collapsed into a single valid-amounts set, same loss of structure as Declare's `valid_certificate_amount`.
