# Airline Policy — Data-Aware DCR (dcr2.yaml) Translation Analysis

Analogue of `rules-translation-analysis.md` (the Declare-YAML version), but
against the data-aware DCR-YAML policy (`dcr2.yaml`). Same 39 rules, same
numbering (R01–R39), so the two documents are directly comparable rule for
rule. Draft-form working notes are in `rules-trans-dcr.md`.

This pass found **5 rules with no equivalent in `dcr2.yaml`** (R08, R13, R20,
R23, and the certificate-exclusion half of R30) and **1 rule that was present
but incorrectly over-restrictive** (R19 — see below). All were added/fixed in
`dcr2.yaml`, `dcr2_data_resolver.py`, and `dcr2_predicates.py` as part of this
pass, plus a new file `dcr2_expr_predicates.py` for R13 (dcr2's first use of
the `FunctionCallExpression`/`predicate_registry` mechanism — previously
unused; see `DCR_DATA_ARCHITECTURE.md` §2.6/§6). Live verification (3 test
tasks) is in §7.

**Legend**
| Code | Meaning |
|------|---------|
| **DI** | Direct — rule maps cleanly onto a DCR event/relation |
| **WO** | Workaround — rule can be approximated but requires an indirect encoding that does not fully capture the intent |
| **IN** | Inexpressible — the constraint cannot be encoded in DCR-YAML (typically because it involves NL content, message inspection, or open-world reasoning) |
| **NN** | Not Needed — constraint is automatically enforced by the API / tool layer, or structurally impossible to violate given the tool's argument shape; no DCR rule is required |

---

## 1. Domain Basic

### R01 — Confirmation Before Database-Modifying Actions

**NL-Rule:** Before taking any action that updates the booking database, the agent must list the action details and obtain explicit user confirmation (`yes`) to proceed.

**Classification:** IN

**Comment:** Identical limitation to Declare: DCR's event log records only which tool executed with which arguments. Neither the agent's outgoing messages nor the user's incoming replies are visible to `DCRStateTracker`, so "were details listed" and "did the user say yes" cannot be checked regardless of formalism.

---

### R02 — Restrict to Known Information and Tools

**NL-Rule:** The agent must not provide information not grounded in the user or tool outputs, or give subjective recommendations.

**Classification:** IN

**Comment:** Requires semantic evaluation of message content — not available to either formalism.

---

### R03 — One Tool Call at a Time / No Simultaneous Tool Call and Response

**NL-Rule:** Only one tool call at a time; no simultaneous message + tool call.

**Classification:** IN

**Comment:** DCR's marking evolves by discrete event execution; there is no representation of intra-step concurrency in either formalism.

---

### R04 — Human Agent Transfer Protocol

**NL-Rule:** Transfer to a human agent if and only if the request cannot be handled within scope; try everything first, refuse explicitly, only transfer if the customer insists.

**Classification:** IN (trigger condition) — **DI** for the structural consequence

**Comment:** The "iff out of scope" trigger remains open-world reasoning, IN exactly like Declare. But `dcr2.yaml` directly and structurally encodes the *consequence* of a transfer — "once it happens, every write action becomes permanently unavailable" — via an unguarded `excludes` relation:
```yaml
excludes:
  transfer_to_human_agents:
    - book_reservation
    - update_reservation_flights
    - update_reservation_baggages
    - update_reservation_passengers
    - cancel_reservation
    - send_certificate
```
This is a genuine DCR-vs-Declare expressiveness difference worth flagging even though it doesn't change R04's own IN classification: DCR's `excludes` relation is a first-class primitive for "this event permanently disables that one," something Declare's Absence/Precedence/Response vocabulary has no equally direct equivalent for.

---

## 2. Book Flight

### R05 — Obtain User ID Before Booking

**NL-Rule:** The agent must first obtain the user id from the user before booking.

**Classification:** WO

**Comment:** An unguarded `conditions` edge enforcing `get_user_details` before `book_reservation` — structurally identical semantics to Declare's `Precedence` template (source must execute before target is enabled). Same limitation: enforces the lookup happened, not that the id came *from the user*.

**DCR-YAML:**
```yaml
conditions:
  get_user_details:
    - book_reservation
```

---

### R06 — Ask for Trip Type, Origin, Destination

**Classification:** IN — same as Declare (NL conversational step).

---

### R07 — At Most Five Passengers Per Booking

**NL-Rule:** Each reservation can have at most five passengers.

**Classification:** DI

**Comment:** Folded into the single combined `booking_params_valid` gate alongside the payment-method-count checks (R10) — one computed event, one condition, one guarded-exclude/include pair, per the "combined gate" pattern (`DCR_DATA_ARCHITECTURE.md` §4.5): independent guard pairs sourced from different events would let whichever resolves last silently overwrite the others' effect on `included`.

**DCR-YAML:**
```yaml
booking_num_passengers:
  type: int
booking_params_valid:
  type: bool
  expr: "[booking_num_passengers] <= 5 and [booking_num_credit_cards] <= 1 and [booking_num_gift_cards] <= 3 and [booking_num_certificates] <= 1"
```

---

### R08 — Collect Full Passenger Information

**NL-Rule:** The agent must collect first name, last name, and date of birth for each passenger.

**Classification:** DI — **added this session** (was missing; confirmed via `tools.py` that `book_reservation` never validates this itself).

**Comment:** New input event resolved via the resolver hook, folded into `book_reservation_valid`.

**DCR-YAML:**
```yaml
booking_passengers_info_complete:
  type: bool
book_reservation_valid:
  expr: "[booking_params_valid] and [booking_payment_methods_valid] and [booking_passengers_info_complete] and [booking_baggage_allowance_valid]"
```
```python
# dcr2_data_resolver.py
def _booking_passengers_info_complete(event):
    return _p.pass_info_complete(event.get("passengers"))
# dcr2_predicates.py
def pass_info_complete(passengers):
    for p in _parse_json_if_str(passengers):
        if not (p.get("first_name") and p.get("last_name") and p.get("dob")):
            return False
    return True
```

---

### R09 — All Passengers on Same Flights / Same Cabin

**Classification:** NN — same as Declare. `book_reservation` takes one `cabin`/`flights` for the whole reservation; no per-passenger divergence is constructible.

---

### R10 — Payment Method Limits Per Booking

**NL-Rule:** At most one certificate, one credit card, three gift cards.

**Classification:** DI — already present, same combined gate as R07.

---

### R11 — Travel Certificate Remainder Not Refundable

**Classification:** IN — same as Declare. No refund action, no certificate-origin tracking in the graph.

---

### R12 — Payment Methods Must Be in User Profile

**Classification:** DI — already present.

**DCR-YAML:**
```yaml
booking_payment_methods_valid:
  type: bool
```
Resolver: `_booking_payment_methods_valid` → `has_unknown_payment(payment_methods, user)`.

---

### R13 — Checked Bag Allowance (Membership × Cabin)

**NL-Rule:** Free checked-bag allowance depends on membership level × cabin class (Regular 0/1/2, Silver 1/2/3, Gold 2/3/4 for basic_economy/economy/business); extra bags $50 each (see R14).

**Classification:** DI — **added this session** (was missing entirely; confirmed via `tools.py` that `book_reservation` only *prices* `nonfree_baggages`, never validates it against an allowance).

**Comment:** dcr2's first genuine use of `FunctionCallExpression`/`predicate_registry`, not the resolver hook. The allowance formula (`membership_code + cabin_code`, times passenger count, compared against `total_baggages - nonfree_baggages`) is a pure function of already-resolved int event values — exactly the case flagged as `FunctionCallExpression`'s actual niche in `DCR_DATA_ARCHITECTURE.md` §6: too awkward to nest as nine `if/then/else` branches in an expr string, natural as real Python. The four inputs (`booking_membership_code`, `booking_cabin_code`, `booking_total_baggages`, `booking_nonfree_baggages`) still need the resolver hook to get the raw facts into the graph as event values first — `FunctionCallExpression` only composes over values already there, it can't reach `FlightDB`/tool-call arguments itself (verified by reading `FunctionCallExpression.evaluate`, `DCR_DATA_ARCHITECTURE.md` §6). Both mechanisms are exercised together, deliberately.

**DCR-YAML:**
```yaml
booking_membership_code:
  type: int   # regular=0, silver=1, gold=2
booking_cabin_code:
  type: int   # basic_economy=0, economy=1, business=2
booking_total_baggages:
  type: int
booking_nonfree_baggages:
  type: int
booking_baggage_allowance_valid:
  type: bool
  expr: "baggage_allowance_ok([booking_membership_code], [booking_cabin_code], [booking_num_passengers], [booking_total_baggages], [booking_nonfree_baggages])"
```
```python
# dcr2_expr_predicates.py — registered via predicate_file_path=
def baggage_allowance_ok(membership_code, cabin_code, num_passengers,
                          total_baggages, nonfree_baggages) -> bool:
    free_total = (membership_code + cabin_code) * num_passengers
    return nonfree_baggages >= max(0, total_baggages - free_total)
```

---

### R14 — Extra Baggage Price ($50 Each)

**Classification:** NN — same as Declare. Priced automatically in `AirlineTools.book_reservation`.

---

### R15 — Ask About Travel Insurance

**Classification:** IN — same as Declare.

---

### R16 — Travel Insurance Terms ($30/pax, Health/Weather Refund)

**Classification:** NN — same as Declare. Pricing handled by the API; the cancellation-eligibility half is captured separately under R35.

---

## 3. Modify Flight

### R17 — Obtain User ID Before Modifications

**Classification:** WO — already present, three unguarded conditions (flights/baggage/passengers), same limitation as Declare's `Precedence` set.

---

### R18 — Help Locate Reservation ID

**Classification:** WO — already present, three unguarded `get_reservation_details` conditions.

---

### R19 — Basic Economy Flights Cannot Be Modified (Flight Change)

**NL-Rule:** Basic economy flights cannot have their flights changed (cabin changes still allowed — R23).

**Classification:** DI — **refined this session**. The pre-existing gate blocked `update_reservation_flights` unconditionally whenever `reservation_is_basic_economy`, which is over-restrictive: it also blocked the cabin-only upgrades that R23 explicitly permits, since `update_reservation_flights` is one combined API call carrying both `cabin` and `flights` together.

**Comment:** Fixed by scoping the block to cases where the flight segments actually changed, via a new `reservation_flights_changed` bool (order-independent comparison of submitted vs. current `(flight_number, date)` pairs).

**DCR-YAML:**
```yaml
reservation_flights_changed:
  type: bool
update_reservation_flights_valid:
  expr: "(not ([reservation_is_basic_economy] and [reservation_flights_changed])) and (not [reservation_has_flown]) and (not [reservation_route_changed]) and [update_payment_in_profile] and [update_payment_method_type_valid]"
```

**Verified live** (§7): a same-flights, cabin-only change from a real basic-economy reservation → `ALLOW`; a different-flights change from the same reservation → `DECLINE`.

---

### R20 — No Change of Origin, Destination, or Trip Type

**NL-Rule:** Modifications must not change origin, destination, or trip type.

**Classification:** DI — **added this session** (was missing entirely; `update_reservation_flights_valid` previously had no route check at all).

**Comment:** Declare encoded this as three separate `Absence` constraints (origin/destination/trip-type). DCR folds it into one resolver-computed bool feeding the same combined gate as R19/R22/R30, reusing `predicates_v4.route_changed`'s round-trip-aware logic (ported to `dcr2_predicates.route_changed` for consistency with the Declare policy's existing, already-tested semantics rather than reinventing it).

**DCR-YAML:**
```yaml
reservation_route_changed:
  type: bool
update_reservation_flights_valid:
  expr: "... and (not [reservation_route_changed]) and ..."
```

---

### R21 — Kept Segments Have Frozen Prices

**Classification:** NN — same as Declare, API-internal pricing behavior.

---

### R22 — No Cabin Change if Any Flight Already Flown

**Classification:** DI — already present (`reservation_has_flown` in the combined gate).

---

### R23 — All Reservations May Change Cabin Without Changing Flights

**NL-Rule:** All reservations, including basic economy, can change cabin without changing the actual flight segments.

**Classification:** DI — **added this session**, same underlying construct as R20 (`reservation_route_changed`) plus the R19 refinement above. Together, R19+R20+R23 form one coherent "route/segments preserved, with the basic-economy flight-change exception" check.

---

### R24 — Cabin Class Same Across All Flights in Reservation

**NL-Rule:** Cabin class must remain the same across all flights in a reservation; changing cabin for one segment isn't possible.

**Classification:** NN

**Comment:** A classification *refinement* relative to Declare, which counted this as DI via `all_flights_same_cabin_class` — but that predicate's own docstring states it's vacuously `True` whenever no per-flight cabin field is present, which is *always*, since `update_reservation_flights(reservation_id, cabin, flights, payment_id)` takes a single `cabin` for the whole call. There is no way to submit divergent per-segment cabins in the first place. `dcr2.yaml` doesn't encode a check that can structurally never fire; this NN classification is more precise than Declare's technically-present-but-dead-code DI.

---

### R25 / R26 — Pay/Refund Price Difference on Cabin Change

**Classification:** NN — same as Declare, API-internal.

---

### R27 — Baggage: Add Only, No Removal

**Classification:** DI — already present.

**DCR-YAML:**
```yaml
bags_add_only_valid:
  expr: "[new_total_baggages] >= [reservation_current_bag_count]"
```

---

### R28 — Cannot Add Insurance After Booking

**Classification:** NN — same as Declare, no such tool exists.

---

### R29 — Modify Passengers but Not Passenger Count

**Classification:** DI — already present.

**DCR-YAML:**
```yaml
passenger_count_unchanged_valid:
  expr: "[update_num_passengers] == [reservation_num_passengers]"
```

---

### R30 — Single Gift Card or Credit Card for Payment When Flights Changed

**NL-Rule:** Exactly one gift card or credit card, already in the user's profile.

**Classification:** DI (cardinality: NN) — **improved from Declare's WO**, and the type-exclusion half **added this session**.

**Comment:** Declare classified this WO because its `payment_methods` argument is a list, requiring a disjunctive-count custom predicate (`count_gift_cards==1 OR count_credit_cards==1`) that YAML's conjunctive `where` clauses can't express natively. DCR's `update_reservation_flights(reservation_id, cabin, flights, payment_id)` takes a single `payment_id: str`, not a list — the "exactly one" cardinality is **structurally guaranteed by the API shape** (NN, no rule needed at all). "Must be in profile" was already DI (`update_payment_in_profile`). "Must be a gift/credit card, not a certificate" was missing and is now added:

**DCR-YAML:**
```yaml
update_payment_method_type_valid:
  type: bool
update_reservation_flights_valid:
  expr: "... and [update_payment_in_profile] and [update_payment_method_type_valid]"
```
```python
def payment_method_type_ok(payment_id) -> bool:
    pid = str(payment_id).lower()
    return pid.startswith("credit_card") or pid.startswith("gift_card")
```

**Verified live** (§7): submitting a certificate id for `update_reservation_flights` → `DECLINE`.

---

## 4. Cancel Flight

### R31 — Obtain User ID Before Cancellation

**Classification:** WO — same as Declare.

---

### R32 — Help Locate Reservation ID for Cancellation

**Classification:** WO — same as Declare.

---

### R33 — Obtain Cancellation Reason

**Classification:** NN — same as Declare. `cancel_reservation(reservation_id, reason)` requires `reason` as a call argument; the tool call itself enforces it's provided.

---

### R34 — No Cancellation if Any Flight Already Flown

**Classification:** DI — already present.

---

### R35 — Cancellation Eligibility (Disjunctive Conditions)

**NL-Rule:** Cancellation allowed iff: within 24h of booking, OR airline-cancelled, OR business class, OR insured with a covered (health/weather) reason.

**Classification:** WO — **partially improved presentation over Declare**, same underlying limitation.

**Comment:** The *outer* disjunction is now directly visible as a native `or`:
```yaml
reservation_cancellation_eligible:
  expr: "[reservation_cancellation_eligible_base] or [booking_within_24h]"
```
— versus Declare, which had to hide the entire four-way disjunction inside one opaque `cancellation_eligible()` predicate with no partial structure at all. But the *inner* three-way disjunction (business / insured / airline-cancelled) still requires DB lookups (cabin, insurance flag, per-flight status) that can't be pushed into the expr grammar regardless of its native `or` support, so `reservation_cancellation_eligible_base` remains an opaque resolver-computed bool — same fundamental limitation as Declare, just with a more granular DECLINE diagnostic (base vs. 24h are now independently visible, not collapsed into one verdict).

**Shared pre-existing simplification, not fixed here:** neither this dcr2 predicate nor Declare's `cancellation_eligible_base` actually checks that the cancellation *reason* argument matches "health or weather" — both just check `insurance == "yes"` unconditionally. Out of scope for this pass (the `reason` field is free text, not a matchable enum), noted for completeness.

---

### R36 — Refund to Original Payment Methods (5–7 Business Days)

**Classification:** NN — same as Declare.

---

## 5. Refunds and Compensation

### R37 — Do Not Proactively Offer Compensation

**Classification:** IN — same as Declare.

---

### R38 — Compensation Eligibility Gate

**Classification:** WO — same as Declare. The full disjunction requires cross-reservation DB traversal (every reservation of the user, checking for a disrupted flight that's insured or business class), collapsed into one opaque resolver bool `user_compensation_eligible` regardless of grammar power — the complexity is in the DB traversal, not the boolean composition.

---

### R39 — Certificate Amounts ($100 Cancelled / $50 Delayed / No Other Reasons)

**Classification:** WO — same as Declare. `send_certificate(user_id, amount)` carries no `reservation_id`, forcing the resolver to search all of the user's reservations for one that justifies the amount; the cancelled-vs-delayed reason distinction collapses into a single valid-amounts set, same structural loss as Declare's `valid_certificate_amount`.

---

## 6. Summary

### 6.1 Classification Counts

| Classification | Count | Rules |
|----------------|-------|-------|
| **DI** — Direct | 13 | R07, R08, R10, R12, R13, R19, R20, R22, R23, R27, R29, R30, R34 |
| **WO** — Workaround | 8 | R05, R17, R18, R31, R32, R35, R38, R39 |
| **IN** — Inexpressible | 8 | R01, R02, R03, R04, R06, R11, R15, R37 |
| **NN** — Not Needed | 10 | R09, R14, R16, R21, R24, R25, R26, R28, R33, R36 |
| **Total** | 39 | |

**Delta vs. Declare** (`rules-translation-analysis.md` §6.1: DI 13 / WO 9 / IN 8 / NN 9): IN is identical (message/concurrency/open-world blindness doesn't depend on formalism). WO drops by one (R30 moves out) and NN gains one (R24 moves in, on a *stricter, more accurate* classification, not a laxer one) — DI's count is unchanged at 13, but its *membership* changed: R24 left (reclassified NN) and R30 entered (reclassified from Declare's WO), a like-for-like swap that happens to preserve the total.

### 6.2 Classification Distribution by Section

| Section | DI | WO | IN | NN | Total |
|---------|----|----|----|----|-------|
| Domain Basic | 0 | 0 | 4 | 0 | 4 |
| Book Flight | 5 | 1 | 3 | 3 | 12 |
| Modify Flight | 7 | 2 | 0 | 5 | 14 |
| Cancel Flight | 1 | 3 | 0 | 2 | 6 |
| Compensation | 0 | 2 | 1 | 0 | 3 |

> Book Flight (R05–R16): DI = R07, R08, R10, R12, R13; WO = R05; IN = R06, R11, R15; NN = R09, R14, R16.
> Modify Flight (R17–R30): DI = R19, R20, R22, R23, R27, R29, R30; WO = R17, R18; NN = R21, R24, R25, R26, R28.

### 6.3 Pattern Analysis

#### Direct (DI) — What Changed Relative to Declare

The structural story from Declare still holds — parameter-level constraints on a single tool call's payload map directly onto DCR events/relations — but three concrete mechanisms broadened what counts as "direct" for DCR specifically:

1. **API-argument shape can turn a Declare WO into a DCR NN+DI split.** R30's "exactly one gift/credit card" was WO in Declare because `payment_methods` is a list there; DCR's `update_reservation_flights` takes a single `payment_id` string, so the cardinality constraint simply doesn't need encoding (NN) — only the type-exclusion needed adding (DI). This isn't a DCR expressiveness advantage per se, it's a difference in the underlying tool signature that happens to interact differently with each formalism's constraint shape.
2. **`FunctionCallExpression`/`predicate_registry` genuinely earns its keep exactly once** (R13): a lookup-table formula over already-resolved int event values, too awkward to nest as `if/then/else` chains, cleanly expressed as real Python composed *after* the resolver hook gets the raw facts in. This was previously flagged as a hypothetical in `DCR_DATA_ARCHITECTURE.md` §6 ("dcr2 has never needed this") — R13 is now the first real instance.
3. **The combined-gate pattern scales cleanly to more sub-checks.** Adding R08/R13 to `book_reservation_valid` and R20/R23/R30 to `update_reservation_flights_valid` was purely additive `and`-ing onto an existing single computed event — no new guarded relations, no risk of the "last writer wins" bug documented in `DCR_DATA_ARCHITECTURE.md` §4.5, because the combined-gate structure was already in place before this pass.

#### Workaround (WO) — Same Root Causes as Declare, One Real Improvement

Both underlying causes from the Declare analysis persist unchanged in DCR:

1. **Disjunctive eligibility requiring DB traversal** (R35, R38, R39): DCR's native `or` operator helps only at the *outermost* level where the disjunction is over facts already resolved as separate events (R35's `base OR 24h`); it cannot help where the disjunction itself requires searching across reservations or entities not yet represented as graph events (R38, R39, and R35's *inner* three-way OR) — that complexity is DB-traversal complexity, not boolean-composition complexity, and no expression grammar removes it.
2. **Identity/provenance verification via lookup enforcement** (R05, R17, R18, R31, R32): DCR's unguarded `conditions` relation and Declare's `Precedence` template express *identical* semantics here — "source must execute before target is enabled" — so this WO carries over completely unchanged, formalism-independent.

The one genuine improvement: R35's diagnostic granularity. Collapsing four conditions into one opaque predicate (Declare) vs. one native `or` over two already-distinguishable sub-verdicts (DCR) doesn't change the classification, but it changes what a DECLINE message can actually tell the agent — this is a byproduct of DCR's `enablement_blockers` reporting the state of each event independently (`DCR_DATA_ARCHITECTURE.md` §2.4), not something Declare's Absence-violation reporting does.

#### Inexpressible (IN) — Identical Boundary to Declare

All 8 IN rules are unchanged and for the identical reasons as Declare: outgoing/incoming message content (R01, R02, R03, R06, R15, R37), open-world scope/subjectivity reasoning (R02, R04), and concurrency (R03). This is not a DCR-specific limitation — it's a limitation of evaluating *any* event-log-shaped policy formalism against a benchmark harness that only exposes tool-call events, never conversational content, to the constraint checker. Neither Declare nor DCR can close this gap without a fundamentally different input (e.g. an NL-aware judge model reading the transcript, which is a different mechanism entirely, orthogonal to either formalism).

One partial exception, noted under R04: DCR's `excludes` relation captures the *consequence* of a transfer (irrevocably disabling further write actions) more directly and structurally than Declare's template vocabulary would, even though the *antecedent* (was this transfer actually justified?) remains equally IN in both.

#### Not Needed (NN) — Two Additions, Both Refinements Not New Findings

NN grew from 9 to 10 rules, but the addition (R24) is a *classification correction*, not a newly-discovered API guarantee: Declare's own R24 predicate (`all_flights_same_cabin_class`) already documented in its docstring that it's vacuously true given the API's single-`cabin`-per-call shape — this pass just stopped encoding a check that can structurally never fire, rather than carrying dead code forward. R09/R14/R16/R21/R25/R26/R28/R33/R36 are unchanged from Declare, covering the same pricing mechanics, structural API constraints, and backend side-effects.

### 6.4 Expressiveness Boundary

| Requirement | Declare | DCR |
|-------------|---------|-----|
| Count / cardinality check on tool argument | Yes (DI) | Yes (DI) |
| Ordering / precedence between tool calls | Yes (DI/WO) | Yes (DI/WO, identical mechanism to Declare's `Precedence`) |
| Derived-state predicate on prior events | Yes (DI) | Yes (DI) |
| Disjunctive condition over **already-graph-resolved** values | No — requires custom predicate (WO) | **Yes** — native `or`/`and`/`not`/`if-then-else` (DI) |
| Disjunctive condition requiring **fresh DB traversal** | No — requires custom predicate (WO) | No — still requires the resolver hook (WO); grammar-level disjunction doesn't reach into unresolved facts |
| Pure formula over already-resolved values too complex for inline boolean/comparison ops | Not distinguished from other WO cases | `FunctionCallExpression`/`predicate_registry` (DI) — demonstrated once, R13 |
| "This event permanently disables that one" (terminal/irrevocable state) | Not a first-class template | Native `excludes` relation (DI at the mechanism level, though R04's own trigger condition stays IN) |
| Reading agent's outgoing NL message | No (IN) | No (IN) |
| Reading user's incoming NL message | No (IN) | No (IN) |
| Reasoning about scope, intent, or subjectivity | No (IN) | No (IN) |
| Concurrent / simultaneous events | No (IN) | No (IN) |

The headline finding: **DCR's expressiveness advantage over Declare is real but narrower than the grammar's native disjunction/if-then-else support might suggest.** It only converts a Declare WO into a DCR DI when the disjunction is over values *already resolvable as graph events* (R35's outer OR) — the moment a rule needs to reach into live DB state or traverse multiple entities to even *form* the disjunction's operands (R38, R39, R35's inner OR), both formalisms are equally stuck funneling that logic through an opaque predicate function, and the choice of grammar features becomes irrelevant. The message/concurrency/scope IN boundary is completely unaffected by which constraint formalism sits behind it — that boundary is set by what the benchmark harness exposes to *any* constraint checker, not by DCR vs. Declare.

---

## 7. Live Verification

### 7.1 Compile / load check

`AgentDCRConstraints().parse_from_yaml("dcr2.yaml", data_event_resolver=..., predicate_file_path="dcr2_expr_predicates.py")` — the exact call `SecureAirlineAgent` now makes — was run directly. Result: `DataDcrGraph` with **47 events** (up from 38 before this pass), `predicate_registry == {'baggage_allowance_ok'}` confirming the new `FunctionCallExpression` wiring is live, and `is_decision_event('booking_baggage_allowance_valid') == True` with the expected parsed expression tree. No parse errors, no import errors.

### 7.2 Unit-level regression check

`pytest tests/test_domains/test_airline/test_dcr2_data_resolver.py tests/test_secure_airline_agent.py` — **20/20 passed**, including the pre-existing adversarial `book_reservation` cases (unknown payment method, too many passengers) that exercise the same combined gate now extended with two more conjuncts. The fixture was updated to pass `predicate_file_path=` (without it, `booking_baggage_allowance_valid`'s `FunctionCallExpression` would raise `KeyError` on an empty registry, which `resolve_closure` catches as "not yet resolvable" — leaving `book_reservation_valid` permanently pending and every booking permanently blocked; this was caught and fixed before the regression could land).

### 7.3 Targeted interactive verification of the 5 new/refined rules

Each new/refined check was exercised directly against the real `dcr2.yaml` + real `db.json`, isolated from the other checks, via `DCRStateValidator.validate`:

| Rule | Scenario | Result |
|---|---|---|
| R08 | `book_reservation` with a passenger missing `dob` | `DECLINE` — `booking_passengers_info_complete` correctly fails |
| R13 | Gold member, economy, 1 pax (free allowance = 3), `total_baggages=5, nonfree_baggages=1` (under-declared) | `DECLINE` — `baggage_allowance_ok` (via `FunctionCallExpression`) correctly fails |
| R13 | Same scenario, `nonfree_baggages=2` (exactly covers the excess) | `ALLOW` |
| R19 | Real basic-economy reservation (`1OWO6T`), same flights resubmitted, cabin upgraded to economy | `ALLOW` — confirms the refinement correctly permits cabin-only changes from basic economy |
| R19/R20 | Same reservation, different flight segments submitted | `DECLINE` — confirms flight-segment changes from basic economy are still blocked |
| R30 | `update_reservation_flights` with a certificate `payment_id` | `DECLINE` — `update_payment_method_type_valid` correctly rejects it |

All six matched the intended policy behavior exactly.

### 7.4 Three live test tasks (real LLM agent + user simulator)

`tau2 run --domain airline --agent secure_airline_agent --sec-file dcr2.yaml --agent-llm azure/gpt-5-mini-US --user-llm azure/gpt-5-mini-US --num-trials 1 --task-set basic --num-tasks 3`, run twice — once as the pre-existing baseline (before this session's rule additions) and once after:

| Task | Before this pass | After this pass |
|---|---|---|
| 0 | Reward 1.0 — DCR correctly declined `get_reservation_details` before `get_user_details` | Reward 1.0 — identical DECLINE, byte-identical diagnostic text |
| 1 | Reward 1.0 | Reward 1.0 |
| 2 | Reward 0.0 (agent transfers to human instead of calling `send_certificate`) | Reward 0.0 — identical outcome, identical action sequence |
| **Average** | **0.6667** | **0.6667** |

Identical per-task outcomes, identical average reward, identical DECLINE diagnostic text. None of these three tasks happened to exercise `book_reservation` or `update_reservation_flights` with payloads that would trigger the newly-added checks (that's what §7.3's targeted interactive tests are for) — this run's purpose is confirming **no regression**: the policy still loads, the agent/user simulation still runs end-to-end against real Azure-hosted GPT-5-mini agent and user models without errors, and every pre-existing behavior is unchanged.

### 7.5 Conclusion

`dcr2.yaml` compiles and runs correctly after adding the 5 missing/refined rules (R08, R13, R19-refinement, R20/R23, R30-refinement). No regressions in either the unit suite or live task behavior; all new checks independently verified to fire correctly in both directions (block the violating case, allow the compliant one).
