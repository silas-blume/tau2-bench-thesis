# Prong 1 — Scenario Expressiveness Analysis: Airline Agent Policy

---
YAML encodings, a classification (direct / partial / workaround / inexpressible), and a qualitative commentary on why
## S01: Confirmation Before Database-Modifying Actions

**Natural Language:**
Before taking any action that updates the booking database (booking, modifying flights, editing baggage, changing cabin class, or updating passenger information), the agent must list the action details and obtain explicit user confirmation (yes) to proceed.

**YAML Encoding:**
```yaml
# TODO
```

**Classification:** <!-- DI / PA / WO / IN -->

**Commentary:**
<!-- What construct(s) did you use? What is lost (PA), unnatural (WO), or impossible (IN)? Alternatives considered? -->

---

## S02: Restrict to Known Information and Tools Only

**Natural Language:**
The agent must not provide any information, knowledge, or procedures not provided by the user or available tools. It must not give subjective recommendations or comments.

**YAML Encoding:**
```yaml
# TODO
```

**Classification:** <!-- DI / PA / WO / IN -->

**Commentary:**
<!-- -->

---

## S03: One Tool Call at a Time / No Simultaneous Tool Call and User Response

**Natural Language:**
The agent must only make one tool call at a time. If it makes a tool call, it must not respond to the user simultaneously. If it responds to the user, it must not make a tool call at the same time.

**YAML Encoding:**
```yaml
# TODO
```

**Classification:** <!-- DI / PA / WO / IN -->

**Commentary:**
<!-- -->

---

## S04: Deny Policy-Violating Requests

**Natural Language:**
The agent must deny user requests that are against this policy.

**YAML Encoding:**
```yaml
# TODO
```

**Classification:** <!-- DI / PA / WO / IN -->

**Commentary:**
<!-- -->

---

## S05: Human Agent Transfer Protocol

**Natural Language:**
The agent must transfer the user to a human agent if and only if the request cannot be handled within the scope of its actions. It must try everything before transferring. It must first explicitly refuse and explain, and only transfer if the customer insists. To transfer, the agent must first call transfer_to_human_agents, then send the message 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.'

**YAML Encoding:**
```yaml
# TODO
```

**Classification:** <!-- DI / PA / WO / IN -->

**Commentary:**
<!-- -->

---

## S06: Payment Method Constraints (Booking)

**Natural Language:**
Each reservation can use at most one travel certificate, at most one credit card, and at most three gift cards. The remaining amount of a travel certificate is not refundable. All payment methods must already be in the user profile.

**YAML Encoding:**
```yaml
# TODO
```

**Classification:** <!-- DI / PA / WO / IN -->

**Commentary:**
<!-- -->

---

## S07: Cabin Class Consistency Across Flights

**Natural Language:**
Cabin class must be the same across all flights in a reservation.

**YAML Encoding:**
```yaml
# TODO
```

**Classification:** <!-- DI / PA / WO / IN -->

**Commentary:**
<!-- -->

---

## S08: Passenger Count Limit

**Natural Language:**
Each reservation can have at most five passengers.

**YAML Encoding:**
```yaml
# TODO
```

**Classification:** <!-- DI / PA / WO / IN -->

**Commentary:**
<!-- -->

---

## S09: Checked Bag Allowance (Membership × Cabin Class)

**Natural Language:**
Free checked bag allowance depends on the booking user's membership level and each passenger's cabin class:
- Regular: 0 (basic economy) / 1 (economy) / 2 (business)
- Silver: 1 (basic economy) / 2 (economy) / 3 (business)
- Gold: 2 (basic economy) / 3 (economy) / 4 (business)
Each extra bag costs $50.

**YAML Encoding:**
```yaml
# TODO
```

**Classification:** <!-- DI / PA / WO / IN -->

**Commentary:**
<!-- -->

---

## S10: Travel Insurance Offer and Conditions

**Natural Language:**
The agent must ask if the user wants to buy travel insurance during booking. Insurance costs $30 per passenger and enables a full refund for cancellations due to health or weather reasons.

**YAML Encoding:**
```yaml
# TODO
```

**Classification:** <!-- DI / PA / WO / IN -->

**Commentary:**
<!-- -->

---

## S11: Basic Economy Cannot Be Modified (Flight Change)

**Natural Language:**
Basic economy flights cannot be modified (flight change only). Other reservations can be modified without changing the origin, destination, or trip type.

**YAML Encoding:**
```yaml
# TODO
```

**Classification:** <!-- DI / PA / WO / IN -->

**Commentary:**
<!-- -->

---

## S12: Cabin Change Rules

**Natural Language:**
Cabin cannot be changed if any flight in the reservation has already been flown. Otherwise, all reservations (including basic economy) can change cabin without changing flights. Cabin class must remain the same across all flights. If the new price is higher, the user pays the difference; if lower, the user is refunded the difference.

**YAML Encoding:**
```yaml
# TODO
```

**Classification:** <!-- DI / PA / WO / IN -->

**Commentary:**
<!-- -->

---

## S13: Baggage and Insurance Modification Restrictions

**Natural Language:**
The user can add but not remove checked bags. The user cannot add insurance after initial booking.

**YAML Encoding:**
```yaml
# TODO
```

**Classification:** <!-- DI / PA / WO / IN -->

**Commentary:**
<!-- -->

---

## S14: Passenger Modification (No Count Change)

**Natural Language:**
The user can modify passenger details but cannot modify the number of passengers. Even a human agent cannot modify the number of passengers.

**YAML Encoding:**
```yaml
# TODO
```

**Classification:** <!-- DI / PA / WO / IN -->

**Commentary:**
<!-- -->

---

## S15: Payment Method for Flight Changes

**Natural Language:**
If flights are changed, the user must provide a single gift card or credit card for payment or refund. The payment method must already be in the user profile.

**YAML Encoding:**
```yaml
# TODO
```

**Classification:** <!-- DI / PA / WO / IN -->

**Commentary:**
<!-- -->

---

## S16: Cancellation Eligibility Rules

**Natural Language:**
If any portion of the flight has already been flown, cancellation is not possible and a transfer is needed. Otherwise, cancellation is allowed if any of the following is true:
- The booking was made within the last 24 hours
- The flight was cancelled by the airline
- It is a business class flight
- The user has travel insurance and the cancellation reason is covered (health or weather)
The agent must verify these rules before calling the API.

**YAML Encoding:**
```yaml
# TODO
```

**Classification:** <!-- DI / PA / WO / IN -->

**Commentary:**
<!-- -->

---

## S17: Cancellation Refund

**Natural Language:**
The refund goes to the original payment methods within 5 to 7 business days.

**YAML Encoding:**
```yaml
# TODO
```

**Classification:** <!-- DI / PA / WO / IN -->

**Commentary:**
<!-- -->

---

## S18: Compensation Eligibility Gate

**Natural Language:**
Do not proactively offer compensation unless the user explicitly asks. Do not compensate if the user is a regular member with no travel insurance flying basic economy or economy. Only compensate if the user is a silver/gold member, has travel insurance, or flies business class. Always confirm the facts before offering compensation.

**YAML Encoding:**
```yaml
# TODO
```

**Classification:** <!-- DI / PA / WO / IN -->

**Commentary:**
<!-- -->

---

## S19: Cancelled Flight Compensation

**Natural Language:**
If the user complains about a cancelled flight, the agent can offer a certificate after confirming the facts. Amount: $100 × number of passengers.

**YAML Encoding:**
```yaml
# TODO
```

**Classification:** <!-- DI / PA / WO / IN -->

**Commentary:**
<!-- -->

---

## S20: Delayed Flight Compensation

**Natural Language:**
If the user complains about a delayed flight and wants to change or cancel the reservation, the agent can offer a certificate after confirming the facts and completing the change or cancellation. Amount: $50 × number of passengers.

**YAML Encoding:**
```yaml
# TODO
```

**Classification:** <!-- DI / PA / WO / IN -->

**Commentary:**
<!-- -->

---

## Cross-Scenario Pattern Analysis

<!-- To be filled after all encodings are complete -->

### Friction Patterns
<!-- Which constraint properties consistently yield PA / WO / IN? -->

### Strength Patterns
<!-- Which constraint properties consistently yield DI? -->

### Construct-Level Patterns
<!-- Which format constructs appear in workarounds most often? Which are never needed? -->

### Boundary Patterns
<!-- Do IN cases share a common property? -->

---

## Root Cause Analysis

<!-- To be filled after pattern analysis -->

---

## Expressiveness Characterization

<!-- Final synthesis — to be filled last -->

### Expressive Strengths

### Expressive Limitations

### Expressive Boundary

### Agent-Specificity Assessment

### Overall Expressiveness Verdict