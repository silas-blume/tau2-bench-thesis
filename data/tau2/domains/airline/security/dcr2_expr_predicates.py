"""Predicate functions for dcr2.yaml's in-expression FunctionCallExpression calls.

Loaded via ``AgentDCRConstraints.parse_from_yaml(..., predicate_file_path=...)``
and registered on the compiled graph's ``predicate_registry`` (pm4py's
``load_predicates``), resolved at guard/decision evaluation time from bare
function-call syntax inside an ``expr:``/``guard`` string, e.g.
``baggage_allowance_ok([booking_membership_code], ...)``.

Deliberately a separate file from ``dcr2_predicates.py`` and from
``dcr2_data_resolver.py``, and wired through a different mechanism entirely
(``graph.predicate_registry``, not the ``DataEventResolver`` hook):

- ``dcr2_predicates.py`` functions are called by ``dcr2_data_resolver.py``
  with already-looked-up *domain objects* (``Reservation``, ``User``,
  ``FlightDB``) -- they do their own DB-shaped reasoning but never see a DCR
  event value directly.
- Functions here are called by the *engine's* ``FunctionCallExpression``
  evaluator with already-resolved DCR *event values* (plain ints/bools) as
  positional arguments -- they never see a domain object or do a lookup of
  their own; getting the raw facts into the graph as event values in the
  first place is still the resolver hook's job (see ``dcr2_data_resolver.py``
  and ``booking_membership_code``/``booking_cabin_code``/etc. in dcr2.yaml).

This is the "pure function of already-resolved event values, too complex to
comfortably nest as and/or/if-then-else" case documented in
DCR_DATA_ARCHITECTURE.md §6 -- the checked-bag-allowance table is dcr2's
first real use of this mechanism (previously unused, only the resolver hook
was exercised).
"""

from __future__ import annotations


def baggage_allowance_ok(
    membership_code: int,
    cabin_code: int,
    num_passengers: int,
    total_baggages: int,
    nonfree_baggages: int,
) -> bool:
    """Free-bag allowance per passenger = membership_code + cabin_code
    (regular/basic_economy=0 ... gold/business=4, matching policy.md's
    table: regular 0/1/2, silver 1/2/3, gold 2/3/4 for basic_economy/
    economy/business). Valid iff nonfree_baggages covers whatever total
    exceeds the free allowance.
    """
    free_per_passenger = membership_code + cabin_code
    free_total = free_per_passenger * num_passengers
    required_nonfree = max(0, total_baggages - free_total)
    return nonfree_baggages >= required_nonfree
