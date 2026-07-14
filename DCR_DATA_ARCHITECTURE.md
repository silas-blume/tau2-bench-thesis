# Data-Aware DCR Evaluation Architecture

This document explains, end to end, how a data-aware DCR (Dynamic Condition
Response) policy is defined, compiled, loaded, and evaluated against a live
LLM agent's tool calls in this project — including the predicate/resolver
mechanism that supplies real-world values into the graph.

The system spans **three repositories**, each with a distinct
responsibility:

| Repo | Role | Tag used below |
|---|---|---|
| [`pm4py-dcr`](/home/silas/projects/pm4py-dcr) | The DCR **engine**: graph data structures, enablement/execution semantics, the expression language, XML import/export. Domain- and agent-agnostic. | `[pm4py-dcr]` |
| [`thesis-dpm-secure-langgraph`](/home/silas/projects/thesis-dpm-secure-langgraph) | The **wrapper**: tracks committed vs. candidate DCR state across a conversation, validates/executes agent tool calls against it, integrates with LangGraph's tool-call lifecycle. | `[thesis-dpm-secure-langgraph]` |
| `tau2-bench-thesis` (this repo) | The **domain**: the airline agent's actual policy files, the DB-backed resolver that answers the graph's open questions, and the LangGraph↔tau2-orchestrator glue. | `[tau2-bench-thesis]` |

Each depends only on the one below it: `tau2-bench-thesis → thesis-dpm-secure-langgraph → pm4py-dcr`. Nothing in `pm4py-dcr` knows about agents or tool calls; nothing in `thesis-dpm-secure-langgraph` knows about airlines or reservations.

## 1. Overview & layering

```mermaid
flowchart TD
    subgraph TAU2["tau2-bench-thesis  [domain integration]"]
        SAA["SecureAirlineAgent"]
        SLA["SecureLangGraphAdapter"]
        AT["AirlineTools / FlightDB"]
        DCR2Y["dcr2.yaml → dcr2.xml"]
        DCR2R["dcr2_data_resolver.py"]
        DCR2P["dcr2_predicates.py"]
        CDY["scripts/compile_dcr_yaml.py"]
    end

    subgraph WRAP["thesis-dpm-secure-langgraph  [agent-validation wrapper]"]
        ADC["AgentDCRConstraints"]
        DST["DCRStateTracker"]
        DSV["DCRStateValidator"]
        SSG["SecureStateGraph / SecureToolNode"]
        TC["TraceCollector"]
        YC["dcr_yaml_parser / dcr_yaml_compiler"]
    end

    subgraph ENGINE["pm4py-dcr  [DCR engine]"]
        DDG["DataDcrGraph"]
        DS["DataSemantics"]
        EXPR["Expression AST + parser"]
        XMLIO["XML_DCR_DATA importer / exporter"]
    end

    SAA --> ADC
    SLA --> SSG
    SLA --> AT
    DCR2R --> DCR2P
    SAA -. dynamically loads .-> DCR2R
    CDY -. compiles .-> DCR2Y
    ADC --> DST
    DSV --> DST
    SSG --> DSV
    SSG --> TC
    TC -. publishes events to .-> DST
    ADC --> YC
    DST --> DS
    DST --> DDG
    YC --> EXPR
    DS --> DDG
    DDG --> XMLIO
```

The short version: **`pm4py-dcr` knows how to enable/execute one event on one graph. `thesis-dpm-secure-langgraph` decides *when* to ask that question and what to do with the answer (allow/decline a tool call, commit real progress). `tau2-bench-thesis` supplies the actual policy content and the real-world data that answers the graph's open input events.**

---

## 2. The engine layer `[pm4py-dcr]`

### 2.1 Base DCR model

A DCR graph is a set of **events** plus relations between them, evaluated against a **marking** of three sets:

- `included` — events currently "in play." A non-included event cannot execute.
- `pending` — events with an outstanding obligation (a required response).
- `executed` — events that have run at least once.

The four core relations (`pm4py/objects/dcr/obj.py`, `Relations`/`TemplateRelations` enums):

| Relation | Template key | Effect on `execute(e)` |
|---|---|---|
| condition | `conditionsFor` (keyed by **target**) | blocks *enablement* of the target until the source has executed |
| response | `responseTo` (keyed by **source**) | adds the target to `pending` |
| include | `includesTo` (keyed by **source**) | adds the target to `included` |
| exclude | `excludesTo` (keyed by **source**) | removes the target from `included` |

Extended by two more (`pm4py/objects/dcr/extended/*`): **milestone** (`milestonesFor`, blocks enablement while a milestone source is included-and-pending) and **no-response** (`noResponseTo`, clears a target's pending obligation).

`DcrSemantics` (`pm4py/objects/dcr/semantics.py`):
- `enabled(graph)` — start from `included`; discard any target whose `conditionsFor`/`milestonesFor` source is included-but-not-executed (condition) or included-and-pending (milestone).
- `execute(graph, event)` — move `event` from `pending` to `executed`; apply `excludesTo` (discard from `included`), `includesTo` (add to `included`), `responseTo` (add to `pending`).
- `is_accepting(graph)` — true iff `pending ∩ included == ∅`.

### 2.2 Class hierarchy

```mermaid
flowchart TD
    subgraph Graphs
        A["DcrGraph<br/>(events, 4 relations, marking)"] --> B["ExtendedDcrGraph<br/>(+ milestones, no-response)"]
        B --> C["HierarchicalDcrGraph<br/>(+ nesting/subprocesses)"]
        C --> D["TimedDcrGraph<br/>(+ timed conditions/responses)"]
        D --> E["DataDcrGraph<br/>(+ types, decisions, guards, event values)"]
    end
    subgraph Semantics
        F["DcrSemantics"] --> G["ExtendedSemantics"] --> H["DataSemantics"]
    end
    E -.evaluated by.-> H
```

Note the asymmetry: `DataDcrGraph` sits at the top of the full graph-object chain (through `TimedDcrGraph`), but `DataSemantics` subclasses `ExtendedSemantics` **directly** — it does not chain through `TimedSemantics`, so data-aware graphs don't get timed-deadline enforcement "for free."

### 2.3 `DataDcrGraph` — what "data-aware" adds

File: `pm4py/objects/dcr/data/obj.py`. On top of everything above:

- **`event_types: Dict[event_id, DataType]`** — `DataType` is `INT | BOOL | VOID`.
- **`decisions: Dict[event_id, INPUT_MARKER | Expression]`** — this is *the* classifier:
  - `decisions[e] == INPUT_MARKER` ("`?`") → **input event**: its value must be supplied from outside (`is_input_event(e)` returns `True`).
  - `decisions[e]` is an `Expression` AST → **decision/computed event**: its value is derived from other events (`is_decision_event(e)` returns `True`).
  - `e` absent from `decisions` entirely → a plain **void event** (may still carry `dataType`, but nothing computes or supplies it — it's a structural/sequencing event only).
- **`event_values`** (on `DataMarking`, which extends the base `Marking`) — the most-recently-produced value for each *executed* event. An event has no value until it has executed at least once.
- **Six "guarded" relations** — `guarded_conditions`, `guarded_responses`, `guarded_includes`, `guarded_excludes`, `guarded_milestones`, `guarded_noresponses`. Structurally these are the data-aware analogue of the four-plus-two base relations, but shaped as **dict-of-dict-of-`Guard`** instead of the base's dict-of-set:

  | Guarded relation | Shape | Keying (matches its unguarded counterpart) |
  |---|---|---|
  | `guarded_conditions` | `{target: {source: Guard}}` | by **target** |
  | `guarded_milestones` | `{target: {source: Guard}}` | by **target** |
  | `guarded_responses` | `{source: {target: Guard}}` | by **source** |
  | `guarded_includes` | `{source: {target: Guard}}` | by **source** |
  | `guarded_excludes` | `{source: {target: Guard}}` | by **source** |
  | `guarded_noresponses` | `{source: {target: Guard}}` | by **source** |

  This dict-of-dict-with-`Guard`-values vs. dict-of-set shape is the structural tell for "this relation is data-aware" anywhere you see it in code (e.g. `DCRStateTracker._collect_enablement_blockers`, described below, branches on exactly this).
- **`predicate_registry: Dict[name, callable]`** — injectable, used by `FunctionCallExpression` guards (see §2.6).
- **`obj_to_template()`** — extends the base template dict with `eventTypes`, `decisions`, the six `guarded*` keys, and `marking.eventValues`.

### 2.4 `DataSemantics` — enablement and execution

File: `pm4py/objects/dcr/data/semantics.py`.

**`_evaluate_guard(guard, event_values, registry)`** wraps `Guard.evaluate(...)` and treats any `ValueError`/`KeyError`/`TypeError` (e.g. referencing an event with no value yet) as **`False`** — an unanswerable guard blocks nothing, it simply isn't "true" yet.

**`enabled(graph)`** (falls back to plain `DcrSemantics.enabled` if `graph` isn't a `DataDcrGraph` — the data-aware semantics are a strict superset):
1. Start from `included`.
2. Unguarded conditions/milestones behave exactly as in the base engine — they **always** block if the source is unmet.
3. **Guarded conditions/milestones only block when the guard evaluates `True`.** An inactive/false/unanswerable guard does not block enablement — this is the crucial asymmetry between guarded and unguarded relations.

**`execute(graph, event, input_value=None)`**:
1. **Compute the event's value**: input events take the caller-supplied `input_value`; decision events evaluate their `Expression`; void events store nothing.
2. Mark executed, store the value in `event_values` (if not `None`).
3. Apply relation effects **in this fixed order — no-response → exclude → include → response — each as unguarded-then-guarded**:

```mermaid
flowchart TD
    A["execute(graph, event, input_value)"] --> B{"input event? (decision == '?')"}
    B -- yes --> C["value = input_value"]
    B -- no, decision is an Expression --> D["value = decision.evaluate(event_values, registry)"]
    B -- no decision at all (void) --> E["value = None"]
    C --> F["mark executed, remove from pending"]
    D --> F
    E --> F
    F --> G["if value is not None: event_values[event] = value"]
    G --> H["apply noResponseTo(event): discard targets from pending"]
    H --> I["apply guardedNoResponses(event): discard targets whose guard is now true"]
    I --> J["apply excludesTo(event): discard targets from included"]
    J --> K["apply guardedExcludes(event): discard targets whose guard is now true"]
    K --> L["apply includesTo(event): add targets to included"]
    L --> M["apply guardedIncludes(event): add targets whose guard is now true"]
    M --> N["apply responseTo(event): add targets to pending"]
    N --> O["apply guardedResponses(event): add targets whose guard is now true"]
```

The event's own value is stored in step G **before** any guard is evaluated in H–O — so a guard is allowed to reference the very event that's executing (e.g. `[Decision] == 2`).

### 2.5 Expression language

Grammar (`pm4py/objects/dcr/data/expressions.py`, `expression_parser.py`), surface syntax designed to be embeddable as an XML attribute string:

```
?                              input-event marker
void                           VoidExpression
42 / true / false              int / bool constants
[EventId]                      reference to another event's current value
[A] + [B], [A] - 5, [A] * 2    arithmetic  (+ - *)
[A] == 2, <, >, <=, >=         comparison  (note: == not =)
and / or / not (...)           boolean connectives
if C then X else Y             conditional
name(arg1, arg2, ...)          predicate function call (FunctionCallExpression)
```

Precedence, lowest to highest: `if/then/else < or < and < not < comparison < + - < * < atom`.

`Guard(expression=None)` wraps an *optional* boolean `Expression` — a guard with no expression (`is_trivial`) always evaluates `True`. In the XML format, a relation with no `guard=` attribute is exactly this: a trivial always-true guard (equivalent to an unguarded relation, just expressed through the guarded machinery).

### 2.6 Two distinct "predicate" mechanisms — don't conflate them

This project actually has **two unrelated things that could be called "predicates,"** at two different layers. Knowing which one dcr2 uses matters:

1. **`[pm4py-dcr]` built-in `FunctionCallExpression` + `predicate_registry`.** A guard/expr string can call a bare function name, e.g. `requiresApproval([Amount])`, resolved at evaluation time against `graph.predicate_registry` (a plain `Dict[str, callable]`). `load_predicates(path)`/`resolve_predicates(...)` (`pm4py/objects/dcr/data/predicate_loader.py`) dynamically import a `.py` file and register its top-level functions. **This is an engine-level capability that dcr2 does not use** — none of dcr2.yaml's `expr:`/guard strings call a function; they only combine event references with comparison/boolean operators.
2. **The external *data-event-resolver hook*, `[thesis-dpm-secure-langgraph]`.** This is a completely separate mechanism (§3.2/§3.6): a callback invoked by `DCRStateTracker`, *outside* the expression language entirely, that supplies the **value of an input event** before enablement is checked. **This is what dcr2 actually uses** — `dcr2_predicates.py`'s functions are plain Python called from `dcr2_data_resolver.py`, never referenced from inside an `.xml`/`.yaml` expression string at all.

In short: pm4py's predicate registry lets a *guard expression* call out to Python; the resolver hook lets *something outside the graph* decide what an *input event's value* is. dcr2 only uses the second.

### 2.7 XML import/export — `XML_DCR_DATA`

Registered as `Variants.XML_DCR_DATA` in `pm4py/objects/dcr/importer/importer.py` / `exporter/exporter.py`, implemented in `importer/variants/xml_dcr_data.py` / `exporter/variants/xml_dcr_data.py`.

**Event XML shape:**
```xml
<event id="Amount"   dataType="int"  decision="?"/>                 <!-- input event -->
<event id="Submit"   dataType="void"/>                              <!-- void event -->
<event id="Decision" dataType="int"  decision="if [Amount] &lt; 200 then 1 else 2"/>  <!-- computed event -->
```
`decision="?"` → input; `decision="<expr>"` → computed; `dataType` present with no `decision` attribute at all → void.

**Guarded-relation XML shape**, one section per relation type, key/value attribute names matching the table in §2.3:
```xml
<guardedExcludes>
  <exclude sourceId="book_reservation_valid" targetId="book_reservation" guard="(not [book_reservation_valid])"/>
</guardedExcludes>
```
A missing `guard=` attribute parses to a trivial always-true `Guard()`.

**Dispatch to the right graph class** happens in `cast_to_dcr_object` (`pm4py/objects/dcr/utils/utils.py`): if *any* of `eventTypes`/`decisions`/`guarded*` keys are populated in the parsed template, the result is promoted all the way to `DataDcrGraph` — ahead of the timed/hierarchical checks further down the same dispatch function. **This single check is why a file merely containing `dataType="..."` attributes anywhere is enough to make it a data-aware graph** — relevant to the `.xml` sniff-check in `[tau2-bench-thesis]`, §4.1.

---

## 3. The wrapper layer `[thesis-dpm-secure-langgraph]`

### 3.1 `AgentDCRConstraints`

File: `src/thesis_dpm_secure_langgraph/constraints/agent_dcr_constraints.py`. A thin container around one parsed graph plus per-event error messages, lazily building a `DCRStateTracker`.

- **`parse_from_file(model_path, variant=..., predicate_file_path=None, data_event_resolver=None)`** — imports via pm4py's `dcr_importer.apply(...)`; also scans the raw XML for `description` attributes to seed per-event error messages (constructor-supplied messages win over these).
- **`parse_data_from_file(model_path, ...)`** — convenience wrapper selecting the `XML_DCR_DATA` importer variant, so the result is guaranteed a `DataDcrGraph`. This is what `[tau2-bench-thesis]`'s `SecureAirlineAgent` calls for `dcr2.xml`.
- **`parse_from_yaml(yaml_path, ...)`** — parses the DCR-YAML dialect and compiles **directly to an in-memory graph** via this package's own compiler (§3.5), with no XML round-trip.
- **`to_dcr_graph()`** — returns the raw pm4py graph object.
- **`get_state_tracker()`** — returns (lazily building if needed) the `DCRStateTracker`.
- **`get_violation_messages(violated_indices, fallback_constraints, violated_activities)`** — for each violated index, combines the **static** per-event description (from the XML/YAML `description`) with the **dynamic** diagnostic (`serialized_constraints[idx]`, produced by `DCRStateTracker._collect_enablement_blockers`): `"<static> [Diagnostic: <dynamic>]"` when both exist, falling back gracefully when only one is present.

### 3.2 `DCRStateTracker` — the validate/commit split

File: `src/thesis_dpm_secure_langgraph/constraints/dcr_state.py`. This is the core state machine.

- **Two graph copies**: `_initial_graph` (rollback baseline, never mutated after construction) and `_committed_graph` (the real, persistent, live state — advances only on genuinely successful, completed tool calls).
- **`_committed_events: dict[id(event) → Event]`** — deliberately a dict holding a strong reference, not a bare `set[int]`. A `set[int]` of raw `id()`s would be unsound: once a short-lived `Event` object is garbage-collected, CPython can reuse its memory address for a *different* later `Event`, producing the same `id()` and causing the tracker to silently (and wrongly) treat a genuinely new event as an already-committed duplicate.
- **`_relevant_event_ids`** — precomputed once from every relation key (including the data-aware ones if applicable); any tool call whose activity name isn't part of *any* relation is a guaranteed no-op, cheaply skipped.
- **`_is_commit_ready(event)`** — only `status/lifecycle:transition == "complete"` (or absent any negative marker) counts as commit-ready; `start`/`error`/`declined`/`planned` never commit.
- **`_is_enabled(event_id, graph)`** — `DataSemantics.enabled(graph)` for a `DataDcrGraph`, else base `DcrSemantics.is_enabled(...)`.
- **`_compute_data_closure(candidate_event_id, graph)`** — walks *backward* from the candidate over the four "gating" relation keys (`conditionsFor`, `guardedConditions`, `milestonesFor`, `guardedMilestones`), collecting every upstream node that is an input or decision event — **but not recursing into void sources**, which must still be genuinely executed by real agent activity, never auto-resolved.
- **`_resolve_data_events(graph, candidate_event_id, event)`** — the auto-execution loop (detailed in §3.6/diagram 5). No-op if no resolver is configured or the graph isn't data-aware — a resolver-less setup is byte-identical to pre-hook behavior.
- **`validate_planned_event(candidate_event)`** — operates on a **fresh deep copy** of `_committed_graph` (`replay_graph`), **never mutates committed state**. Runs `_resolve_data_events` then checks `_is_enabled`; on failure, returns a `ConformanceCheckResult` carrying a human-readable blocker string from `_collect_enablement_blockers`.
- **`on_trace_event_published(event)`** — dedupes by `id(event)`; resolves the activity name to an event id; unknown/irrelevant events and non-commit-ready events are silent no-ops; runs `_resolve_data_events` against **`_committed_graph` this time**; re-checks `_is_enabled` (raising `ValueError` if somehow still not enabled — a defensive check, since `validate_planned_event` should already have blocked this earlier); then actually executes and records the event.

### 3.3 `DCRStateValidator`

File: `src/thesis_dpm_secure_langgraph/validation/dcr_state_validator.py` — a thin composition wrapper, not a subclass:

```python
self._state_tracker = constraints.get_state_tracker()
self.trace_event_subscriber = self._state_tracker   # hand-off point, see §3.4
```

`validate(tool_call, trace=None) → (ValidationDecision, str | None)`: builds a candidate `Event` from the tool-call dict (`status="planned"`), calls `validate_planned_event`, and maps an empty `violated_indices` to `ALLOW` / anything else to `DECLINE` with text built via `get_violation_messages`. `ValidationDecision` also has an `APPROVE` value (human-in-the-loop path) that the DCR validator never itself returns.

### 3.4 LangGraph integration — the full tool-call lifecycle

Files: `langgraph_integration/secure_state_graph.py`, `secure_tool_node.py`, `callbacks/trace_collector.py`.

- **`TraceCollector`** is a LangChain `BaseCallbackHandler`. It appends events to a running `Trace` on `on_tool_start`/`on_tool_end`/`on_tool_error`, and **fans every appended/updated event out** to registered `TraceEventSubscriber`s via `_publish_trace_event`.
- **`SecureStateGraph.add_node`** transparently upgrades any plain LangGraph `ToolNode` into a `SecureToolNode` (unless `secure=False`), wiring in the graph's `trace_collector`/`validator`.
- **`SecureToolNode.__init__`** registers `validator.trace_event_subscriber` (the `DCRStateTracker`) with the `trace_collector`, so every published event reaches `on_trace_event_published`.
- **`SecureToolNode.__call__`**, per proposed tool call: `validator.validate(tool_call, trace)` — this is the **pre-execution gate**. `DECLINE` short-circuits with a synthetic `ToolMessage`; the real tool function never runs, and (because nothing was ever appended to the trace) `on_trace_event_published` is never invoked for it — a declined call leaves *zero* trace on the DCR graph. `ALLOW` proceeds to `ToolNode.invoke(...)`, which runs the actual tool under LangChain's callback machinery — `on_tool_start` fires first (published, but filtered out downstream since it isn't commit-ready), the tool runs, then `on_tool_end`/`on_tool_error` fires with the real result attached to the *same* `Event` object, and this publish is what finally drives `on_trace_event_published` with a genuinely complete (or errored) event.

```mermaid
sequenceDiagram
    participant LLM
    participant STN as SecureToolNode (LangGraph)
    participant DSV as DCRStateValidator
    participant DST as DCRStateTracker
    participant Stub as Stub tool (LangGraph)
    participant SLA as SecureLangGraphAdapter
    participant Orch as tau2 Environment
    participant Tool as AirlineTools / FlightDB
    participant TC as TraceCollector

    LLM->>STN: proposes tool_call
    STN->>DSV: validate(tool_call, trace)
    DSV->>DST: validate_planned_event(candidate_event)
    Note over DST: replay on a throwaway deep copy of _committed_graph;<br/>_resolve_data_events fills the data closure first
    alt DECLINE (not enabled)
        DST-->>DSV: violated_indices=[0] + diagnostic
        DSV-->>STN: DECLINE, message
        STN-->>LLM: synthetic decline ToolMessage (nothing executes, no trace)
    else ALLOW
        DST-->>DSV: violated_indices=[]
        DSV-->>STN: ALLOW
        STN->>Stub: invoke stub tool (schema-only, returns marker)
        Stub-->>SLA: hand off real ToolCall to tau2 orchestrator
        SLA->>Orch: (tau2's own turn) run the REAL tool
        Orch->>Tool: execute against FlightDB (validate fully, then mutate)
        Tool-->>Orch: result, or a raised exception on failure
        Orch-->>SLA: ToolMessage(error=True/False)
        SLA->>SLA: _record_completed_tool_call: build completion Event,<br/>status = "error" or "complete"
        SLA->>TC: _publish_trace_event(event)  (explicit call — see §4.2)
        TC->>DST: on_trace_event_published(event)
        alt status == "error"
            DST-->>DST: _is_commit_ready() == False → no-op, graph unchanged
        else status == "complete"
            DST->>DST: _resolve_data_events (against _committed_graph, for real)
            DST->>DST: _execute → DataSemantics.execute mutates _committed_graph
        end
    end
```

Note the two-phase execution: `SecureToolNode`'s "tools" node only ever runs a **stub** tool (schema-only, returns a marker) to keep validation inside the LangGraph subgraph fast and side-effect-free. The **real** tool execution happens outside LangGraph entirely, in tau2's own orchestrator/environment, and its result is stitched back in on the *next* turn — see §4.2.

### 3.5 The in-package DCR-YAML compiler

Files: `constraints/dcr_yaml_types.py` (dataclass AST: `DcrEventDecl`, `EventKind{VOID,INPUT,COMPUTED}`, `DcrRelation`, `DcrGuardedRelation`, `DcrMarking`, `DcrYamlAst`), `dcr_yaml_parser.py` (`parse_dcr_yaml`/`parse_dcr_yaml_file`, validates every referenced id is known, raises `DcrYamlParseError` otherwise), `dcr_yaml_compiler.py` (`compile_to_graph` → `DataDcrGraph` directly in memory, `compile_to_xml` → an `XML_DCR_DATA` string, `extract_descriptions`).

Crucially, `compile_to_graph`/`compile_to_xml` run every `expr:`/`guard` string through pm4py's own `parse_expression`/`parse_guard` **at compile time** — a malformed expression fails fast as a `DcrYamlParseError`, before ever reaching the engine. `AgentDCRConstraints.parse_from_yaml` (§3.1) uses `compile_to_graph` directly (no XML round-trip at all).

**This is a separate implementation from `[tau2-bench-thesis]`'s `scripts/compile_dcr_yaml.py`** (§4.3) — the two are not connected by imports; only the tau2 script is used to produce `dcr1.xml`/`dcr2.xml`.

### 3.6 The data-event resolver contract

File: `constraints/data_resolver.py`:

```python
UNRESOLVED = _Unresolved()                              # sentinel: "no answer yet"
DataEventResolver = Callable[[str, Any, Any], Any]       # (event_id, event, graph) -> value | UNRESOLVED | None
```

Exported from the **top-level** package (`from thesis_dpm_secure_langgraph import UNRESOLVED, DataEventResolver`), not from the `constraints` subpackage.

**The core invariant, and why**: `_resolve_data_events` re-consults the resolver for **every** closure member on **every** call — even events that already executed earlier in the conversation. This is required because DCR events are **global, non-parameterized graph nodes**: there is exactly one `reservation_has_flown` node in the whole graph, shared across every reservation the agent might ever look at in that conversation. If the tracker trusted a stale value from checking reservation A, it would silently apply that same stale answer to a later candidate action on reservation B. Always re-resolving fresh against the *current* candidate's own arguments is what keeps a shared node correct across different logical "instances."

```mermaid
flowchart TD
    A["_resolve_data_events(graph, candidate_event_id, event)"] --> B{"resolver set AND graph is data-aware?"}
    B -- no --> Z["no-op (identical to having no resolver at all)"]
    B -- yes --> C["closure = _compute_data_closure(candidate_event_id, graph)"]
    C --> D{"closure empty?"}
    D -- yes --> Z
    D -- no --> E["pending = closure"]
    E --> F["for each event_id in pending"]
    F --> G{"is_input_event(event_id)?"}
    G -- yes --> H["value = data_event_resolver(event_id, event, graph)"]
    H --> I{"value is UNRESOLVED or None?"}
    I -- yes --> J["keep in still_pending, try again next pass"]
    I -- no --> K["DataSemantics.execute(graph, event_id, input_value=value)"]
    G -- no, decision event --> L["try: DataSemantics.execute(graph, event_id)"]
    L --> M{"raised ValueError / KeyError / TypeError?"}
    M -- yes, deps not ready --> J
    M -- no --> K
    K --> N["mark progress"]
    F --> O{"pass over all pending done?"}
    O -- yes --> P{"still_pending empty, or no progress this pass?"}
    P -- no --> F
    P -- yes --> Q["stop (bounded by len(closure) passes total)"]
```

This is a fixpoint loop because closure members can depend on each other in an arbitrary order — e.g. `booking_num_passengers` (input) must resolve before `booking_params_valid` (computed, reads it) can execute, which in turn must resolve before `book_reservation_valid` (computed, reads that) can execute. Any exception raised *by the resolver itself* is caught and treated as `UNRESOLVED`, so a resolver bug degrades to "gate stays closed," never a crash.

---

## 4. The domain layer `[tau2-bench-thesis]`

### 4.1 `SecureAirlineAgent` — policy routing

File: `src/tau2/agent/secure_airline_agent.py`. `TAU2_AIRLINE_POLICY_PATH` (default `policy_v3.yaml`) selects the active policy by **file suffix**:

- **`.xml`** → DCR path. Sniffs the raw file text for `dataType="` (an XML *attribute*, never an element tag — a check like `"<dataType" in xml_text` can never match anything real, a bug fixed this session). If present, loads a sibling `<stem>_data_resolver.py` (via `_load_data_event_resolver`, same dynamic-file-import idiom as Declare predicate files) and calls `AgentDCRConstraints().parse_data_from_file(path, data_event_resolver=resolve_fn)`; otherwise `parse_from_file` (plain `DcrGraph`).
- **anything else** → Declare/MP-Declare path via `AgentDeclareConstraints`, with a sibling predicate file resolved by the naming convention `stem.replace("policy", "predicates", 1)`.

Directory inventory, `data/tau2/domains/airline/security/`:

| Family | Source | Predicate/resolver module | Compiled/derived artifact |
|---|---|---|---|
| DCR (void) | `dcr1.yaml` | — | `dcr1.xml` (0 `dataType=` occurrences) |
| DCR (data-aware) | `dcr2.yaml` | `dcr2_data_resolver.py` → `dcr2_predicates.py` | `dcr2.xml` (24 `dataType=` occurrences) |
| Declare | `policy.decl` | `predicates.py` | `policy_mp.decl` *(auto-generated cache, not hand-authored)* |
| Declare | `policy_v2.yaml` | `predicates_v2.py` | `policy_v2_mp.decl` |
| Declare | `policy_v3.yaml` | `predicates_v3.py` | `policy_v3_mp.decl` |
| Declare | `policy_v4.yaml` | `predicates_v4.py` | `policy_v4_mp.decl` |

The `*_mp.decl` files are a side effect of every `AgentDeclareConstraints.parse_from_file` call (it compiles the source to MP-Declare and writes it out) — not an input anyone hand-edits.

### 4.2 `SecureLangGraphAdapter` — bridging LangGraph and the real orchestrator

File: `src/tau2/agent/secure_langgraph_adapter.py`.

- **Stub tools** (`_create_stub_tools`): every real `AirlineTools` method is wrapped in a `StructuredTool` whose body just returns a `[VALIDATED: delegated to tau2]` marker. These exist purely so `SecureToolNode` has real argument schemas to validate against inside the LangGraph subgraph — **no actual DB mutation happens there.**
- **Real execution happens entirely outside LangGraph**, in tau2's own `Environment.get_response` (`src/tau2/environment/environment.py`), called from the tau2 orchestrator on a later turn.
- **`_record_completed_tool_call`** is the bridge back: once tau2 hands the adapter a real `ToolMessage` (with `error=True/False`), it builds a completion `Event` (`status="error"` or `"complete"`) and — critically — calls `trace_collector._publish_trace_event(event)` **explicitly**. A plain `.append()` onto the trace would bypass subscriber fan-out entirely; without this explicit call, `DCRStateTracker` would never learn the real tool result, and would keep declining anything gated behind that action having completed.

### 4.3 `scripts/compile_dcr_yaml.py`

Standalone, dependency-free CLI script — **does not import `thesis_dpm_secure_langgraph`** — reimplementing the same YAML→`XML_DCR_DATA` mapping as §3.5's in-package compiler:

- Classifies each `events:` entry as void / input (`type` + no `expr`) / computed (`type` + `expr`), emitting `dataType`/`decision` attributes accordingly (identical rules to §2.7).
- Maps `conditions`/`responses`/`excludes`/`includes`/`milestones`/`co_responses` (unguarded, `{source: [targets]}`) and `guarded_conditions`/…/`guarded_no_responses` (guarded, `[[source, target, guard], ...]`) to the matching XML sections — including preserving pm4py's own `coresponces`/`coresponse` tag misspelling for compatibility.
- **No AST, no pre-flight expression validation** — unlike §3.5, a malformed expression here is only caught later, when pm4py actually imports the resulting XML.

Used as: `python scripts/compile_dcr_yaml.py data/tau2/domains/airline/security/dcr2.yaml` (regenerates `dcr2.xml`).

### 4.4 dcr1 vs dcr2

`dcr1.yaml` is explicitly documented in its own header comment as "standard DCR (no data events, no guarded relations) — all events void, all relations unguarded; encodable constraints are sequencing/precedence rules only." Confirmed: `dcr1.xml` has zero `dataType=` occurrences, so it always loads as a plain `DcrGraph`, and the whole resolver mechanism in §3.2/§3.6 is a no-op for it regardless of configuration.

`dcr2.yaml`/`dcr2.xml` is the data-aware policy: 24 `dataType=` attributes, input and computed events, and `guardedIncludes`/`guardedExcludes` gating every write tool (`book_reservation`, `cancel_reservation`, `update_reservation_*`, `send_certificate`) behind those events' resolved values.

### 4.5 `dcr2_data_resolver.py` + `dcr2_predicates.py` — a concrete `DataEventResolver`

Two files, deliberately separated by concern:

- **`dcr2_data_resolver.py`** owns all the "is this even available yet" plumbing: extracting arguments off the pending tool-call `Event`, looking up `FlightDB` entities, and deciding `UNRESOLVED` when something (a reservation, a user) doesn't exist yet. A `_HANDLERS: dict[event_id, handler]` table dispatches per input event; the top-level `resolve(event_id, event, graph)` wraps every handler call in a `try/except → UNRESOLVED`, so a resolver bug can never crash validation, only leave a gate closed.
- **`dcr2_predicates.py`** is pure business logic over **already-looked-up** domain objects (`Reservation`, `User`, `FlightDB`) — no DB access, no fail-open defaults, because it's only ever called by the resolver once presence has already been confirmed. This is a deliberate correction of `predicates_v4.py` (the Declare-path predicate file, reused by dcr2's resolver in an earlier iteration): that file is fail-*open* by design (e.g. `cancellation_eligible` returns `True` when the DB is unreachable — correct for its original "flag only what's provably wrong" Declare semantics), which is exactly backwards for a DCR gate, where "no answer" must leave the gate blocked. `dcr2_predicates.valid_certificate_amount`, for instance, returns `False` (not `True`) when a user has no disrupted reservations to check against at all.

**The "combined gate" pattern** — used for all four write-tool gates in dcr2.yaml (`book_reservation_valid`, `cancel_reservation_valid`, `update_reservation_flights_valid`, `send_certificate_valid`):

```mermaid
flowchart LR
    BPV["booking_params_valid (computed)<br/>≤5 passengers, ≤1 credit card,<br/>≤3 gift cards, ≤1 certificate"]
    BPMV["booking_payment_methods_valid (input)<br/>all payment methods in user profile"]
    BRV["book_reservation_valid (computed)<br/>expr: booking_params_valid AND booking_payment_methods_valid"]
    GE["guarded_excludes → book_reservation<br/>guard: NOT book_reservation_valid"]
    GI["guarded_includes → book_reservation<br/>guard: book_reservation_valid"]
    BR["book_reservation (gated write tool)"]

    BPV --> BRV
    BPMV --> BRV
    BRV --> GE --> BR
    BRV --> GI --> BR
```

Every check that gates a given write action is combined into **one** computed event, which then drives **exactly one** `guarded_excludes`/`guarded_includes` pair. This matters because `DataSemantics.execute` applies guarded relations independently, in event-execution order (§2.4) — if `book_reservation` instead had *two separate* guard pairs (one sourced from `booking_params_valid`, one from `booking_payment_methods_valid`), whichever one happened to resolve/execute *last* would silently overwrite the other's effect on `included`, since both write to the same target set. That "last writer wins" composition bug (fixed this session) is exactly what the combined-gate pattern above structurally prevents: with a single AND'd gate, there is only ever one relation touching `book_reservation`'s inclusion state.

### 4.6 The commit-ordering guarantee

`AirlineTools` methods (`src/tau2/domains/airline/tools.py`) validate fully **before** mutating `FlightDB` — e.g. `book_reservation` checks flight availability, seat counts, payment-method existence, and that the payment total matches the price, all before the first `self.db.reservations[...] = ...` write. A failure (like `"Payment amount does not add up..."`) raises before any mutation.

`Environment.get_response` catches any such exception and returns `ToolMessage(error=True, ...)` instead of propagating it. Back in `[thesis-dpm-secure-langgraph]`, `_is_commit_ready` (§3.2) treats `status == "error"` as **not** commit-ready — so `on_trace_event_published` returns immediately, and `DataSemantics.execute` never runs for that event. **The DCR graph's committed state can only ever advance on genuinely successful tool executions** — there is no path by which a failed/erroring tool call can leave a trace on `_committed_graph`.

---

## 5. End-to-end walkthrough

A real trace from this session's live-run testing (`cancel_reservation` on reservation `Z7GOZK`, task 19 of the airline benchmark), tracing through every layer:

1. **Agent proposes** `cancel_reservation(reservation_id="Z7GOZK", reason="health")` before having called `get_flight_status` at all.
2. `SecureToolNode` → `DCRStateValidator.validate` → `DCRStateTracker.validate_planned_event` on a replay copy of `_committed_graph`. `[thesis-dpm-secure-langgraph]`
3. `_resolve_data_events` computes the closure and resolves it fine (the resolver can answer `reservation_has_flown`/`reservation_cancellation_eligible_base` regardless of whether `get_flight_status` was called — see below) — but `_is_enabled` still fails, because `cancel_reservation` also has an **unguarded** `conditions:` prerequisite requiring the void event `get_flight_status` to have executed at all, independent of any data gate. `[pm4py-dcr]` `enabled()` / `[thesis-dpm-secure-langgraph]` `_collect_enablement_blockers`
4. **DECLINE**, message: `"...Ensure: (1) get_user_details, get_reservation_details, and get_flight_status called... [Diagnostic: 'cancel_reservation' is not enabled (unmet condition(s): get_flight_status)]"` — the static description from `dcr2.yaml`'s `events:` block concatenated with the dynamic blocker. `[thesis-dpm-secure-langgraph]` `get_violation_messages`
5. Agent calls `get_flight_status`, then retries `cancel_reservation`. This time the unguarded condition is satisfied.
6. `validate_planned_event` re-runs `_resolve_data_events`: the closure includes `reservation_has_flown` (input) and `reservation_cancellation_eligible_base`/`reservation_cancellation_eligible`/`cancel_reservation_valid` (computed). `dcr2_data_resolver.resolve` looks up `Z7GOZK` in `FlightDB` and answers `reservation_has_flown = False` (all flights `available`); `dcr2_predicates.cancellation_eligible_base` answers `True` (the reservation has `insurance: yes`, independent of cabin class). `[tau2-bench-thesis]`
7. Both computed events execute: `reservation_cancellation_eligible_base OR booking_within_24h → True`, then `(NOT reservation_has_flown) AND reservation_cancellation_eligible → True → cancel_reservation_valid`. `[pm4py-dcr]` `DataSemantics.execute`
8. `_is_enabled` now succeeds → **ALLOW**.
9. `SecureToolNode` runs the stub tool; control returns to the tau2 orchestrator, which runs the **real** `AirlineTools.cancel_reservation` against `FlightDB` (§4.2/§4.6) and returns a successful `ToolMessage`.
10. `_record_completed_tool_call` builds a `status="complete"` event and explicitly publishes it; `on_trace_event_published` re-resolves the closure against `_committed_graph` (for real this time), confirms enablement again, and calls `DataSemantics.execute` — the reservation is now genuinely cancelled in both the DB and the DCR graph's committed marking.

---

## 6. Glossary / quick reference

| Component | Purpose | Repo | File |
|---|---|---|---|
| `DcrGraph` / `Marking` | base graph structure + included/pending/executed sets | `pm4py-dcr` | `pm4py/objects/dcr/obj.py` |
| `DcrSemantics` | base `enabled()`/`execute()` | `pm4py-dcr` | `pm4py/objects/dcr/semantics.py` |
| `DataDcrGraph` | + event types, decisions, event values, 6 guarded relations, predicate registry | `pm4py-dcr` | `pm4py/objects/dcr/data/obj.py` |
| `DataSemantics` | guard-aware `enabled()`/`execute()` | `pm4py-dcr` | `pm4py/objects/dcr/data/semantics.py` |
| `Expression` / `Guard` AST | expression tree + string parser/serializer | `pm4py-dcr` | `pm4py/objects/dcr/data/expressions.py`, `expression_parser.py` |
| `predicate_registry` / `FunctionCallExpression` | in-expression function calls — **unused by dcr2** | `pm4py-dcr` | `pm4py/objects/dcr/data/predicate_loader.py` |
| `XML_DCR_DATA` importer/exporter | XML ⇄ `DataDcrGraph`, `cast_to_dcr_object` dispatch | `pm4py-dcr` | `pm4py/objects/dcr/importer/variants/xml_dcr_data.py` |
| `AgentDCRConstraints` | parses a policy file, builds the tracker | `thesis-dpm-secure-langgraph` | `constraints/agent_dcr_constraints.py` |
| `DCRStateTracker` | validate/commit split, data-event resolver hook | `thesis-dpm-secure-langgraph` | `constraints/dcr_state.py` |
| `DataEventResolver` / `UNRESOLVED` | resolver callback contract | `thesis-dpm-secure-langgraph` | `constraints/data_resolver.py` |
| `DCRStateValidator` | thin `validate()` wrapper, `ValidationDecision` | `thesis-dpm-secure-langgraph` | `validation/dcr_state_validator.py` |
| `SecureStateGraph` / `SecureToolNode` | LangGraph tool-call gate | `thesis-dpm-secure-langgraph` | `langgraph_integration/*.py` |
| `TraceCollector` | LangChain callback handler, publishes to subscribers | `thesis-dpm-secure-langgraph` | `callbacks/trace_collector.py` |
| `dcr_yaml_parser` / `dcr_yaml_compiler` | validated DCR-YAML → graph/XML compiler | `thesis-dpm-secure-langgraph` | `constraints/dcr_yaml_*.py` |
| `SecureAirlineAgent` | policy routing (`.xml` DCR vs Declare) | `tau2-bench-thesis` | `src/tau2/agent/secure_airline_agent.py` |
| `SecureLangGraphAdapter` | LangGraph ↔ tau2 orchestrator bridge | `tau2-bench-thesis` | `src/tau2/agent/secure_langgraph_adapter.py` |
| `dcr2.yaml` / `dcr2.xml` | the airline domain's data-aware policy | `tau2-bench-thesis` | `data/tau2/domains/airline/security/` |
| `dcr2_data_resolver.py` | `DataEventResolver` implementation (DB plumbing) | `tau2-bench-thesis` | `data/tau2/domains/airline/security/dcr2_data_resolver.py` |
| `dcr2_predicates.py` | fail-closed pure predicate logic | `tau2-bench-thesis` | `data/tau2/domains/airline/security/dcr2_predicates.py` |
| `scripts/compile_dcr_yaml.py` | standalone, unvalidated YAML → XML compiler | `tau2-bench-thesis` | `scripts/compile_dcr_yaml.py` |
| `AirlineTools` / `FlightDB` | real domain mutation, validate-then-mutate | `tau2-bench-thesis` | `src/tau2/domains/airline/tools.py` |
