"""Tests for the SecureLangGraphAdapter and SecureAirlineAgent.

Covers:
- Policy compilation
- Agent registration
- Allowed tool call produces validation_check event + AssistantMessage with tool_calls
- Declined tool call produces validation_check event + text fallback + record_declined
- Multi-tool-call rejection
- stop() records agent_end
- INTEGRATION: declined tool call is never executed by tau2 environment
- INTEGRATION: allowed tool call executes exactly once (no double execution)
"""

import uuid
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage
from pm4py.objects.log.obj import Trace
from thesis_dpm_secure_langgraph import (
    AgentDeclareConstraints,
    DeclareTraceValidator,
    TraceCollector,
    ValidationDecision,
)

from tau2.agent.secure_langgraph_adapter import SecureLangGraphAdapter, _DECLINE_PREFIX
from tau2.agent.secure_airline_agent import (
    AGENT_INSTRUCTION as SECURE_AGENT_INSTRUCTION,
    SOFT_AGENT_INSTRUCTION as SECURE_SOFT_AGENT_INSTRUCTION,
    build_secure_system_prompt,
)
from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.environment import Environment
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.run import get_tasks
from tau2.user.user_simulator import UserSimulator, UserState


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeModel:
    """Returns pre-scripted AIMessages in sequence."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        if not self.outputs:
            return AIMessage(content="No output")
        return self.outputs.pop(0)


class _FakeValidator:
    """Returns pre-configured decisions by tool name."""

    def __init__(self, decisions):
        self.decisions = decisions
        self.validate_calls = []

    def validate(self, tool_call, trace):
        self.validate_calls.append(tool_call)
        return self.decisions.get(
            tool_call["name"], (ValidationDecision.ALLOW, None)
        )


class _FakeTraceCollector:
    def __init__(self):
        self.trace = Trace()
        self.declined = []
        self._agent_end_calls = []

    def get_trace(self):
        return self.trace

    def record_declined(self, action, error_message):
        self.declined.append((action, error_message))

    def record_agent_end(self, *, decided_by="", content=""):
        self._agent_end_calls.append({"decided_by": decided_by, "content": content})


class _StopUser(UserSimulator):
    """User that sends a request on the first turn, then ###STOP### after."""

    def __init__(self, first_message="Please help me", **kwargs):
        self._first_message = first_message
        self._turn = 0

    def get_init_state(self, message_history=None):
        return UserState(messages=[], system_messages=[])

    def generate_next_message(self, message, state):
        self._turn += 1
        if self._turn == 1:
            return UserMessage(role="user", content=self._first_message), state
        return UserMessage(role="user", content="###STOP###"), state

    def set_seed(self, seed):
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(model, validator, collector, tau2_tools=None):
    """Build a SecureLangGraphAdapter with fakes."""
    from tau2.environment.tool import Tool

    if tau2_tools is None:
        # Minimal stub tau2 tool
        def get_user_details(user_id: str) -> str:
            """Get user details by ID."""
            return f"details for {user_id}"

        def book_reservation(user_id: str) -> str:
            """Book a reservation."""
            return f"booked for {user_id}"

        tau2_tools = [
            Tool(get_user_details),
            Tool(book_reservation),
        ]

    return SecureLangGraphAdapter(
        model=model,
        system_prompt="You are a test agent.",
        tools=tau2_tools,
        domain_policy="test policy",
        validator=validator,
        trace_collector=collector,
        max_internal_retries=1,
    )


def _find_events(trace, concept_name):
    """Return all events matching a concept:name."""
    return [e for e in trace if e.get("concept:name") == concept_name]


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_policy_decl_compiles():
    decl_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "tau2"
        / "domains"
        / "airline"
        / "security"
        / "policy.decl"
    )
    constraints = AgentDeclareConstraints().parse_from_file(str(decl_path))
    assert constraints.to_declare_model() is not None


def test_secure_airline_agent_registered():
    assert "secure_airline_agent" in registry.get_agents()


def test_build_secure_system_prompt_default_instruction():
    prompt = build_secure_system_prompt(
        domain_policy="policy text",
        soft_agent=False,
    )
    assert SECURE_AGENT_INSTRUCTION in prompt
    assert SECURE_SOFT_AGENT_INSTRUCTION not in prompt
    assert "<instructions>" in prompt
    assert "<policy>" in prompt


def test_build_secure_system_prompt_soft_instruction():
    prompt = build_secure_system_prompt(
        domain_policy="policy text",
        soft_agent=True,
    )
    assert SECURE_SOFT_AGENT_INSTRUCTION in prompt
    assert SECURE_AGENT_INSTRUCTION not in prompt
    assert "<instructions>" in prompt
    assert "<policy>" in prompt


def test_allowed_tool_call_produces_validation_event_and_tool_calls():
    model = _FakeModel([
        AIMessage(
            content="",
            tool_calls=[{
                "id": "tc1",
                "name": "get_user_details",
                "args": {"user_id": "u1"},
                "type": "tool_call",
            }],
        ),
    ])
    validator = _FakeValidator({})  # default: ALLOW everything
    collector = _FakeTraceCollector()
    agent = _make_agent(model, validator, collector)

    state = agent.get_init_state()
    out, _ = agent.generate_next_message(
        UserMessage(role="user", content="hello"), state
    )

    # Agent should return tool_calls to tau2
    assert out.tool_calls is not None
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "get_user_details"

    # Trace should contain a validation_check with decision=allow
    checks = _find_events(collector.trace, "validation_check")
    assert len(checks) == 1
    assert checks[0]["decision"] == "allow"
    assert checks[0]["tool_name"] == "get_user_details"
    assert checks[0]["violations"] is None

    # Validator was actually called
    assert len(validator.validate_calls) == 1


def test_declined_tool_call_produces_validation_event_and_text_fallback():
    model = _FakeModel([
        AIMessage(
            content="",
            tool_calls=[{
                "id": "tc1",
                "name": "book_reservation",
                "args": {"user_id": "u1"},
                "type": "tool_call",
            }],
        ),
        # After decline, LLM retries and gives text
        AIMessage(content="I cannot do that per policy."),
    ])
    validator = _FakeValidator({
        "book_reservation": (ValidationDecision.DECLINE, "blocked by policy"),
    })
    collector = _FakeTraceCollector()
    agent = _make_agent(model, validator, collector)

    state = agent.get_init_state()
    out, _ = agent.generate_next_message(
        UserMessage(role="user", content="book"), state
    )

    # Should return text (no tool_calls)
    assert out.tool_calls is None
    assert out.content is not None

    # Trace should contain a validation_check with decision=decline
    checks = _find_events(collector.trace, "validation_check")
    assert any(c["decision"] == "decline" for c in checks)

    # record_declined was called
    assert len(collector.declined) >= 1
    assert collector.declined[0][0] == "book_reservation"


def test_multi_tool_call_rejected_with_retry():
    model = _FakeModel([
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "get_user_details",
                    "args": {"user_id": "u1"},
                    "type": "tool_call",
                },
                {
                    "id": "tc2",
                    "name": "book_reservation",
                    "args": {"user_id": "u1"},
                    "type": "tool_call",
                },
            ],
        ),
        # After rejection, LLM sends single call
        AIMessage(
            content="",
            tool_calls=[{
                "id": "tc3",
                "name": "get_user_details",
                "args": {"user_id": "u1"},
                "type": "tool_call",
            }],
        ),
    ])
    validator = _FakeValidator({})  # ALLOW everything
    collector = _FakeTraceCollector()
    agent = _make_agent(model, validator, collector)

    state = agent.get_init_state()
    out, _ = agent.generate_next_message(
        UserMessage(role="user", content="do both"), state
    )

    # Should succeed with the single retried call
    assert out.tool_calls is not None
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "get_user_details"

    # Model was called twice (original + retry)
    assert model.call_count >= 2


def test_stop_records_agent_end():
    model = _FakeModel([])
    validator = _FakeValidator({})
    collector = _FakeTraceCollector()
    agent = _make_agent(model, validator, collector)

    agent.stop(message=UserMessage(role="user", content="bye"))

    assert len(collector._agent_end_calls) == 1
    assert collector._agent_end_calls[0]["content"] == "tau2 orchestrator stop"


# ---------------------------------------------------------------------------
# Integration tests: safety invariants
# ---------------------------------------------------------------------------


def test_declined_tool_call_never_executed_by_tau2():
    """A declined tool call must NEVER reach tau2's environment for execution."""
    # Setup: agent tries book_reservation (DECLINED), then gives text fallback
    model = _FakeModel([
        AIMessage(
            content="",
            tool_calls=[{
                "id": "tc1",
                "name": "book_reservation",
                "args": {"user_id": "u1"},
                "type": "tool_call",
            }],
        ),
        AIMessage(content="Sorry, I cannot do that."),
    ])
    validator = _FakeValidator({
        "book_reservation": (
            ValidationDecision.DECLINE,
            "Precedence constraint violated",
        ),
    })
    collector = _FakeTraceCollector()

    # Use mock domain environment
    env_constructor = registry.get_env_constructor("mock")
    task = get_tasks("mock", task_ids=["create_task_1"])[0]
    env = env_constructor()
    tau2_tools = env.get_tools()

    agent = _make_agent(model, validator, collector, tau2_tools=tau2_tools)
    user = _StopUser(first_message="Book a reservation")

    orch = Orchestrator(
        domain="mock",
        agent=agent,
        user=user,
        environment=env,
        task=task,
        max_steps=10,
        max_errors=5,
        validate_communication=False,
    )

    with patch.object(env, "get_response", wraps=env.get_response) as spy:
        orch.run()

        # CRITICAL: environment.get_response was NEVER called
        assert spy.call_count == 0, (
            f"Environment.get_response was called {spy.call_count} time(s) "
            f"but should never be called when tool calls are declined. "
            f"Calls: {[str(c) for c in spy.call_args_list]}"
        )

    # Trajectory should contain the text fallback, not a ToolMessage
    trajectory = orch.get_trajectory()
    tool_messages = [m for m in trajectory if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 0, (
        f"Found {len(tool_messages)} ToolMessage(s) in trajectory — "
        "declined calls should produce no ToolMessages"
    )

    # There should be an AssistantMessage with text content
    agent_msgs = [
        m for m in trajectory
        if isinstance(m, AssistantMessage) and m.has_text_content()
    ]
    assert len(agent_msgs) >= 1, "Expected a text fallback from the agent"


def test_allowed_tool_call_executes_exactly_once():
    """An allowed tool call must be executed exactly once by tau2, not twice
    (once in the graph stub + once in tau2 environment)."""
    # Setup: agent calls create_task (ALLOWED)
    model = _FakeModel([
        AIMessage(
            content="",
            tool_calls=[{
                "id": "tc1",
                "name": "create_task",
                "args": {
                    "user_id": "user_1",
                    "title": "Test Task",
                },
                "type": "tool_call",
            }],
        ),
        # After getting tool result, agent responds with text
        AIMessage(content="Task created successfully!"),
    ])
    validator = _FakeValidator({})  # ALLOW everything
    collector = _FakeTraceCollector()

    env_constructor = registry.get_env_constructor("mock")
    task = get_tasks("mock", task_ids=["create_task_1"])[0]
    env = env_constructor()
    tau2_tools = env.get_tools()

    agent = _make_agent(model, validator, collector, tau2_tools=tau2_tools)
    user = _StopUser(first_message="Create a task for me")

    orch = Orchestrator(
        domain="mock",
        agent=agent,
        user=user,
        environment=env,
        task=task,
        max_steps=10,
        max_errors=5,
        validate_communication=False,
    )

    with patch.object(env, "get_response", wraps=env.get_response) as spy:
        orch.run()

        # CRITICAL: environment.get_response was called EXACTLY ONCE
        assert spy.call_count == 1, (
            f"Environment.get_response was called {spy.call_count} time(s) "
            f"but should be called exactly once for an allowed tool call"
        )

        # The call was for the correct tool
        actual_call = spy.call_args_list[0]
        tool_call_arg = actual_call[0][0]  # first positional arg
        assert tool_call_arg.name == "create_task"

    # Trajectory should contain a ToolMessage with REAL output (not the stub marker)
    trajectory = orch.get_trajectory()
    tool_messages = [m for m in trajectory if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1, (
        f"Expected exactly 1 ToolMessage, got {len(tool_messages)}"
    )
    assert "[VALIDATED: delegated to tau2]" not in tool_messages[0].content, (
        "ToolMessage contains the stub marker — tool was not executed by tau2"
    )


# ---------------------------------------------------------------------------
# Binary Declare constraint tests (real validator)
# ---------------------------------------------------------------------------


def _make_real_agent(policy_text, tmp_path, model, tools_funcs, max_retries=1):
    """Build a SecureLangGraphAdapter with a real DeclareTraceValidator."""
    from tau2.environment.tool import Tool

    decl_file = tmp_path / "test.decl"
    decl_file.write_text(policy_text)

    constraints = AgentDeclareConstraints().parse_from_file(str(decl_file))
    validator = DeclareTraceValidator(constraints)
    collector = TraceCollector(trace_id=uuid.uuid4())
    tau2_tools = [Tool(fn) for fn in tools_funcs]

    agent = SecureLangGraphAdapter(
        model=model,
        system_prompt="Test agent.",
        tools=tau2_tools,
        domain_policy="test",
        validator=validator,
        trace_collector=collector,
        max_internal_retries=max_retries,
    )
    return agent, collector


def test_not_response_b_allowed_before_a_declined_after_a(tmp_path):
    """NotResponse(a,b): b is allowed before a executes, but declined after."""

    policy = (
        "activity2 escalate\n"
        "activity2 modify_order\n"
        "bind escalate: reason |\n"
        "bind modify_order: order_id |\n"
        "\n"
        "NotResponse[escalate, modify_order] | | |\n"
        '"Cannot modify orders after escalation."\n'
    )

    def escalate(reason: str) -> str:
        """Escalate to supervisor."""
        return "escalated"

    def modify_order(order_id: str) -> str:
        """Modify an order."""
        return "modified"

    # Turn 1: modify_order → ALLOWED (no escalation yet)
    # Turn 2: escalate → ALLOWED
    # Turn 3: modify_order → DECLINED → text fallback
    model = _FakeModel([
        AIMessage(
            content="",
            tool_calls=[{
                "id": "tc1", "name": "modify_order",
                "args": {"order_id": "o1"}, "type": "tool_call",
            }],
        ),
        AIMessage(
            content="",
            tool_calls=[{
                "id": "tc2", "name": "escalate",
                "args": {"reason": "customer request"}, "type": "tool_call",
            }],
        ),
        AIMessage(
            content="",
            tool_calls=[{
                "id": "tc3", "name": "modify_order",
                "args": {"order_id": "o2"}, "type": "tool_call",
            }],
        ),
        AIMessage(content="Cannot modify after escalation."),
    ])

    agent, collector = _make_real_agent(
        policy, tmp_path, model, [escalate, modify_order],
    )
    state = agent.get_init_state()

    # Step 1: modify_order → ALLOWED
    out1, state = agent.generate_next_message(
        UserMessage(role="user", content="modify order"), state,
    )
    assert out1.tool_calls is not None, (
        f"modify_order should be allowed before escalate, got: {out1.content}"
    )
    assert out1.tool_calls[0].name == "modify_order"

    # Feed tau2 tool result back
    out2, state = agent.generate_next_message(
        ToolMessage(id="tc1", role="tool", content="modified"), state,
    )
    assert out2.tool_calls is not None
    assert out2.tool_calls[0].name == "escalate"

    # Feed escalate result back
    out3, state = agent.generate_next_message(
        ToolMessage(id="tc2", role="tool", content="escalated"), state,
    )
    # modify_order should now be DECLINED → text fallback
    assert out3.tool_calls is None, (
        f"modify_order should be declined after escalate, "
        f"got tool_calls: {[tc.name for tc in out3.tool_calls]}"
    )
    assert out3.content is not None

    # Verify trace: allow(modify_order), allow(escalate), decline(modify_order)
    checks = _find_events(collector.get_trace(), "validation_check")
    allow_modify = [
        c for c in checks
        if c["tool_name"] == "modify_order" and c["decision"] == "allow"
    ]
    decline_modify = [
        c for c in checks
        if c["tool_name"] == "modify_order" and c["decision"] == "decline"
    ]
    assert len(allow_modify) >= 1, "modify_order should have been allowed once"
    assert len(decline_modify) >= 1, (
        "modify_order should have been declined after escalate"
    )


def test_not_response_with_argument_conditions(tmp_path):
    """NotResponse(a,b) with argument conditions on both activities.

    Constraint: NotResponse[escalate, modify_order]
        activation: A.priority == urgent  (only urgent escalations trigger)
        target:     T.change_type == critical  (only critical modifications blocked)

    The compiler converts == to 'is' for string-enum values.

    Flow:
    1. modify_order(critical) → ALLOWED (no urgent escalation yet)
    2. escalate(urgent) → ALLOWED
    3. modify_order(critical) → DECLINED (activation=urgent, target=critical)
       → retry: modify_order(minor) → ALLOWED (target condition ≠ critical)
    """

    policy = (
        "activity2 escalate\n"
        "activity2 modify_order\n"
        "bind escalate: priority, reason |\n"
        "bind modify_order: order_id, change_type |\n"
        "priority: normal, urgent\n"
        "change_type: minor, critical\n"
        "\n"
        "NotResponse[escalate, modify_order] "
        "| A.priority == urgent | T.change_type == critical |\n"
        '"Cannot make critical modifications after urgent escalation."\n'
    )

    def escalate(priority: str, reason: str) -> str:
        """Escalate to supervisor."""
        return "escalated"

    def modify_order(order_id: str, change_type: str) -> str:
        """Modify an order."""
        return "modified"

    model = _FakeModel([
        # 1) modify_order(critical) → ALLOWED (no escalation yet)
        AIMessage(
            content="",
            tool_calls=[{
                "id": "tc1", "name": "modify_order",
                "args": {"order_id": "o1", "change_type": "critical"},
                "type": "tool_call",
            }],
        ),
        # 2) escalate(urgent) → ALLOWED
        AIMessage(
            content="",
            tool_calls=[{
                "id": "tc2", "name": "escalate",
                "args": {"priority": "urgent", "reason": "policy"},
                "type": "tool_call",
            }],
        ),
        # 3a) modify_order(critical) → DECLINED
        AIMessage(
            content="",
            tool_calls=[{
                "id": "tc3", "name": "modify_order",
                "args": {"order_id": "o2", "change_type": "critical"},
                "type": "tool_call",
            }],
        ),
        # 3b) retry: modify_order(minor) → ALLOWED (target condition doesn't match)
        AIMessage(
            content="",
            tool_calls=[{
                "id": "tc4", "name": "modify_order",
                "args": {"order_id": "o2", "change_type": "minor"},
                "type": "tool_call",
            }],
        ),
    ])

    agent, collector = _make_real_agent(
        policy, tmp_path, model, [escalate, modify_order], max_retries=2,
    )
    state = agent.get_init_state()

    # Step 1: modify_order(critical) → ALLOWED (no escalation yet)
    out1, state = agent.generate_next_message(
        UserMessage(role="user", content="modify order"), state,
    )
    assert out1.tool_calls is not None, (
        f"modify_order should be allowed before escalate, got: {out1.content}"
    )
    assert out1.tool_calls[0].name == "modify_order"
    assert out1.tool_calls[0].arguments["change_type"] == "critical"

    # Step 2: escalate(urgent) → ALLOWED
    out2, state = agent.generate_next_message(
        ToolMessage(id="tc1", role="tool", content="modified"), state,
    )
    assert out2.tool_calls is not None
    assert out2.tool_calls[0].name == "escalate"

    # Step 3: modify_order(critical) → DECLINED → retry → modify_order(minor) → ALLOWED
    out3, state = agent.generate_next_message(
        ToolMessage(id="tc2", role="tool", content="escalated"), state,
    )
    assert out3.tool_calls is not None, (
        f"retry with change_type=minor should succeed, got text: {out3.content}"
    )
    assert out3.tool_calls[0].name == "modify_order"
    assert out3.tool_calls[0].arguments["change_type"] == "minor", (
        f"retry should use the non-matching condition value (minor), "
        f"got {out3.tool_calls[0].arguments['change_type']}"
    )

    # Verify trace events
    checks = _find_events(collector.get_trace(), "validation_check")
    allow_modify = [
        c for c in checks
        if c["tool_name"] == "modify_order" and c["decision"] == "allow"
    ]
    decline_modify = [
        c for c in checks
        if c["tool_name"] == "modify_order" and c["decision"] == "decline"
    ]
    # modify_order was allowed twice (critical before escalation, minor after)
    assert len(allow_modify) >= 2, (
        f"Expected ≥2 allowed modify_order checks, got {len(allow_modify)}"
    )
    # modify_order(critical) was declined once after urgent escalation
    assert len(decline_modify) >= 1, (
        "modify_order(critical) should be declined after urgent escalation"
    )
