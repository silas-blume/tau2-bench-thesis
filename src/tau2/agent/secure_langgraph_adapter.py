"""Secure LangGraph adapter for tau2 agents.

Uses SecureStateGraph to auto-upgrade a ToolNode to SecureToolNode, which
validates every tool call via a Declare/DCR validator before "executing" a
stub.  Approved calls are returned to the tau2 orchestrator for real
execution; declined calls feed back into the LLM for retry.
"""

from __future__ import annotations

import datetime
import json
import logging
from typing import Any, Optional

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage as LCSystemMessage,
    ToolMessage as LCToolMessage,
)
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, MessagesState
from langgraph.prebuilt import ToolNode
from pm4py.objects.log.obj import Event

_SECURE_LANGGRAPH_IMPORT_ERROR: Exception | None = None
SecureStateGraph: Any = None
ValidationDecision: Any = None
_coerce_numeric: Any = None
try:
    from thesis_dpm_secure_langgraph import SecureStateGraph, ValidationDecision
    from thesis_dpm_secure_langgraph.validation.validator import (
        _coerce_numeric as _coerce_numeric_impl,
    )

    _coerce_numeric = _coerce_numeric_impl
except Exception as exc:
    _SECURE_LANGGRAPH_IMPORT_ERROR = exc

from tau2.agent.base import (
    LocalAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
    validate_message_format_default,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

# Marker returned by stub tools so the adapter can distinguish allowed
# executions from declined ones.
_VALIDATED_MARKER = "[VALIDATED: delegated to tau2]"

# Prefix used by SecureToolNode when declining a tool call.
_DECLINE_PREFIX = "Tool call declined."

# Azure OpenAI rejects messages with more than 128 tool_calls. Cap well below
# that; the agent enforces single-call-per-turn anyway.
_MAX_TOOL_CALLS_PER_MSG = 8


def _create_stub_tools(tau2_tools: list[Tool]) -> list[StructuredTool]:
    """Create LangChain StructuredTools that match real tool schemas but only
    return a marker string.  These are used inside the graph so that
    SecureToolNode has concrete tools to validate against, while actual
    execution stays in tau2's environment."""
    stubs: list[StructuredTool] = []
    for tool in tau2_tools:

        def _stub(**kwargs: Any) -> str:  # noqa: ARG001
            return _VALIDATED_MARKER

        stubs.append(
            StructuredTool.from_function(
                func=_stub,
                name=tool.name,
                description=tool.short_desc or tool.name,
                args_schema=tool.params,
            )
        )
    return stubs


class SecureLangGraphAdapter(LocalAgent[list]):
    """Agent adapter that routes every tool call through SecureToolNode for
    validation before handing approved calls to tau2 for execution."""

    def __init__(
        self,
        model: Any,
        system_prompt: str,
        tools: list[Tool],
        domain_policy: str,
        validator: Any,
        trace_collector: Any,
        max_internal_retries: int = 4,
        secure_node_verbose: bool = False,
        secure_graph_logger: Any = None,
    ) -> None:
        if _SECURE_LANGGRAPH_IMPORT_ERROR is not None:
            raise ImportError(
                "SecureLangGraphAdapter requires thesis_dpm_secure_langgraph. "
                "Install it and retry."
            ) from _SECURE_LANGGRAPH_IMPORT_ERROR
        assert SecureStateGraph is not None
        assert ValidationDecision is not None

        super().__init__(tools=tools, domain_policy=domain_policy)

        self._bound_model = model
        self.system_prompt = system_prompt
        self.validator = validator
        self.trace_collector = trace_collector
        self.max_internal_retries = max_internal_retries
        self._logger = secure_graph_logger or logging.getLogger(__name__)

        # Track in-flight tool calls so we can record args when results come
        # back from tau2.
        self._pending_tool_calls: dict[str, dict[str, Any]] = {}

        # --- Build the LangGraph with SecureToolNode -----------------------
        stub_tools = _create_stub_tools(tools)

        graph = SecureStateGraph(
            MessagesState,
            trace_collector=self.trace_collector,
            validator=self.validator,
            secure_node_verbose=secure_node_verbose,
            logger=secure_graph_logger,
            max_tool_calls_per_message=_MAX_TOOL_CALLS_PER_MSG,
        )
        graph.add_node("agent", self._call_model_node, secure=False)
        # ToolNode is auto-upgraded to SecureToolNode by SecureStateGraph
        graph.add_node("tools", ToolNode(stub_tools))

        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {"tools": "tools", END: END},
        )
        graph.add_edge("tools", END)
        self._graph = graph.compile()

    # ------------------------------------------------------------------
    # Graph nodes / edges
    # ------------------------------------------------------------------

    def _call_model_node(self, state: MessagesState) -> dict:
        ai_msg: AIMessage = self._bound_model.invoke(state["messages"])
        return {"messages": [ai_msg]}

    @staticmethod
    def _should_continue(state: MessagesState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    @staticmethod
    def _pending_tool_call_ids(messages: list[Any]) -> set[str]:
        """Collect unresolved tool_call ids from the message stream."""
        pending_ids: set[str] = set()
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_id = tc.get("id")
                    if tc_id is not None:
                        pending_ids.add(str(tc_id))
            elif isinstance(msg, LCToolMessage):
                tc_id = msg.tool_call_id
                if tc_id in pending_ids:
                    pending_ids.remove(tc_id)
        return pending_ids

    def _append_tool_message_if_pending(
        self,
        state: list,
        tool_call_id: str,
        content: str,
    ) -> None:
        """Only append tool messages that correspond to unresolved tool calls."""
        if tool_call_id not in self._pending_tool_call_ids(state):
            self._logger.warning(
                "Dropping orphan tool-role message | tool_call_id=%s",
                tool_call_id,
            )
            return
        state.append(LCToolMessage(content=content, tool_call_id=tool_call_id))

    def _sanitize_tool_messages(self, state: list) -> list:
        """Drop invalid/duplicate tool messages before sending state to the model."""
        sanitized: list = []
        pending_ids: set[str] = set()

        for msg in state:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                # Cap tool_calls to avoid API limits (e.g. Azure max 128).
                if len(msg.tool_calls) > _MAX_TOOL_CALLS_PER_MSG:
                    self._logger.warning(
                        "Capping AIMessage tool_calls from %d to %d to stay within API limits",
                        len(msg.tool_calls),
                        _MAX_TOOL_CALLS_PER_MSG,
                    )
                    msg = AIMessage(
                        content=msg.content,
                        tool_calls=list(msg.tool_calls[:_MAX_TOOL_CALLS_PER_MSG]),
                        id=msg.id,
                    )
                sanitized.append(msg)
                for tc in msg.tool_calls:
                    tc_id = tc.get("id")
                    if tc_id is not None:
                        pending_ids.add(str(tc_id))
                continue

            if isinstance(msg, LCToolMessage):
                tc_id = msg.tool_call_id
                if tc_id in pending_ids:
                    sanitized.append(msg)
                    pending_ids.remove(tc_id)
                else:
                    self._logger.warning(
                        "Removing invalid tool-role message from state | tool_call_id=%s",
                        tc_id,
                    )
                continue

            sanitized.append(msg)

        return sanitized

    # ------------------------------------------------------------------
    # tau2 LocalAgent interface
    # ------------------------------------------------------------------

    def get_init_state(
        self,
        message_history: Optional[list[Message]] = None,
    ) -> list:
        state: list = [LCSystemMessage(content=self.system_prompt)]
        if not message_history:
            return state

        for msg in message_history:
            if not is_valid_agent_history_message(msg):
                continue

            if isinstance(msg, UserMessage):
                state.append(HumanMessage(content=msg.content or ""))
            elif isinstance(msg, AssistantMessage):
                if msg.tool_calls:
                    tool_calls = [
                        {
                            "id": tc.id,
                            "name": tc.name,
                            "args": tc.arguments,
                            "type": "tool_call",
                        }
                        for tc in msg.tool_calls
                    ]
                    state.append(AIMessage(content="", tool_calls=tool_calls))
                    for tc in msg.tool_calls:
                        self._pending_tool_calls[tc.id] = {
                            "name": tc.name,
                            "args": tc.arguments,
                        }
                else:
                    state.append(AIMessage(content=msg.content or ""))
            elif isinstance(msg, ToolMessage):
                self._append_tool_message_if_pending(
                    state,
                    msg.id,
                    msg.content or "",
                )
                self._record_completed_tool_call(msg)

        return state

    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: list,
    ) -> tuple[AssistantMessage, list]:
        state = list(state)  # avoid mutating caller's list

        # Record completed tool calls coming back from tau2
        if isinstance(message, ToolMessage):
            self._record_completed_tool_call(message)
        elif isinstance(message, MultiToolMessage):
            for tm in message.tool_messages:
                self._record_completed_tool_call(tm)

        # Translate incoming tau2 message -> LangChain
        if isinstance(message, UserMessage):
            state.append(HumanMessage(content=message.content or ""))
        elif isinstance(message, ToolMessage):
            self._append_tool_message_if_pending(
                state,
                message.id,
                message.content or "",
            )
        elif isinstance(message, MultiToolMessage):
            for tm in message.tool_messages:
                self._append_tool_message_if_pending(
                    state,
                    tm.id,
                    tm.content or "",
                )

        state = self._sanitize_tool_messages(state)

        retries = 0
        while True:
            state = self._sanitize_tool_messages(list(state))
            # Invoke the graph (agent node -> SecureToolNode -> END)
            result = self._graph.invoke({"messages": state})
            updated_messages: list = list(result.get("messages", state))

            # Find the last AIMessage and any ToolMessages after it
            last_ai_idx = None
            for i in range(len(updated_messages) - 1, -1, -1):
                if isinstance(updated_messages[i], AIMessage):
                    last_ai_idx = i
                    break

            if last_ai_idx is None:
                state = updated_messages
                return (
                    AssistantMessage(role="assistant", content="I cannot continue."),
                    state,
                )

            ai_msg: AIMessage = updated_messages[last_ai_idx]
            tool_results = [
                m
                for m in updated_messages[last_ai_idx + 1 :]
                if isinstance(m, LCToolMessage)
            ]

            # Case 1: text response (no tool calls)
            if not ai_msg.tool_calls:
                state = self._sanitize_tool_messages(updated_messages)
                return (
                    AssistantMessage(role="assistant", content=ai_msg.content),
                    state,
                )

            # Case 2: tool calls were made -- check SecureToolNode results
            approved_calls: list[dict] = []
            declined_calls: list[dict] = []

            for tc in ai_msg.tool_calls:
                tc_id = tc["id"]
                result_msg = next(
                    (m for m in tool_results if m.tool_call_id == tc_id),
                    None,
                )

                if result_msg is None:
                    # No result -- treat as approved (graph exited before tools)
                    approved_calls.append(tc)
                    self._record_validation_event(tc, "allow", None)
                    self._logger.info(
                        "TOOL APPROVED | %s | args=%s",
                        tc["name"],
                        json.dumps(tc.get("args", {})),
                    )
                elif result_msg.content.startswith(_DECLINE_PREFIX):
                    declined_calls.append(tc)
                    violations = result_msg.content
                    self._record_validation_event(tc, "decline", violations)
                    self._logger.warning(
                        "TOOL DECLINED | %s | args=%s | reason=%s",
                        tc["name"],
                        json.dumps(tc.get("args", {})),
                        violations,
                    )
                    if hasattr(self.trace_collector, "record_declined"):
                        self.trace_collector.record_declined(
                            tc["name"], violations
                        )
                else:
                    # Marker or real output -- this was approved
                    approved_calls.append(tc)
                    self._record_validation_event(tc, "allow", None)
                    self._logger.info(
                        "TOOL APPROVED | %s | args=%s",
                        tc["name"],
                        json.dumps(tc.get("args", {})),
                    )

            # All calls approved -> return to tau2
            if approved_calls and not declined_calls:
                # Enforce single tool call
                if len(approved_calls) > 1:
                    denial = (
                        "Only one tool call at a time is allowed. "
                        "Please choose the single next best tool call."
                    )
                    if retries >= self.max_internal_retries:
                        state = self._sanitize_tool_messages(updated_messages)
                        return (
                            AssistantMessage(role="assistant", content=denial),
                            state,
                        )
                    state = list(updated_messages)
                    approved_ids = {tc["id"] for tc in approved_calls}
                    seen_ids: set[str] = set()
                    for idx, msg in enumerate(state):
                        if (
                            isinstance(msg, LCToolMessage)
                            and msg.tool_call_id in approved_ids
                        ):
                            state[idx] = LCToolMessage(
                                content=denial,
                                tool_call_id=msg.tool_call_id,
                            )
                            seen_ids.add(msg.tool_call_id)
                    for tc in approved_calls:
                        if tc["id"] not in seen_ids:
                            self._append_tool_message_if_pending(
                                state,
                                tc["id"],
                                denial,
                            )
                    retries += 1
                    continue

                tc = approved_calls[0]
                self._pending_tool_calls[tc["id"]] = {
                    "name": tc["name"],
                    "args": tc.get("args", {}),
                }
                # Return state up to the AIMessage -- drop stub ToolMessages
                state = self._sanitize_tool_messages(updated_messages[: last_ai_idx + 1])
                out = AssistantMessage(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id=tc["id"],
                            name=tc["name"],
                            arguments=tc.get("args", {}),
                            requestor="assistant",
                        )
                    ],
                )
                valid, err = validate_message_format_default(out)
                if not valid:
                    return (
                        AssistantMessage(role="assistant", content=err),
                        state,
                    )
                return out, state

            # Some/all calls declined -> retry
            if retries >= self.max_internal_retries:
                decline_text = "\n".join(
                    r.content
                    for r in tool_results
                    if r.content.startswith(_DECLINE_PREFIX)
                ) or "Tool call declined by policy."
                state = self._sanitize_tool_messages(updated_messages)
                return (
                    AssistantMessage(role="assistant", content=decline_text),
                    state,
                )

            state = self._sanitize_tool_messages(list(updated_messages))
            retries += 1

    # ------------------------------------------------------------------
    # Trace event recording
    # ------------------------------------------------------------------

    def _record_validation_event(
        self,
        tool_call: dict[str, Any],
        decision: str,
        violations: str | None,
    ) -> None:
        event = Event(
            {
                "concept:name": "validation_check",
                "time:timestamp": datetime.datetime.now(datetime.timezone.utc),
                "tool_name": tool_call["name"],
                "tool_call_id": tool_call.get("id", ""),
                "decision": decision,
                "violations": violations,
                "tool_args": json.dumps(tool_call.get("args", {})),
            }
        )
        self.trace_collector.get_trace().append(event)

    def _record_completed_tool_call(self, tool_msg: ToolMessage) -> None:
        pending = self._pending_tool_calls.pop(tool_msg.id, None)
        if pending is None:
            return

        status = "error" if tool_msg.error else "complete"
        event = Event(
            {
                "concept:name": pending["name"],
                "time:timestamp": datetime.datetime.now(datetime.timezone.utc),
                "lifecycle:transition": status,
                "status": status,
                "id": tool_msg.id,
                "output": tool_msg.content or "",
            }
        )
        args = pending.get("args", {}) or {}
        if isinstance(args, dict):
            for key, value in args.items():
                event[key] = _coerce_numeric(value) if _coerce_numeric else value
        self.trace_collector.get_trace().append(event)
        # Notify subscribers (e.g. DCRStateTracker) so the committed DCR graph
        # is updated. Direct .append() bypasses _publish_trace_event, which
        # means the state tracker never sees the completion event and keeps
        # declining calls that depend on this tool having executed.
        _publish = getattr(self.trace_collector, "_publish_trace_event", None)
        if callable(_publish):
            _publish(event)

        # Truncate output for readability in logs
        output = tool_msg.content or ""
        if len(output) > 200:
            output = output[:200] + "..."
        self._logger.info(
            "TOOL EXECUTED  | %s | status=%s | args=%s | output=%s",
            pending["name"],
            status,
            json.dumps(args),
            output,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(
        self,
        message: Optional[ValidAgentInputMessage] = None,
        state: Optional[list] = None,
    ) -> None:
        reason = "session_end"
        if message is not None:
            reason = type(message).__name__

        record_agent_end = getattr(self.trace_collector, "record_agent_end", None)
        if callable(record_agent_end):
            record_agent_end(decided_by=reason, content="tau2 orchestrator stop")

    def get_validation_events(self) -> list[dict]:
        """Return all 'decline' validation events recorded during this run."""
        events = []
        for event in self.trace_collector.get_trace():
            if event.get("concept:name") != "validation_check":
                continue
            if event.get("decision") != "decline":
                continue
            ts = event.get("time:timestamp")
            events.append(
                {
                    "tool_name": event.get("tool_name", ""),
                    "tool_call_id": event.get("tool_call_id", ""),
                    "tool_args": event.get("tool_args", "{}"),
                    "decision": "decline",
                    "violations": event.get("violations"),
                    "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts) if ts else None,
                }
            )
        return events
