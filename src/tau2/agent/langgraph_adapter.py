"""
Middleware that bridges a LangChain/LangGraph bound model and the tau2 orchestrator.

Usage:
    Subclass LangGraphAdapter, build a bound model (e.g. ChatLiteLLM().bind_tools(...)),
    and pass it to super().__init__(). The adapter handles all message translation and
    the single-step invocation pattern expected by the tau2 orchestrator.
"""
from typing import Optional

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage as LCSystemMessage,
    ToolMessage as LCToolMessage,
)

from tau2.agent.base import LocalAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool


class LangGraphAdapter(LocalAgent[list]):
    """
    Generic adapter that lets any LangChain bound model participate in a tau2 simulation.

    The adapter translates tau2 messages to LangChain messages and back, and makes a
    single LLM call per generate_next_message() invocation. When the model requests
    tool calls the adapter returns an AssistantMessage with tool_calls set; the tau2
    orchestrator then dispatches those calls to the environment and feeds the
    ToolMessage(s) back into the next generate_next_message() call. This unrolls the
    full ReAct loop into the orchestrator's existing step loop, making every
    intermediate tool call visible in the trajectory and to the evaluator.

    Args:
        model: A LangChain Runnable that accepts a list of messages and returns an
               AIMessage (typically ChatLiteLLM().bind_tools(lc_tools)).
        system_prompt: Text to prepend as the system message in every conversation.
        tools: tau2 Tool list (forwarded to LocalAgent, used for metadata/validation).
        domain_policy: Domain policy string (forwarded to LocalAgent).
    """

    def __init__(
        self,
        model,
        system_prompt: str,
        tools: list[Tool],
        domain_policy: str,
    ) -> None:
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.model = model
        self.system_prompt = system_prompt

    # ------------------------------------------------------------------
    # tau2 LocalAgent interface
    # ------------------------------------------------------------------

    def get_init_state(self, message_history: Optional[list[Message]] = None) -> list:
        """Seed the conversation with the system prompt."""
        return [LCSystemMessage(content=self.system_prompt)]

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: list
    ) -> tuple[AssistantMessage, list]:
        """
        Single-step LLM call.

        Converts the incoming tau2 message to LangChain format, appends it to the
        state, invokes the model once, and returns either:
        - AssistantMessage(tool_calls=[...]) if the model wants to call tools, or
        - AssistantMessage(content="...") if the model produced a text reply.

        The orchestrator loop handles the rest of the ReAct cycle.
        """
        state = list(state)  # avoid mutating caller's list

        # --- tau2 → LangChain message translation ---
        if isinstance(message, UserMessage):
            state.append(HumanMessage(content=message.content or ""))
        elif isinstance(message, ToolMessage):
            state.append(
                LCToolMessage(content=message.content or "", tool_call_id=message.id)
            )
        elif isinstance(message, MultiToolMessage):
            for tm in message.tool_messages:
                state.append(
                    LCToolMessage(content=tm.content or "", tool_call_id=tm.id)
                )

        # --- Single LLM call ---
        ai_msg: AIMessage = self.model.invoke(state)
        state.append(ai_msg)

        # --- LangChain → tau2 message translation ---
        if ai_msg.tool_calls:
            tau2_tool_calls = [
                ToolCall(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc["args"],
                    requestor="assistant",
                )
                for tc in ai_msg.tool_calls
            ]
            return AssistantMessage(role="assistant", tool_calls=tau2_tool_calls), state

        return AssistantMessage(role="assistant", content=ai_msg.content), state
