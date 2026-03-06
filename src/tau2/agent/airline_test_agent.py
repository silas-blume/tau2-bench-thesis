
import os
from copy import deepcopy
from typing import Optional

from langchain_core.tools import StructuredTool
from langchain_litellm import ChatLiteLLM

from tau2.agent.langgraph_adapter import LangGraphAdapter
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.tools import AirlineTools
from tau2.domains.airline.utils import AIRLINE_DB_PATH
from tau2.environment.tool import Tool

SYSTEM_PROMPT = """\
You are a customer service agent. Help the user by following the policy below.

<policy>
{domain_policy}
</policy>
"""


class AirlineTestAgent(LangGraphAdapter):
    """
    Airline customer-service agent backed by a LangChain/LiteLLM model.

    Tool execution and message logging are handled by LangGraphAdapter so that
    every tool call appears in the tau2 trajectory and is visible to the evaluator.
    Swapping the underlying LLM or tools only requires changes inside __init__.
    """

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
    ) -> None:
        llm_args = deepcopy(llm_args) if llm_args is not None else {}

        # Build LangChain tools from AirlineTools bound methods
        db = FlightDB.load(AIRLINE_DB_PATH)
        toolkit = AirlineTools(db)
        lc_tools = [
            StructuredTool.from_function(method, name=name)
            for name, method in toolkit.tools.items()
        ]

        model_str = llm or os.environ.get("AGENT_MODEL", "gpt-4o")
        bound_model = ChatLiteLLM(model=model_str, **llm_args).bind_tools(lc_tools)
        system_prompt = SYSTEM_PROMPT.format(domain_policy=domain_policy)

        super().__init__(
            model=bound_model,
            system_prompt=system_prompt,
            tools=tools,
            domain_policy=domain_policy,
        )
