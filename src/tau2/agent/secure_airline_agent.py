from __future__ import annotations

import json
import os
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, cast
import logging

from langchain_core.tools import StructuredTool
from langchain_litellm import ChatLiteLLM

_SECURE_LANGGRAPH_IMPORT_ERROR: Exception | None = None
AgentDeclareConstraints: Any = None
DeclareTraceValidator: Any = None
TraceCollector: Any = None
try:
	from thesis_dpm_secure_langgraph import (
		AgentDeclareConstraints,
		DeclareTraceValidator,
		TraceCollector,
	)
except Exception as exc:
	# Keep module importable so unrelated TAU2 commands still work when
	# optional secure-agent dependencies are not installed.
	_SECURE_LANGGRAPH_IMPORT_ERROR = exc

from tau2.agent.secure_langgraph_adapter import SecureLangGraphAdapter
from tau2.domains.airline.data_model import FlightDB, get_db
from tau2.domains.airline.tools import AirlineTools
from tau2.environment.tool import Tool

SYSTEM_PROMPT = """\
You are a customer service agent. Help the user by following the policy below.

<policy>
{domain_policy}
</policy>
"""


class SecureAirlineAgent(SecureLangGraphAdapter):
	"""Airline agent using SecureStateGraph with agent-side pre-tool validation."""

	def __init__(
		self,
		tools: list[Tool],
		domain_policy: str,
		llm: Optional[str] = None,
		llm_args: Optional[dict] = None,
	) -> None:
		if _SECURE_LANGGRAPH_IMPORT_ERROR is not None:
			raise ImportError(
				"SecureAirlineAgent requires optional secure-langgraph dependencies. "
				"Install the pm4py DCR fork and thesis_dpm_secure_langgraph, then retry."
			) from _SECURE_LANGGRAPH_IMPORT_ERROR
		assert AgentDeclareConstraints is not None
		assert DeclareTraceValidator is not None
		assert TraceCollector is not None

		llm_args = deepcopy(llm_args) if llm_args is not None else {}

		db = cast(FlightDB, get_db())
		toolkit = AirlineTools(db)
		lc_tools = [
			StructuredTool.from_function(method, name=name)
			for name, method in toolkit.tools.items()
		]

		model_str = llm or os.environ.get("AGENT_MODEL", "gpt-4o")
		model = ChatLiteLLM(model=model_str, **llm_args).bind_tools(lc_tools)
		system_prompt = SYSTEM_PROMPT.format(domain_policy=domain_policy)

		decl_path = Path(
			os.environ.get(
				"TAU2_AIRLINE_POLICY_DECL_PATH",
				Path(__file__).resolve().parents[3]
				/ "data"
				/ "tau2"
				/ "domains"
				/ "airline"
				/ "security"
				/ "policy.decl",
			)
		)
		if not decl_path.exists():
			raise FileNotFoundError(f"Agent Declare policy file not found: {decl_path}")

		# Resolve predicate file: explicit env-var > convention > None.
		predicate_env = os.environ.get("TAU2_AIRLINE_PREDICATE_PATH")
		if predicate_env:
			predicate_path: Path | None = Path(predicate_env)
		else:
			# Convention: predicates file lives next to the policy file
			# and shares its stem (policy_v2.yaml -> predicates_v2.py).
			stem = decl_path.stem  # e.g. "policy", "policy_v2"
			pred_stem = stem.replace("policy", "predicates", 1)
			candidate = decl_path.parent / f"{pred_stem}.py"
			predicate_path = candidate if candidate.exists() else None

		constraints = AgentDeclareConstraints().parse_from_file(
			str(decl_path),
			predicate_file_path=predicate_path,
		)
		validator = DeclareTraceValidator(constraints)

		default_log_dir = (
			Path(__file__).resolve().parents[3]
			/ "data"
			/ "tau2"
			/ "secure_logs"
		)
		self._secure_log_dir = Path(
			os.environ.get("TAU2_SECURE_LOG_DIR", str(default_log_dir))
		)
		self._secure_log_dir.mkdir(parents=True, exist_ok=True)
		self._secure_logger = setup_logger(self._secure_log_dir / "agent.log")
		self._trace_path = self._secure_log_dir / f"trace_events_{uuid.uuid4().hex}.json"

		trace_collector = TraceCollector(
			trace_id=uuid.uuid4(),
			auto_record_agent_start_on_first_run=True,
			verbose=True,
			logger=self._secure_logger,
			result_aliases=constraints.get_result_aliases(),
			bind_schemas=constraints.get_bind_schemas(),
		)
  

		super().__init__(
			model=model,
			system_prompt=system_prompt,
			tools=tools,
			domain_policy=domain_policy,
			validator=validator,
			trace_collector=trace_collector,
			secure_node_verbose=True,
			secure_graph_logger=self._secure_logger,
		)

	def stop(self, message=None, state=None) -> None:
		reason = "session_end"
		if message is not None:
			reason = type(message).__name__

		record_agent_end = getattr(self.trace_collector, "record_agent_end", None)
		if callable(record_agent_end):
			record_agent_end(decided_by=reason, content="tau2 orchestrator stop")

		persist_trace(self.trace_collector, self._trace_path)
		self._secure_logger.info("Secure trace snapshot: %s", self._trace_path)


def setup_logger(log_path: Path) -> logging.Logger:
	logger = logging.getLogger("secure_agent_demo")
	logger.setLevel(logging.INFO)
	logger.propagate = False

	if logger.handlers:
		return logger

	formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

	stream_handler = logging.StreamHandler()
	stream_handler.setLevel(logging.WARNING)
	stream_handler.setFormatter(formatter)

	file_handler = logging.FileHandler(log_path, encoding="utf-8")
	file_handler.setLevel(logging.INFO)
	file_handler.setFormatter(formatter)

	logger.addHandler(stream_handler)
	logger.addHandler(file_handler)

	logging.getLogger("LiteLLM").setLevel(logging.WARNING)
	logging.getLogger("litellm").setLevel(logging.WARNING)
	logging.getLogger("langgraph").setLevel(logging.WARNING)
	logging.getLogger("langchain_core").setLevel(logging.WARNING)
	logging.getLogger("langchain_openai").setLevel(logging.WARNING)
	logging.getLogger("httpcore").setLevel(logging.WARNING)
	logging.getLogger("httpx").setLevel(logging.WARNING)
	logging.getLogger("openai").setLevel(logging.WARNING)
	logging.getLogger("asyncio").setLevel(logging.WARNING)

	return logger


def persist_trace(trace_collector: Any, path: Path) -> None:
	events = [dict(event) for event in trace_collector.get_trace()]
	path.write_text(json.dumps(events, indent=2, default=str), encoding="utf-8")

