# Secure Policy Validation for tau2-bench

This repository is a thesis extension of [tau2-bench](https://github.com/sierra-research/tau2-bench). It evaluates whether a policy-validation layer can prevent unsafe airline-agent tool calls before they reach the benchmark environment.

The work adds a secure airline agent, two policy encodings, adversarial airline tasks, and an experiment corpus. The upstream tau2 benchmark remains the simulation harness; this repository is the domain integration and evaluation layer.

## Contribution

The secure agent routes each proposed LangChain/LangGraph tool call through a validator before tau2 executes the real airline tool:

```text
LLM -> SecureToolNode -> policy validator -> tau2 airline environment
                         | ALLOW       | DECLINE and retry
```

The project compares two policy representations for the same airline policy rules:

- **MP-Declare:** trace-based constraints with Python predicates.
- **Data-aware DCR:** a graph-based, fail-closed policy with DB-backed input-event resolution and computed boolean gates.

The implementation requires three local research dependencies in addition to
this repository:

| Repository | Responsibility |
| --- | --- |
| `Declare4PyRM` (`declare4py-rm`) | MP-Declare runtime monitoring and data-condition support for the Declare validator. |
| `thesis-dpm-secure-langgraph` | Secure LangGraph wrapper, candidate validation, and committed-state tracking. |
| `pm4py-dcr` (`pm4py`) | Data-aware DCR graph model, semantics, and expression language. |
| `tau2-bench-thesis` | Airline policies, DB resolver, secure-agent adapter, tasks, tests, and experiment results. |

## Repository Map

| Path | Contents |
| --- | --- |
| [src/tau2/agent](src/tau2/agent) | `SecureAirlineAgent` and the tau2/LangGraph adapter. |
| [data/tau2/domains/airline/security](data/tau2/domains/airline/security) | Declare policies/predicates and DCR policies/resolvers. |
| [tests/test_secure_airline_agent.py](tests/test_secure_airline_agent.py) | Secure interception, rejection, retry, and execution-boundary tests. |
| [tests/test_domains/test_airline/test_dcr2_data_resolver.py](tests/test_domains/test_airline/test_dcr2_data_resolver.py) | DCR resolver and policy-gate integration tests. |
| [artifacts/README.md](artifacts/README.md) | Thesis analysis, architecture notes, and policy walkthroughs. |
| [data/simulations/README.md](data/simulations/README.md) | Retained Declare/DCR experiment corpus. |

## Setup

Requirements:

- Python 3.10 or newer.
- Local checkouts of `../Declare4PyRM`, `../thesis-dpm-secure-langgraph`, and `../pm4py-dcr` beside this repository, or equivalent editable paths configured in the `Makefile`.
- Credentials for the LiteLLM provider used by the agent and user simulator. The default configuration uses Azure model names; set the appropriate `AZURE_*` variables. OpenAI model names instead require `OPENAI_API_KEY`.

`Declare4PyRM` is the required runtime-monitoring fork. It publishes the
`declare4py-rm` distribution but retains the `Declare4Py` Python import namespace.
It is distinct from a plain upstream `Declare4Py` checkout. `pm4py-dcr` likewise
publishes the `pm4py` namespace, so do not substitute a standard PyPI PM4Py
installation for the local fork.

Create and activate an environment, then install all four editable packages in
dependency order. `make install` does this using the configured local paths:

```bash
python -m venv .venv
source .venv/bin/activate
make install
```

Check the available runtime flags:

```bash
tau2 run --help
```

## Run a Secure Evaluation

The current verified end-to-end policy route is the Declare YAML policy. This smoke run evaluates three basic airline tasks and writes an ignored local result under `data/simulations/`:

```bash
tau2 run \
  --domain airline \
  --agent secure_airline_agent \
  --sec-file policy_v3.yaml \
  --agent-llm azure/gpt-5-mini-US \
  --user-llm azure/gpt-5-mini-US \
  --task-set basic \
  --num-trials 1 \
  --num-tasks 3 \
  --save-to secure_airline_smoke
```

Use `--task-set adv` for the adversarial airline task set. The `--soft-agent` switch deliberately instructs the model to follow user pressure while the validator remains active; it is useful for testing whether the enforcement layer blocks unsafe proposals.

## Policies and DCR Status

Declare policy files use the `policy*.yaml` / `predicates*.py` convention. Pass a policy with `--sec-file`, relative to `data/tau2/domains/airline/security/`.

The data-aware DCR research policy is [dcr2.yaml](data/tau2/domains/airline/security/dcr2.yaml), supported by [dcr2_data_resolver.py](data/tau2/domains/airline/security/dcr2_data_resolver.py), [dcr2_predicates.py](data/tau2/domains/airline/security/dcr2_predicates.py), and [dcr2_expr_predicates.py](data/tau2/domains/airline/security/dcr2_expr_predicates.py). It is covered by an integration test and described in the artifacts below.

**Current checkout status:** the secure-agent code currently routes YAML files through the Declare parser, while the DCR test and architecture notes expect a DCR-YAML loader and resolver hook. Consequently, do not use `--sec-file dcr2.yaml` as a reproducible end-to-end command until that integration drift is repaired. The DCR policy and analysis remain retained research evidence.

## Validate

Run the focused tests after installing the local dependencies:

```bash
pytest tests/test_secure_airline_agent.py
pytest tests/test_domains/test_airline/test_dcr2_data_resolver.py
make lint
```

`make test` runs the full suite. The test commands require the editable
`Declare4PyRM`, secure-LangGraph, and DCR-engine dependencies described above.

## Research Evidence

The detailed results and design rationale are intentionally retained:

- [Declare vs. DCR comparison](artifacts/DECLARE_VS_DCR_COMPARISON.md) documents the formal and engineering trade-offs across 39 policy rules.
- [DCR architecture](artifacts/DCR_DATA_ARCHITECTURE.md) follows candidate validation, resolution, diagnostics, and state commit end to end.
- [DCR validation notes](artifacts/DCR_VALIDATION_THESIS_NOTES.md) records design choices and structural limits.
- [OWASP evaluation](artifacts/OWASP_ASI_EVALUATION.md) maps the adversarial benchmark to the OWASP Agentic AI Security Top 10.
- [Simulation corpus](data/simulations/README.md) identifies the retained result families without discarding exploratory runs.

## Provenance

This is a research fork based on tau2-bench. For benchmark-wide domains, Gym support, leaderboard submission, and general framework behavior, refer to the upstream [tau2-bench repository](https://github.com/sierra-research/tau2-bench).
