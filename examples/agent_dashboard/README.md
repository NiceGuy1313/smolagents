# Agent Observatory (smolagents)

Local web app that runs a smolagents agent and shows two panels side by side:

- **Left:** the agent loop as an architecture diagram, marking where the LLM calls and tool calls happen.
- **Right:** the run's call history, grouped per step. Each step shows the plan, the LLM calls (the code it wrote or the tool calls it chose), each tool execution with its params, output or error, the observation, and the duration and token counts.

It works with both `CodeAgent` and `ToolCallingAgent`. Like the browser-use dashboard, it gets the history from boundary logs that are placed directly in the agent loop (`agents.py`, `tools.py`).

## Run

Same steps as the browser-use dashboard:

```bash
cp .env.example .env      # then set ANTHROPIC_API_KEY in .env
uv sync
uv run uvicorn examples.agent_dashboard.app:app --port 8000
```

Open http://127.0.0.1:8000. The key is read from the repository `.env` on the server and never sent to the browser.

`uv sync` also installs the default `dashboard` dependency group from `pyproject.toml`: LiteLLM, the web search and page-reading tools, FastAPI, uvicorn and pytest. The upstream `.gitignore` ignores `uv.lock`, so the lock file stays local.

In the model field you can enter a plain Claude name such as `claude-sonnet-5`; the app adds the `anthropic/` prefix for you. Other providers take a LiteLLM id such as `openai/gpt-4o`, plus that provider's key in `.env`.

The agent gets `WebSearchTool` (DuckDuckGo) and `VisitWebpageTool`. To change the tools or the agent settings, edit `build_agent` in `app.py`.

## How tracing works

The LLM and tool boundaries log at DEBUG level on the `smolagents.agents` and `smolagents.tools` loggers. The tags and arguments follow the browser-use fork:

| Log | Where | Args |
|---|---|---|
| `[LLM_QUERY]` | `agents.py`, before every model call: planning (`_generate_planning_step`), action (`CodeAgent._step_stream`, `ToolCallingAgent._step_stream`), forced final answer (`provide_final_answer`) | `step, model, messages, purpose` |
| `[LLM_RESPONSE]` | after the reply is parsed into actions: the code blob (`{"python_interpreter": code}`) or the tool calls | `step, purpose, actions_json, input_tokens, output_tokens` |
| `[LLM_ERROR]` | when generation raises | `step, purpose, error_type` |
| `[TOOL_EXECUTION_START/END/ERROR]` | `tools.py`, `Tool.__call__` around `forward`. Both agents go through it: CodeAgent from generated code, ToolCallingAgent via `execute_tool_call` | `action, params` / `action, result_type, duration` / `action, error_type, duration` |

The logging calls live in three helpers on `MultiStepAgent` (`_log_llm_query`, `_log_llm_response`, `_log_llm_error`), and every call site uses them. `[LLM_RESPONSE]` skips its JSON serialization when DEBUG is off.

While a run is active, `DashboardLogHandler` in `app.py` is attached to the `smolagents` logger. It builds trace events from each record's positional `args`, not from the formatted text. If you change the argument order of those log calls, update `parse_boundary_record` as well. Step results (code, observations, errors, `is_final_answer`, plan) come from `step_callbacks` for `ActionStep` and `PlanningStep`.

How the events are assembled:

- Tool events don't carry a step number, so the hub assigns each one to the step of the most recent `[LLM_QUERY]`, the same way the browser-use dashboard does.
- `[TOOL_EXECUTION_ERROR]` carries only the error type. The full message is in the `step` event's `error`, which also covers errors outside tools: code parse errors, code execution errors and failed final answer checks.
- When `Model.generate` raises, smolagents raises `AgentGenerationError` and stops the run. The trace records `[LLM_ERROR]`, then a failed `run` event.

## API

- `POST /api/run` starts a run. The body is `{task, max_steps, model, agent_type: "code" | "tool_calling"}`. Only one run can be active at a time.
- `GET /api/events` is an SSE stream. It replays the current run's events and then streams new ones live.
- `GET /api/runs/current/trace.json` returns the full run record. Each finished run is also saved to `runs/<run_id>.json`, which is gitignored.

## Tests

```bash
uv run pytest tests/test_boundary_logging.py examples/agent_dashboard/test_app.py
```

`tests/test_boundary_logging.py` checks the core log lines: both agent types, planning, a failing tool, a failing model call and the forced final answer. `examples/agent_dashboard/test_app.py` runs `execute_agent` end to end and checks the trace events built from those logs. Both use a scripted `Model`, so neither needs network access or an API key.

## Known gaps

- Tools that run in a remote executor (E2B, Docker, Modal) execute in another process, so their `[TOOL_EXECUTION_*]` logs don't reach this handler. Use the local executor.
- Managed agents (sub-agents) log to the same `smolagents` logger, so their calls are mixed into the parent run under the sub-agent's own step numbers.

## Architecture diagram

`static/agent-architecture.html` is a self-contained Archify (3.0.1) export generated from `agent-architecture.json`. The server serves it as-is, so there's no build step. Its source links point to `meta.repository.revision` in the spec, which is the fork commit whose line numbers match the `sources[].line` values. Archify checks them against the local checkout when you pass `--repo-root`. If you edit the logging in `src/smolagents/agents.py` or `src/smolagents/tools.py`, update those line numbers and the revision together.

After editing the spec, validate it and then regenerate the HTML. Run these from the Archify repo's `archify/` directory:

```bash
node bin/archify.mjs validate architecture <smolagents>/examples/agent_dashboard/agent-architecture.json \
  --quality showcase --repo-root <smolagents>
node bin/archify.mjs deliver architecture <smolagents>/examples/agent_dashboard/agent-architecture.json \
  <smolagents>/examples/agent_dashboard/static/agent-architecture.html \
  --quality showcase --repo-root <smolagents> --json
```

`deliver` also writes an `agent-architecture.delivery.json` receipt that contains absolute local paths. Delete it rather than committing it.

The `showcase` quality profile also checks layout: crossing edges, labels that don't have room, and text that would render too small. The diagram is shown at about 930px wide on a 1440px screen, so the `viewBox` width must stay at or below 1240 for the smallest labels to stay readable.
