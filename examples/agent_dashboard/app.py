"""Local dashboard: run a smolagents agent and stream its LLM/tool call history next to an architecture map.

The history comes from the boundary logs in smolagents itself ([LLM_QUERY], [LLM_RESPONSE], [LLM_ERROR] in
agents.py and [TOOL_EXECUTION_*] in tools.py), the same way the browser-use dashboard reads browser-use's logs.
"""

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from smolagents import CodeAgent, LiteLLMModel, Model, Tool, ToolCallingAgent, VisitWebpageTool, WebSearchTool
from smolagents.agents import MultiStepAgent
from smolagents.memory import ActionStep, PlanningStep


APP_DIR = Path(__file__).resolve().parent
RUNS_DIR = APP_DIR / "runs"
# Both agent dashboards serve the same paths on the same port; make browsers revalidate instead of reusing the other one's page.
NO_CACHE = {"Cache-Control": "no-cache"}
load_dotenv(APP_DIR.parents[1] / ".env")

TraceKind = Literal[
    "run", "llm_query", "llm_response", "llm_error", "tool_start", "tool_end", "tool_error", "step", "log"
]
TAG_TO_KIND: dict[str, TraceKind] = {
    "[LLM_QUERY]": "llm_query",
    "[LLM_RESPONSE]": "llm_response",
    "[LLM_ERROR]": "llm_error",
    "[TOOL_EXECUTION_START]": "tool_start",
    "[TOOL_EXECUTION_END]": "tool_end",
    "[TOOL_EXECUTION_ERROR]": "tool_error",
}
PREVIEW_CHARS = 2000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def preview(value: Any, limit: int = PREVIEW_CHARS) -> str | None:
    if value is None:
        return None
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[:limit] + "…"


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=3, max_length=2000)
    max_steps: int = Field(default=8, ge=1, le=30)
    model: str = Field(default="claude-sonnet-5", min_length=1, max_length=100)
    agent_type: Literal["code", "tool_calling"] = "code"


class TraceEvent(BaseModel):
    seq: int
    run_id: str
    timestamp: str
    kind: TraceKind
    level: str = "INFO"
    step: int | None = None
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class RunRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task: str
    model: str
    agent_type: str
    max_steps: int
    status: Literal["running", "finished", "failed"] = "running"
    started_at: str = Field(default_factory=_now)
    finished_at: str | None = None
    events: list[TraceEvent] = Field(default_factory=list)


class TraceHub:
    """Owns the current run and fans its events out to subscribers. Safe to publish from any thread."""

    def __init__(self) -> None:
        self.run: RunRecord | None = None
        self.current_step: int | None = None
        self._llm_started_at: float | None = None
        self._subscribers: set[Callable[[TraceEvent], None]] = set()
        self._lock = threading.Lock()

    def start(self, request: RunRequest) -> RunRecord:
        with self._lock:
            self.run = RunRecord(
                task=request.task, model=request.model, agent_type=request.agent_type, max_steps=request.max_steps
            )
            self.current_step = None
            self._llm_started_at = None
            return self.run

    def subscribe(self, callback: Callable[[TraceEvent], None]) -> list[TraceEvent]:
        """Register a subscriber and return the events published so far, atomically."""
        with self._lock:
            self._subscribers.add(callback)
            return list(self.run.events) if self.run else []

    def unsubscribe(self, callback: Callable[[TraceEvent], None]) -> None:
        with self._lock:
            self._subscribers.discard(callback)

    def publish(
        self,
        kind: TraceKind,
        message: str,
        *,
        level: str = "INFO",
        step: int | None = None,
        data: dict[str, Any] | None = None,
    ) -> TraceEvent:
        with self._lock:
            assert self.run is not None, "publish() called without an active run"
            data = dict(data or {})
            if kind == "llm_query":
                self.current_step = step
                self._llm_started_at = time.monotonic()
            elif kind in ("llm_response", "llm_error") and self._llm_started_at is not None:
                data["duration_seconds"] = round(time.monotonic() - self._llm_started_at, 3)
                self._llm_started_at = None
            if step is None and kind in ("tool_start", "tool_end", "tool_error"):
                step = self.current_step
            event = TraceEvent(
                seq=len(self.run.events),
                run_id=self.run.id,
                timestamp=_now(),
                kind=kind,
                level=level,
                step=step,
                message=message,
                data=data,
            )
            self.run.events.append(event)
            subscribers = tuple(self._subscribers)
        for callback in subscribers:
            callback(event)
        return event

    def finish(self, status: Literal["finished", "failed"]) -> None:
        with self._lock:
            assert self.run is not None
            self.run.status = status
            self.run.finished_at = _now()
            RUNS_DIR.mkdir(exist_ok=True)
            (RUNS_DIR / f"{self.run.id}.json").write_text(self.run.model_dump_json(indent=2), encoding="utf-8")


hub = TraceHub()
run_task: asyncio.Task[None] | None = None


def parse_boundary_record(kind: TraceKind, args: tuple[Any, ...]) -> tuple[int | None, str, dict[str, Any]]:
    """Turn the positional args of a smolagents boundary log call into structured trace data."""
    if kind == "llm_query":
        step, model, n_messages, purpose = args
        return (
            int(step),
            f"{model} · {n_messages} messages",
            {"model": model, "messages": n_messages, "purpose": purpose},
        )
    if kind == "llm_response":
        step, purpose, actions_json, input_tokens, output_tokens = args
        actions = json.loads(actions_json)
        names = ", ".join(next(iter(a)) for a in actions if a) or purpose
        return (
            int(step),
            names,
            {"purpose": purpose, "actions": actions, "input_tokens": input_tokens, "output_tokens": output_tokens},
        )
    if kind == "llm_error":
        step, purpose, error_type = args
        return int(step), f"{purpose} call failed: {error_type}", {"purpose": purpose, "error_type": error_type}
    if kind == "tool_start":
        action, params = args
        return None, action, {"action": action, "params": preview(params)}
    action, type_name, duration = args
    key = "result_type" if kind == "tool_end" else "error_type"
    return None, action, {"action": action, key: type_name, "duration_seconds": round(float(duration), 3)}


class DashboardLogHandler(logging.Handler):
    """Convert smolagents log records into trace events."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            boundary: TraceKind | None = next(
                (k for tag, k in TAG_TO_KIND.items() if isinstance(record.msg, str) and record.msg.startswith(tag)),
                None,
            )
            kind: TraceKind
            step: int | None = None
            if boundary is not None:
                kind = boundary
                step, message, data = parse_boundary_record(
                    kind, record.args if isinstance(record.args, tuple) else ()
                )
            elif record.levelno >= logging.INFO:
                kind = "log"
                message, data = record.getMessage(), {"logger": record.name}
            else:
                return
        except Exception:
            self.handleError(record)
            return
        hub.publish(kind, message, level=record.levelname, step=step, data=data)


def publish_action_step(step: ActionStep) -> None:
    error = str(step.error) if step.error else None
    usage = step.token_usage
    hub.publish(
        "step",
        "error" if error else ("final answer" if step.is_final_answer else "ok"),
        level="ERROR" if error else "INFO",
        step=step.step_number,
        data={
            "type": "action",
            "duration_seconds": round(step.timing.duration, 3) if step.timing.duration is not None else None,
            "model_output": preview(step.model_output),
            "code": step.code_action,
            "observations": preview(step.observations),
            "error": error,
            "is_final_answer": step.is_final_answer,
            "output": preview(step.action_output, 600),
            "input_tokens": usage.input_tokens if usage else None,
            "output_tokens": usage.output_tokens if usage else None,
        },
    )


def publish_planning_step(step: PlanningStep) -> None:
    hub.publish(
        "step",
        "plan",
        step=hub.current_step,
        data={
            "type": "planning",
            "plan": preview(step.plan),
            "duration_seconds": round(step.timing.duration, 3) if step.timing.duration is not None else None,
        },
    )


def litellm_model_id(model: str) -> str:
    """Accept plain Claude names (as in the browser-use dashboard); other models need a LiteLLM `provider/model` id."""
    return f"anthropic/{model}" if "/" not in model and model.startswith("claude") else model


def build_model(request: RunRequest) -> Model:
    return LiteLLMModel(model_id=litellm_model_id(request.model))


def build_agent(request: RunRequest, model: Model, tools: list[Tool] | None = None) -> MultiStepAgent:
    agent_cls = CodeAgent if request.agent_type == "code" else ToolCallingAgent
    return agent_cls(
        tools=tools if tools is not None else [WebSearchTool(), VisitWebpageTool()],
        model=model,
        max_steps=request.max_steps,
        step_callbacks={ActionStep: publish_action_step, PlanningStep: publish_planning_step},
    )


def missing_api_key(model: str) -> str | None:
    if litellm_model_id(model).startswith("anthropic/") and not os.getenv("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY"
    return None


def execute_agent(request: RunRequest) -> None:
    """Blocking agent run with the boundary logs routed into the hub; executed in a worker thread."""
    logger = logging.getLogger("smolagents")
    previous_level = logger.level
    handler = DashboardLogHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    status: Literal["finished", "failed"] = "failed"
    try:
        agent = build_agent(request, build_model(request))
        result = agent.run(request.task, return_full_result=True)
        status = "finished"
        usage = result.token_usage
        hub.publish(
            "run",
            f"Run finished: state={result.state} steps={sum(isinstance(s, ActionStep) for s in agent.memory.steps)}",
            data={
                "state": result.state,
                "success": True if result.state == "success" else None,
                "final_result": preview(result.output, 4000),
                "duration_seconds": round(result.timing.duration, 3) if result.timing.duration is not None else None,
                "input_tokens": usage.input_tokens if usage else None,
                "output_tokens": usage.output_tokens if usage else None,
            },
        )
    except Exception as error:
        hub.publish("run", f"Run failed: {type(error).__name__}: {preview(str(error), 600)}", level="ERROR")
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        hub.finish(status)


async def run_in_background(request: RunRequest) -> None:
    global run_task
    try:
        await asyncio.to_thread(execute_agent, request)
    finally:
        run_task = None


app = FastAPI(title="smolagents Agent Observatory")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(APP_DIR / "static" / "index.html", headers=NO_CACHE)


@app.get("/architecture")
async def architecture() -> FileResponse:
    return FileResponse(APP_DIR / "static" / "agent-architecture.html", headers=NO_CACHE)


@app.get("/api/status")
async def status() -> dict[str, Any]:
    return {
        "missing_api_key": missing_api_key(RunRequest.model_fields["model"].default),
        "running": bool(run_task and not run_task.done()),
        "run": hub.run.model_dump(exclude={"events"}) if hub.run else None,
    }


@app.post("/api/run", status_code=202)
async def start_run(request: RunRequest) -> dict[str, str]:
    global run_task
    if key := missing_api_key(request.model):
        raise HTTPException(status_code=400, detail=f"{key} is missing from .env")
    if run_task and not run_task.done():
        raise HTTPException(status_code=409, detail="An agent is already running")
    run = hub.start(request)
    hub.publish("run", f"Run started: {request.task}", data=request.model_dump())
    run_task = asyncio.create_task(run_in_background(request))
    return {"status": "started", "run_id": run.id}


@app.get("/api/runs/current/trace.json")
async def current_trace() -> JSONResponse:
    if hub.run is None:
        raise HTTPException(status_code=404, detail="No run yet")
    return JSONResponse(
        hub.run.model_dump(mode="json"),
        headers={"Content-Disposition": f'attachment; filename="smolagents-trace-{hub.run.id}.json"'},
    )


@app.get("/api/events")
async def events() -> StreamingResponse:
    """SSE: replay the current run's history, then stream new events live."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[TraceEvent] = asyncio.Queue(maxsize=5000)

    def enqueue(event: TraceEvent) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    async def stream() -> AsyncIterator[str]:
        backlog = hub.subscribe(enqueue)
        try:
            snapshot = hub.run.model_dump(mode="json", exclude={"events"}) if hub.run else None
            yield f"event: snapshot\ndata: {json.dumps(snapshot, ensure_ascii=False)}\n\n"
            for event in backlog:
                yield f"data: {event.model_dump_json()}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"data: {event.model_dump_json()}\n\n"
                except TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            hub.unsubscribe(enqueue)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
