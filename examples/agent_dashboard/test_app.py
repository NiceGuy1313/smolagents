"""The dashboard turns smolagents' own boundary logs into trace events."""

import functools

import app as dashboard
import pytest

from smolagents import Tool
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole, Model


class EchoTool(Tool):
    name = "echo"
    description = "Echo the text back."
    inputs = {"text": {"type": "string", "description": "Text to echo."}}
    output_type = "string"

    def forward(self, text: str) -> str:
        return f"echo: {text}"


class BrokenTool(Tool):
    name = "broken"
    description = "Always fails."
    inputs = {"text": {"type": "string", "description": "Ignored."}}
    output_type = "string"

    def forward(self, text: str) -> str:
        raise RuntimeError("tool exploded")


class ScriptedModel(Model):
    def __init__(self, replies: list[ChatMessage | Exception]):
        super().__init__(model_id="scripted-model")
        self.replies = list(replies)

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def code_reply(code: str) -> ChatMessage:
    return ChatMessage(role=MessageRole.ASSISTANT, content=f"Thought: run it.\n<code>\n{code}\n</code>")


def tool_reply(name: str, arguments: dict) -> ChatMessage:
    call = ChatMessageToolCall(
        function=ChatMessageToolCallFunction(name=name, arguments=arguments), id=f"call_{name}", type="function"
    )
    return ChatMessage(role=MessageRole.ASSISTANT, content="", tool_calls=[call])


@pytest.fixture
def run_dashboard(monkeypatch, tmp_path):
    monkeypatch.setattr(dashboard, "RUNS_DIR", tmp_path)

    def run(agent_type: str, replies: list, tools: list[Tool]) -> list[dashboard.TraceEvent]:
        monkeypatch.setattr(dashboard, "build_model", lambda request: ScriptedModel(replies))
        monkeypatch.setattr(dashboard, "build_agent", functools.partial(dashboard.build_agent, tools=tools))
        request = dashboard.RunRequest(task="test task", agent_type=agent_type, max_steps=3)
        dashboard.hub.start(request)
        dashboard.execute_agent(request)
        assert (tmp_path / f"{dashboard.hub.run.id}.json").exists()
        return dashboard.hub.run.events

    return run


def summary(events, step: int) -> list[str]:
    return [f"{e.kind}:{e.data.get('action', '')}".rstrip(":") for e in events if e.step == step]


def test_code_agent_run_is_traced_from_boundary_logs(run_dashboard):
    events = run_dashboard("code", [code_reply('final_answer(echo(text="hi"))')], [EchoTool()])

    assert summary(events, 1) == [
        "llm_query",
        "llm_response",
        "tool_start:echo",
        "tool_end:echo",
        "tool_start:final_answer",
        "tool_end:final_answer",
        "step",
    ]
    response = next(e for e in events if e.kind == "llm_response")
    assert response.data["actions"] == [{"python_interpreter": 'final_answer(echo(text="hi"))'}]
    assert response.data["duration_seconds"] is not None
    assert events[-1].kind == "run" and events[-1].data["success"] is True
    assert [e.seq for e in events] == list(range(len(events)))


def test_tool_calling_agent_actions_come_from_parsed_tool_calls(run_dashboard):
    replies = [tool_reply("echo", {"text": "hi"}), tool_reply("final_answer", {"answer": "done"})]
    events = run_dashboard("tool_calling", replies, [EchoTool()])

    assert [e.data["actions"] for e in events if e.kind == "llm_response"] == [
        [{"echo": {"text": "hi"}}],
        [{"final_answer": {"answer": "done"}}],
    ]
    assert summary(events, 1)[2:] == ["tool_start:echo", "tool_end:echo", "step"]


def test_tool_and_model_failures_are_traced(run_dashboard):
    events = run_dashboard("code", [code_reply('broken(text="x")'), RuntimeError("provider down")], [BrokenTool()])

    tool_error = next(e for e in events if e.kind == "tool_error")
    assert tool_error.step == 1 and tool_error.data["error_type"] == "RuntimeError"
    step_1 = next(e for e in events if e.kind == "step" and e.step == 1)
    assert "tool exploded" in step_1.data["error"]
    llm_error = next(e for e in events if e.kind == "llm_error")
    assert llm_error.step == 2 and llm_error.data["error_type"] == "RuntimeError"
    assert events[-1].kind == "run" and events[-1].level == "ERROR"
