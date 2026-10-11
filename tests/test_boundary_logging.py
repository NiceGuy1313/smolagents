import json
import logging

import pytest

from smolagents import CodeAgent, Tool, ToolCallingAgent
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole, Model
from smolagents.monitoring import TokenUsage
from smolagents.utils import AgentGenerationError


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
    return ChatMessage(
        role=MessageRole.ASSISTANT,
        content=f"Thought: run it.\n<code>\n{code}\n</code>",
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def tool_reply(name: str, arguments: dict) -> ChatMessage:
    call = ChatMessageToolCall(
        function=ChatMessageToolCallFunction(name=name, arguments=arguments), id=f"call_{name}", type="function"
    )
    return ChatMessage(role=MessageRole.ASSISTANT, content="", tool_calls=[call])


@pytest.fixture
def boundary_records(caplog):
    caplog.set_level(logging.DEBUG, logger="smolagents")

    def records() -> list[logging.LogRecord]:
        return [
            r for r in caplog.records if isinstance(r.msg, str) and r.msg.startswith(("[LLM_", "[TOOL_EXECUTION_"))
        ]

    return records


def tags(records) -> list[str]:
    return [r.msg.split("]")[0] + "]" for r in records]


def test_code_agent_logs_llm_and_tool_boundaries(boundary_records):
    agent = CodeAgent(tools=[EchoTool()], model=ScriptedModel([code_reply('final_answer(echo(text="hi"))')]))
    agent.run("Echo hi")

    records = boundary_records()
    assert tags(records) == [
        "[LLM_QUERY]",
        "[LLM_RESPONSE]",
        "[TOOL_EXECUTION_START]",
        "[TOOL_EXECUTION_END]",
        "[TOOL_EXECUTION_START]",
        "[TOOL_EXECUTION_END]",
    ]
    query, response, echo_start, echo_end = records[:4]
    assert query.args[:4] == (1, "scripted-model", 2, "action")
    step, purpose, actions, input_tokens, output_tokens = response.args
    assert (step, purpose, input_tokens, output_tokens) == (1, "action", 10, 5)
    assert json.loads(actions) == [{"python_interpreter": 'final_answer(echo(text="hi"))'}]
    assert echo_start.args == ("echo", {"text": "hi"})
    assert echo_end.args[:2] == ("echo", "str")
    assert records[4].args == ("final_answer", {"answer": "echo: hi"})


def test_tool_calling_agent_logs_parsed_tool_calls(boundary_records):
    model = ScriptedModel([tool_reply("echo", {"text": "hi"}), tool_reply("final_answer", {"answer": "done"})])
    ToolCallingAgent(tools=[EchoTool()], model=model).run("Echo hi")

    responses = [r for r in boundary_records() if r.msg.startswith("[LLM_RESPONSE]")]
    assert [json.loads(r.args[2]) for r in responses] == [
        [{"echo": {"text": "hi"}}],
        [{"final_answer": {"answer": "done"}}],
    ]


def test_failing_tool_logs_tool_execution_error(boundary_records):
    model = ScriptedModel([code_reply('broken(text="x")'), code_reply('final_answer("gave up")')])
    CodeAgent(tools=[BrokenTool()], model=model).run("Use the broken tool")

    error = next(r for r in boundary_records() if r.msg.startswith("[TOOL_EXECUTION_ERROR]"))
    assert error.args[:2] == ("broken", "RuntimeError")


def test_planning_and_generation_errors_are_logged(boundary_records):
    plan = ChatMessage(role=MessageRole.ASSISTANT, content="1. Do it\n<end_plan>")
    agent = CodeAgent(tools=[], model=ScriptedModel([plan, RuntimeError("provider down")]), planning_interval=3)

    with pytest.raises(AgentGenerationError):
        agent.run("Plan, then fail")

    records = boundary_records()
    assert tags(records) == ["[LLM_QUERY]", "[LLM_RESPONSE]", "[LLM_QUERY]", "[LLM_ERROR]"]
    assert records[0].args[3] == "planning" and records[1].args[1] == "planning"
    assert records[3].args == (1, "action", "RuntimeError")


def test_forced_final_answer_after_max_steps_is_logged(boundary_records):
    model = ScriptedModel(
        [code_reply('print("still working")'), ChatMessage(role=MessageRole.ASSISTANT, content="42")]
    )
    CodeAgent(tools=[], model=model, max_steps=1).run("Never finish")

    final = [r for r in boundary_records() if r.msg.startswith("[LLM_") and "final_answer" in r.args]
    assert tags(final) == ["[LLM_QUERY]", "[LLM_RESPONSE]"]
