import argparse
import asyncio

import pytest

import agentdemo
from guardrails.input_classifier import InputClassification


def test_create_bedrock_chat_model_reads_environment(monkeypatch):
    fake_chat = FakeChatOpenAI()
    monkeypatch.setattr(agentdemo, "ChatOpenAI", fake_chat)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "token")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://bedrock.example")

    model = agentdemo.create_bedrock_chat_model("model")

    assert model == fake_chat
    assert fake_chat.kwargs == {
        "model": "model",
        "api_key": "token",
        "base_url": "https://bedrock.example",
        "temperature": 0.2,
        "max_tokens": 800,
    }


def test_create_bedrock_chat_model_requires_api_key(monkeypatch):
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://bedrock.example")

    with pytest.raises(ValueError, match="AWS_BEARER_TOKEN_BEDROCK"):
        agentdemo.create_bedrock_chat_model("model")


def test_create_bedrock_chat_model_requires_base_url(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "token")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
        agentdemo.create_bedrock_chat_model("model")


def test_get_tavily_mcp_env_passes_key_explicitly(monkeypatch):
    monkeypatch.setenv(agentdemo.TAVILY_API_KEY_ENV_VAR, "tvly-token")

    assert agentdemo.get_tavily_mcp_env() == {
        agentdemo.TAVILY_API_KEY_ENV_VAR: "tvly-token"
    }


def test_get_tavily_mcp_env_handles_missing_key(monkeypatch):
    monkeypatch.delenv(agentdemo.TAVILY_API_KEY_ENV_VAR, raising=False)

    assert agentdemo.get_tavily_mcp_env() == {agentdemo.TAVILY_API_KEY_ENV_VAR: ""}


def test_parse_args_reads_prompt_and_model(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["agentdemo.py", "--prompt", "hello", "--model", "custom-model"],
    )

    args = agentdemo.parse_args()

    assert args.prompt == "hello"
    assert args.model == "custom-model"


def test_format_tool_args_handles_none_short_and_long_values():
    assert agentdemo.format_tool_args(None) == ""
    assert agentdemo.format_tool_args({"x": 1}) == "{'x': 1}"

    long_text = "x" * (agentdemo.MAX_TOOL_ARG_CHARS + 1)
    assert agentdemo.format_tool_args(long_text) == ("x" * agentdemo.MAX_TOOL_ARG_CHARS) + "... [truncated]"


def test_extract_tool_calls_deduplicates_structured_and_raw_calls():
    messages = [
        FakeMessage(
            tool_calls=[
                {"id": "1", "name": "calculator", "args": {"expression": "1+1"}},
                {"id": "1", "name": "calculator", "args": {"expression": "1+1"}},
                {"id": "2", "args": {}},
            ],
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "1",
                        "function": {"name": "calculator", "arguments": "{}"},
                    },
                    {
                        "id": "3",
                        "function": {"name": "read_file", "arguments": '{"file_path":"README.md"}'},
                    },
                    {"id": "4", "function": {}},
                ]
            },
        ),
        object(),
    ]

    assert agentdemo.extract_tool_calls(messages) == [
        {"id": "1", "name": "calculator", "args": {"expression": "1+1"}},
        {"id": "3", "name": "read_file", "args": '{"file_path":"README.md"}'},
    ]


def test_extract_tool_results_reads_tool_messages():
    messages = [
        FakeMessage(content="ignore"),
        FakeMessage(
            message_type="tool",
            name="search_and_download_image",
            content=[{"type": "text", "text": "Downloaded image"}],
        ),
        FakeMessage(tool_call_id="2", content="2"),
    ]

    assert agentdemo.extract_tool_results(messages) == [
        {
            "name": "search_and_download_image",
            "content": [{"type": "text", "text": "Downloaded image"}],
        },
        {"name": "tool", "content": "2"},
    ]


def test_normalize_tool_result_content_unwraps_mcp_text_blocks():
    assert agentdemo.normalize_tool_result_content(
        [{"type": "text", "text": "Image Search Result:"}]
    ) == "Image Search Result:"


def test_normalize_tool_result_content_unwraps_stringified_mcp_text_blocks():
    assert agentdemo.normalize_tool_result_content(
        "[{'type': 'text', 'text': 'Image Search Result:'}]"
    ) == "Image Search Result:"


def test_normalize_tool_result_content_returns_plain_text_unchanged():
    assert agentdemo.normalize_tool_result_content("plain text") == "plain text"


def test_print_tool_call_summary_reports_no_tools(capsys):
    agentdemo.print_tool_call_summary([])

    output = capsys.readouterr().out
    assert "Tool Calls:" in output
    assert "No tool was called." in output


def test_print_tool_call_summary_reports_tools(capsys):
    agentdemo.print_tool_call_summary(
        [{"name": "calculator", "args": {"expression": "1+1"}}]
    )

    output = capsys.readouterr().out
    assert "1. calculator" in output
    assert "args: {'expression': '1+1'}" in output


def test_print_tool_result_summary_reports_no_results(capsys):
    agentdemo.print_tool_result_summary([])

    output = capsys.readouterr().out
    assert "Tool Results:" in output
    assert "No tool result was returned." in output


def test_print_tool_result_summary_reports_results(capsys):
    agentdemo.print_tool_result_summary(
        [
            {
                "name": "search_and_download_image",
                "content": [{"type": "text", "text": "Image Search Result:"}],
            }
        ]
    )

    output = capsys.readouterr().out
    assert "1. search_and_download_image" in output
    assert "Image Search Result:" in output
    assert "[{'type': 'text'" not in output


def test_print_tool_result_summary_uses_result_truncation_limit(capsys):
    agentdemo.print_tool_result_summary(
        [{"name": "search_and_download_image", "content": "x" * 800}]
    )

    output = capsys.readouterr().out
    assert "x" * 800 in output
    assert "[truncated]" not in output


def test_print_guardrail_summary(capsys):
    agentdemo.print_guardrail_summary("local_regex", "unsupported_request")

    output = capsys.readouterr().out
    assert "Source: local_regex" in output
    assert "Category: unsupported_request" in output


def test_format_agent_error_explains_expired_bedrock_token():
    message = agentdemo.format_agent_error(
        RuntimeError("Error code: 401 - {'error': {'code': 'invalid_api_key', 'message': 'Signature expired'}}")
    )

    assert agentdemo.BEDROCK_TOKEN_ERROR_MESSAGE in message
    assert "Raw error:" in message
    assert "Signature expired" in message


def test_format_agent_error_returns_unknown_error_unchanged():
    assert agentdemo.format_agent_error(RuntimeError("boom")) == "boom"


def test_get_agent_tools_combines_local_and_mcp_tools(monkeypatch):
    monkeypatch.setattr(agentdemo, "get_local_tools", lambda: ["local"])

    async def fake_get_mcp_tools():
        return ["mcp"]

    monkeypatch.setattr(agentdemo, "get_mcp_tools", fake_get_mcp_tools)

    assert asyncio.run(agentdemo.get_agent_tools()) == ["local", "mcp"]


def test_get_mcp_tools_registers_project_files_and_tavily_servers(monkeypatch):
    fake_client = FakeMultiServerMCPClient()
    monkeypatch.setattr(agentdemo, "MultiServerMCPClient", fake_client)
    monkeypatch.setenv(agentdemo.TAVILY_API_KEY_ENV_VAR, "tvly-token")

    assert asyncio.run(agentdemo.get_mcp_tools()) == ["mcp-tools"]

    assert set(fake_client.servers) == {"project_files", "tavily_image_search"}
    assert fake_client.servers["tavily_image_search"]["args"] == [
        "src/mcp_servers/tavily_image_search_server.py"
    ]
    assert fake_client.servers["tavily_image_search"]["env"] == {
        agentdemo.TAVILY_API_KEY_ENV_VAR: "tvly-token"
    }
    assert fake_client.tool_name_prefix is False


def test_create_agent_runner_builds_agent(monkeypatch):
    monkeypatch.setattr(agentdemo, "create_bedrock_chat_model", lambda model: f"chat:{model}")

    async def fake_get_agent_tools():
        return ["tool"]

    monkeypatch.setattr(agentdemo, "get_agent_tools", fake_get_agent_tools)
    monkeypatch.setattr(agentdemo, "build_agent_system_prompt", lambda: "system")

    fake_create_agent = FakeCreateAgent()
    monkeypatch.setattr(agentdemo, "create_agent", fake_create_agent)

    assert asyncio.run(agentdemo.create_agent_runner("model")) == fake_create_agent.agent
    assert fake_create_agent.kwargs == {
        "model": "chat:model",
        "tools": ["tool"],
        "system_prompt": "system",
    }


def test_run_agent_demo_returns_bedrock_guardrail_refusal(monkeypatch, capsys):
    monkeypatch.setattr(agentdemo, "parse_args", lambda: argparse.Namespace(prompt="bad", model="model"))
    monkeypatch.setattr(
        agentdemo,
        "evaluate_prompt_with_bedrock_guardrail",
        lambda prompt: InputClassification(False, "bedrock_hate", "blocked", "nope"),
    )

    assert asyncio.run(agentdemo.run_agent_demo()) == 0

    output = capsys.readouterr().out
    assert "Source: bedrock" in output
    assert "nope" in output
    assert "Tool Results:" in output


def test_run_agent_demo_returns_local_guardrail_refusal(monkeypatch, capsys):
    monkeypatch.setattr(agentdemo, "parse_args", lambda: argparse.Namespace(prompt="bad", model="model"))
    monkeypatch.setattr(agentdemo, "evaluate_prompt_with_bedrock_guardrail", lambda prompt: None)
    monkeypatch.setattr(
        agentdemo,
        "classify_user_prompt",
        lambda prompt: InputClassification(False, "unsupported_request", "no", "declined"),
    )

    assert asyncio.run(agentdemo.run_agent_demo()) == 0

    output = capsys.readouterr().out
    assert "Source: local_regex" in output
    assert "declined" in output
    assert "Tool Results:" in output


def test_run_agent_demo_invokes_agent_for_allowed_prompt(monkeypatch, capsys):
    monkeypatch.setattr(agentdemo, "parse_args", lambda: argparse.Namespace(prompt="calculate", model="model"))
    monkeypatch.setattr(agentdemo, "evaluate_prompt_with_bedrock_guardrail", lambda prompt: None)
    monkeypatch.setattr(
        agentdemo,
        "classify_user_prompt",
        lambda prompt: InputClassification(True, "calculator", "ok"),
    )
    async def fake_create_agent_runner(model):
        return FakeAgent()

    monkeypatch.setattr(agentdemo, "create_agent_runner", fake_create_agent_runner)

    assert asyncio.run(agentdemo.run_agent_demo()) == 0

    output = capsys.readouterr().out
    assert "1. calculator" in output
    assert "Tool Results:" in output
    assert "done" in output


def test_run_agent_demo_returns_error_code_on_exception(monkeypatch, capsys):
    monkeypatch.setattr(agentdemo, "parse_args", lambda: argparse.Namespace(prompt="hello", model="model"))
    monkeypatch.setattr(
        agentdemo,
        "evaluate_prompt_with_bedrock_guardrail",
        lambda prompt: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert asyncio.run(agentdemo.run_agent_demo()) == 1

    assert "boom" in capsys.readouterr().out


def test_run_agent_demo_prints_friendly_token_error(monkeypatch, capsys):
    monkeypatch.setattr(agentdemo, "parse_args", lambda: argparse.Namespace(prompt="hello", model="model"))
    monkeypatch.setattr(agentdemo, "evaluate_prompt_with_bedrock_guardrail", lambda prompt: None)
    monkeypatch.setattr(
        agentdemo,
        "classify_user_prompt",
        lambda prompt: InputClassification(True, "image_search", "ok"),
    )

    async def fake_create_agent_runner(_model):
        raise RuntimeError("Error code: 401 - invalid_api_key Signature expired")

    monkeypatch.setattr(agentdemo, "create_agent_runner", fake_create_agent_runner)

    assert asyncio.run(agentdemo.run_agent_demo()) == 1

    output = capsys.readouterr().out
    assert agentdemo.BEDROCK_TOKEN_ERROR_MESSAGE in output
    assert "Raw error:" in output


def test_main_runs_async_demo(monkeypatch):
    monkeypatch.setattr(agentdemo, "run_agent_demo", lambda: "coroutine")
    monkeypatch.setattr(agentdemo.asyncio, "run", lambda coroutine: 7)

    assert agentdemo.main() == 7


class FakeChatOpenAI:
    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self


class FakeMessage:
    def __init__(
        self,
        tool_calls=None,
        additional_kwargs=None,
        content="",
        message_type="",
        name="",
        tool_call_id="",
    ):
        self.tool_calls = tool_calls
        self.additional_kwargs = additional_kwargs
        self.content = content
        self.type = message_type
        self.name = name
        self.tool_call_id = tool_call_id


class FakeAgent:
    async def ainvoke(self, payload):
        assert payload == {"messages": [{"role": "user", "content": "calculate"}]}
        return {
            "messages": [
                FakeMessage(
                    tool_calls=[
                        {"id": "1", "name": "calculator", "args": {"expression": "1+1"}}
                    ]
                ),
                FakeMessage(content="done"),
            ]
        }


class FakeCreateAgent:
    def __init__(self):
        self.agent = object()
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self.agent


class FakeMultiServerMCPClient:
    def __call__(self, servers, tool_name_prefix):
        self.servers = servers
        self.tool_name_prefix = tool_name_prefix
        return self

    async def get_tools(self):
        return ["mcp-tools"]
